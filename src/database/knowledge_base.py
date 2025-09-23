"""
Knowledge Base implementation for Legal Document storage and retrieval
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, func, desc, asc
from sqlalchemy.exc import IntegrityError

from .models import (
    Document, LegalEntity, Person, Organization, LegalReference, Case,
    DocumentType, EntityType, ProcessingStatus, SearchIndex, ProcessingLog,
    document_entity_association, entity_relationship_association
)
from .connection import DatabaseConnection, get_database_connection

logger = logging.getLogger(__name__)


class KnowledgeBase:
    """
    Legal Document Knowledge Base for storing and retrieving legal information
    """
    
    def __init__(self, db_connection: Optional[DatabaseConnection] = None):
        """
        Initialize Knowledge Base
        
        Args:
            db_connection: Database connection instance
        """
        self.db = db_connection or get_database_connection()
    
    def store_document(self, 
                      filename: str,
                      file_path: str,
                      document_data: Dict[str, Any],
                      legal_analysis: Dict[str, Any],
                      text_analysis: Dict[str, Any],
                      raw_text: str,
                      is_scanned: bool = False,
                      ocr_confidence: Optional[float] = None) -> int:
        """
        Store a processed legal document in the knowledge base
        
        Args:
            filename: Document filename
            file_path: Full file path
            document_data: Document metadata
            legal_analysis: Legal analysis results
            text_analysis: Text analysis results
            raw_text: Extracted text content
            is_scanned: Whether document was scanned
            ocr_confidence: OCR confidence score if applicable
            
        Returns:
            Document ID
        """
        try:
            with self.db.get_session() as session:
                # Calculate file hash for deduplication
                file_hash = self._calculate_file_hash(file_path)
                
                # Check if document already exists
                existing_doc = session.query(Document).filter_by(file_hash=file_hash).first()
                if existing_doc:
                    logger.info(f"Document {filename} already exists with ID {existing_doc.id}")
                    return existing_doc.id
                
                # Extract document metadata
                doc_info = legal_analysis.get('document_info', {})
                
                # Create document record
                document = Document(
                    filename=filename,
                    file_path=file_path,
                    file_hash=file_hash,
                    document_type=self._map_document_type(doc_info.get('document_type')),
                    case_number=doc_info.get('case_number'),
                    court_name=doc_info.get('court_name'),
                    date_issued=self._parse_date(doc_info.get('date_issued')),
                    processing_status=ProcessingStatus.COMPLETED,
                    processing_start=datetime.now(),
                    processing_end=datetime.now(),
                    processing_time=document_data.get('processing_time', 0),
                    raw_text=raw_text,
                    text_length=len(raw_text),
                    word_count=text_analysis.get('word_count', 0),
                    sentence_count=text_analysis.get('sentence_count', 0),
                    legal_analysis=legal_analysis,
                    text_analysis=text_analysis,
                    confidence_score=document_data.get('confidence_score', 0.0),
                    is_scanned=is_scanned,
                    ocr_confidence=ocr_confidence
                )
                
                session.add(document)
                session.flush()  # Get document ID
                
                # Store entities
                self._store_entities(session, document.id, legal_analysis)
                
                # Store case information
                self._store_case_info(session, document.id, legal_analysis)
                
                # Create search index
                self._create_search_index(session, document)
                
                logger.info(f"Successfully stored document {filename} with ID {document.id}")
                return document.id
                
        except Exception as e:
            logger.error(f"Error storing document {filename}: {e}")
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file for deduplication"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.warning(f"Could not calculate hash for {file_path}: {e}")
            return hashlib.sha256(file_path.encode()).hexdigest()
    
    def _map_document_type(self, doc_type: str) -> DocumentType:
        """Map document type string to enum"""
        type_mapping = {
            'Bản án hình sự': DocumentType.CRIMINAL_JUDGMENT,
            'Bản án dân sự': DocumentType.CIVIL_JUDGMENT,
            'Quyết định thi hành án': DocumentType.EXECUTION_DECISION,
            'Quyết định': DocumentType.COURT_DECISION
        }
        return type_mapping.get(doc_type, DocumentType.OTHER)
    
    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """Parse date string to datetime object"""
        if not date_str:
            return None
        
        # Try different date formats
        formats = ['%d/%m/%Y', '%Y-%m-%d', '%d-%m-%Y']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        
        logger.warning(f"Could not parse date: {date_str}")
        return None
    
    def _store_entities(self, session: Session, document_id: int, legal_analysis: Dict[str, Any]):
        """Store extracted entities and link to document"""
        entities_data = legal_analysis.get('entities', {})
        
        for entity_type, entity_list in entities_data.items():
            if not isinstance(entity_list, list):
                continue
                
            mapped_type = self._map_entity_type(entity_type)
            if not mapped_type:
                continue
            
            for entity_name in entity_list:
                if not entity_name or not entity_name.strip():
                    continue
                
                # Get or create entity
                entity = self._get_or_create_entity(
                    session, mapped_type, entity_name.strip()
                )
                
                # Link entity to document
                self._link_entity_to_document(
                    session, document_id, entity.id, confidence_score=0.8
                )
                
                # Store specialized entity data
                if mapped_type == EntityType.PERSON:
                    self._store_person_details(session, entity.id, entity_name)
                elif mapped_type == EntityType.ORGANIZATION:
                    self._store_organization_details(session, entity.id, entity_name)
                elif mapped_type == EntityType.LEGAL_REFERENCE:
                    self._store_legal_reference_details(session, entity.id, entity_name)
    
    def _map_entity_type(self, entity_type: str) -> Optional[EntityType]:
        """Map entity type string to enum"""
        type_mapping = {
            'persons': EntityType.PERSON,
            'organizations': EntityType.ORGANIZATION,
            'locations': EntityType.LOCATION,
            'dates': EntityType.DATE,
            'amounts': EntityType.AMOUNT,
            'case_numbers': EntityType.CASE_NUMBER,
            'legal_references': EntityType.LEGAL_REFERENCE,
            'charges': EntityType.CHARGE
        }
        return type_mapping.get(entity_type)
    
    def _get_or_create_entity(self, session: Session, entity_type: EntityType, name: str) -> LegalEntity:
        """Get existing entity or create new one"""
        normalized_name = self._normalize_entity_name(name)
        
        # Try to find existing entity
        entity = session.query(LegalEntity).filter(
            and_(
                LegalEntity.entity_type == entity_type,
                LegalEntity.normalized_name == normalized_name
            )
        ).first()
        
        if entity:
            # Update frequency
            entity.frequency += 1
            return entity
        
        # Create new entity
        entity = LegalEntity(
            entity_type=entity_type,
            name=name,
            normalized_name=normalized_name,
            confidence_score=0.8,
            frequency=1
        )
        session.add(entity)
        session.flush()
        return entity
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for matching"""
        import re
        # Remove extra whitespace, convert to lowercase
        normalized = re.sub(r'\s+', ' ', name.strip().lower())
        # Remove common prefixes/suffixes for better matching
        normalized = re.sub(r'^(ông|bà|anh|chị|em)\s+', '', normalized)
        return normalized
    
    def _link_entity_to_document(self, session: Session, document_id: int, entity_id: int, 
                                confidence_score: float = 0.8, context: str = None):
        """Link entity to document with confidence score"""
        try:
            # Check if link already exists
            existing = session.execute(
                document_entity_association.select().where(
                    and_(
                        document_entity_association.c.document_id == document_id,
                        document_entity_association.c.entity_id == entity_id
                    )
                )
            ).first()
            
            if not existing:
                session.execute(
                    document_entity_association.insert().values(
                        document_id=document_id,
                        entity_id=entity_id,
                        confidence_score=confidence_score,
                        context=context
                    )
                )
        except IntegrityError:
            # Link already exists
            pass
    
    def _store_person_details(self, session: Session, entity_id: int, name: str):
        """Store detailed person information"""
        # Check if person details already exist
        existing = session.query(Person).filter_by(entity_id=entity_id).first()
        if existing:
            return
        
        # Parse name and create person record
        person = Person(
            entity_id=entity_id,
            full_name=name
        )
        session.add(person)
    
    def _store_organization_details(self, session: Session, entity_id: int, name: str):
        """Store detailed organization information"""
        existing = session.query(Organization).filter_by(entity_id=entity_id).first()
        if existing:
            return
        
        organization = Organization(
            entity_id=entity_id,
            name=name
        )
        session.add(organization)
    
    def _store_legal_reference_details(self, session: Session, entity_id: int, reference: str):
        """Store detailed legal reference information"""
        existing = session.query(LegalReference).filter_by(entity_id=entity_id).first()
        if existing:
            existing.citation_count += 1
            return
        
        # Parse legal reference
        import re
        law_match = re.search(r'(Bộ luật|Luật|Nghị định|Thông tư)\s+([^,]+)', reference)
        article_match = re.search(r'Điều\s+(\d+)', reference)
        
        legal_ref = LegalReference(
            entity_id=entity_id,
            law_name=law_match.group(2) if law_match else reference,
            article_number=article_match.group(1) if article_match else None,
            full_reference=reference,
            citation_count=1
        )
        session.add(legal_ref)
    
    def _store_case_info(self, session: Session, document_id: int, legal_analysis: Dict[str, Any]):
        """Store case information from legal analysis"""
        doc_info = legal_analysis.get('document_info', {})
        
        if not doc_info.get('case_number'):
            return
        
        case = Case(
            document_id=document_id,
            case_number=doc_info.get('case_number'),
            case_type=self._determine_case_type(doc_info.get('document_type')),
            court_name=doc_info.get('court_name'),
            decision_date=self._parse_date(doc_info.get('date_issued')),
            defendants=doc_info.get('parties', {}).get('defendants', []),
            plaintiffs=doc_info.get('parties', {}).get('plaintiffs', []),
            judges=doc_info.get('parties', {}).get('judges', []),
            prosecutors=doc_info.get('parties', {}).get('prosecutors', []),
            charges=doc_info.get('charges', []),
            verdict=doc_info.get('verdict'),
            sentence=doc_info.get('sentence'),
            legal_references=doc_info.get('legal_references', []),
            case_status='Closed'
        )
        session.add(case)
    
    def _determine_case_type(self, document_type: str) -> str:
        """Determine case type from document type"""
        if 'hình sự' in document_type.lower():
            return 'Criminal'
        elif 'dân sự' in document_type.lower():
            return 'Civil'
        else:
            return 'Administrative'
    
    def _create_search_index(self, session: Session, document: Document):
        """Create search index for document"""
        # Create document search index
        search_content = f"{document.filename} {document.raw_text}"
        keywords = self._extract_keywords(document.raw_text)
        
        search_index = SearchIndex(
            document_id=document.id,
            content=search_content,
            content_type='document',
            keywords=keywords,
            relevance_score=document.confidence_score
        )
        session.add(search_index)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text for search indexing"""
        import re
        # Simple keyword extraction - can be enhanced with NLP
        words = re.findall(r'\b\w{3,}\b', text.lower())
        # Remove common Vietnamese stop words
        stop_words = {'của', 'và', 'với', 'trong', 'trên', 'dưới', 'theo', 'về', 'cho', 'từ', 'đến'}
        keywords = [word for word in set(words) if word not in stop_words]
        return keywords[:50]  # Limit to top 50 keywords
    
    def search_documents(self, 
                        query: str = None,
                        document_type: DocumentType = None,
                        case_number: str = None,
                        court_name: str = None,
                        date_from: datetime = None,
                        date_to: datetime = None,
                        limit: int = 50,
                        offset: int = 0) -> List[Dict[str, Any]]:
        """
        Search documents in the knowledge base
        
        Args:
            query: Text search query
            document_type: Filter by document type
            case_number: Filter by case number
            court_name: Filter by court name
            date_from: Filter by date range start
            date_to: Filter by date range end
            limit: Maximum number of results
            offset: Result offset for pagination
            
        Returns:
            List of matching documents
        """
        try:
            with self.db.get_session() as session:
                query_obj = session.query(Document)
                
                # Apply filters
                if document_type:
                    query_obj = query_obj.filter(Document.document_type == document_type)
                
                if case_number:
                    query_obj = query_obj.filter(Document.case_number.ilike(f'%{case_number}%'))
                
                if court_name:
                    query_obj = query_obj.filter(Document.court_name.ilike(f'%{court_name}%'))
                
                if date_from:
                    query_obj = query_obj.filter(Document.date_issued >= date_from)
                
                if date_to:
                    query_obj = query_obj.filter(Document.date_issued <= date_to)
                
                if query:
                    query_obj = query_obj.filter(Document.raw_text.ilike(f'%{query}%'))
                
                # Order by relevance and date
                query_obj = query_obj.order_by(desc(Document.confidence_score), desc(Document.created_at))
                
                # Apply pagination
                documents = query_obj.offset(offset).limit(limit).all()
                
                # Convert to dictionaries
                results = []
                for doc in documents:
                    results.append({
                        'id': doc.id,
                        'filename': doc.filename,
                        'document_type': doc.document_type.value if doc.document_type else None,
                        'case_number': doc.case_number,
                        'court_name': doc.court_name,
                        'date_issued': doc.date_issued.isoformat() if doc.date_issued else None,
                        'confidence_score': doc.confidence_score,
                        'text_length': doc.text_length,
                        'word_count': doc.word_count,
                        'created_at': doc.created_at.isoformat(),
                        'summary': doc.raw_text[:500] + '...' if len(doc.raw_text) > 500 else doc.raw_text
                    })
                
                return results
                
        except Exception as e:
            logger.error(f"Error searching documents: {e}")
            raise
    
    def get_document_by_id(self, document_id: int) -> Optional[Dict[str, Any]]:
        """Get document by ID with full details"""
        try:
            with self.db.get_session() as session:
                document = session.query(Document).filter_by(id=document_id).first()
                
                if not document:
                    return None
                
                # Get related entities
                entities = session.query(LegalEntity).join(
                    document_entity_association
                ).filter(
                    document_entity_association.c.document_id == document_id
                ).all()
                
                # Get case information
                case = session.query(Case).filter_by(document_id=document_id).first()
                
                return {
                    'id': document.id,
                    'filename': document.filename,
                    'file_path': document.file_path,
                    'document_type': document.document_type.value if document.document_type else None,
                    'case_number': document.case_number,
                    'court_name': document.court_name,
                    'date_issued': document.date_issued.isoformat() if document.date_issued else None,
                    'processing_time': document.processing_time,
                    'confidence_score': document.confidence_score,
                    'text_length': document.text_length,
                    'word_count': document.word_count,
                    'sentence_count': document.sentence_count,
                    'is_scanned': document.is_scanned,
                    'ocr_confidence': document.ocr_confidence,
                    'raw_text': document.raw_text,
                    'legal_analysis': document.legal_analysis,
                    'text_analysis': document.text_analysis,
                    'entities': [
                        {
                            'id': entity.id,
                            'type': entity.entity_type.value,
                            'name': entity.name,
                            'frequency': entity.frequency,
                            'confidence': entity.confidence_score
                        }
                        for entity in entities
                    ],
                    'case_info': {
                        'case_number': case.case_number,
                        'case_type': case.case_type,
                        'court_name': case.court_name,
                        'decision_date': case.decision_date.isoformat() if case.decision_date else None,
                        'defendants': case.defendants,
                        'plaintiffs': case.plaintiffs,
                        'judges': case.judges,
                        'charges': case.charges,
                        'verdict': case.verdict,
                        'sentence': case.sentence
                    } if case else None,
                    'created_at': document.created_at.isoformat(),
                    'updated_at': document.updated_at.isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error getting document {document_id}: {e}")
            raise
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get knowledge base statistics"""
        try:
            with self.db.get_session() as session:
                # Document statistics
                total_docs = session.query(func.count(Document.id)).scalar()
                doc_types = session.query(
                    Document.document_type, func.count(Document.id)
                ).group_by(Document.document_type).all()
                
                # Entity statistics
                total_entities = session.query(func.count(LegalEntity.id)).scalar()
                entity_types = session.query(
                    LegalEntity.entity_type, func.count(LegalEntity.id)
                ).group_by(LegalEntity.entity_type).all()
                
                # Processing statistics
                avg_processing_time = session.query(
                    func.avg(Document.processing_time)
                ).scalar()
                
                avg_confidence = session.query(
                    func.avg(Document.confidence_score)
                ).scalar()
                
                # Text statistics
                total_text_length = session.query(
                    func.sum(Document.text_length)
                ).scalar()
                
                avg_text_length = session.query(
                    func.avg(Document.text_length)
                ).scalar()
                
                return {
                    'documents': {
                        'total': total_docs,
                        'by_type': {dt.value: count for dt, count in doc_types},
                        'avg_processing_time': float(avg_processing_time) if avg_processing_time else 0,
                        'avg_confidence_score': float(avg_confidence) if avg_confidence else 0,
                        'total_text_length': int(total_text_length) if total_text_length else 0,
                        'avg_text_length': float(avg_text_length) if avg_text_length else 0
                    },
                    'entities': {
                        'total': total_entities,
                        'by_type': {et.value: count for et, count in entity_types}
                    },
                    'database_info': self.db.get_database_info()
                }
                
        except Exception as e:
            logger.error(f"Error getting statistics: {e}")
            raise