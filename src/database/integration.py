"""
Integration module for connecting PDF processing with Knowledge Base storage
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from .knowledge_base import KnowledgeBase
from .connection import get_database_connection

logger = logging.getLogger(__name__)


class PDFProcessingIntegration:
    """
    Integration layer between PDF processing and Knowledge Base storage
    """
    
    def __init__(self, knowledge_base: Optional[KnowledgeBase] = None):
        """
        Initialize integration
        
        Args:
            knowledge_base: Knowledge Base instance
        """
        self.kb = knowledge_base or KnowledgeBase()
    
    def store_processed_document(self, 
                                result_file_path: str,
                                pdf_file_path: str) -> int:
        """
        Store a processed document from JSON result file
        
        Args:
            result_file_path: Path to JSON result file
            pdf_file_path: Path to original PDF file
            
        Returns:
            Document ID in knowledge base
        """
        try:
            # Load processing results
            with open(result_file_path, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Extract required data
            filename = Path(pdf_file_path).name
            document_data = result_data.get('document_info', {})
            legal_analysis = result_data.get('legal_analysis', {})
            text_analysis = result_data.get('text_analysis', {})
            raw_text = result_data.get('extracted_text', '')
            
            # Check if document was processed with OCR
            is_scanned = result_data.get('is_scanned', False)
            ocr_confidence = result_data.get('ocr_confidence')
            
            # Store in knowledge base
            document_id = self.kb.store_document(
                filename=filename,
                file_path=pdf_file_path,
                document_data=document_data,
                legal_analysis=legal_analysis,
                text_analysis=text_analysis,
                raw_text=raw_text,
                is_scanned=is_scanned,
                ocr_confidence=ocr_confidence
            )
            
            logger.info(f"Stored document {filename} with ID {document_id}")
            return document_id
            
        except Exception as e:
            logger.error(f"Error storing processed document {result_file_path}: {e}")
            raise
    
    def batch_store_from_directory(self, 
                                  results_directory: str,
                                  pdf_directory: str) -> List[int]:
        """
        Batch store all processed documents from a directory
        
        Args:
            results_directory: Directory containing JSON result files
            pdf_directory: Directory containing original PDF files
            
        Returns:
            List of document IDs
        """
        try:
            results_dir = Path(results_directory)
            pdf_dir = Path(pdf_directory)
            
            if not results_dir.exists():
                raise FileNotFoundError(f"Results directory not found: {results_directory}")
            
            if not pdf_dir.exists():
                raise FileNotFoundError(f"PDF directory not found: {pdf_directory}")
            
            document_ids = []
            json_files = list(results_dir.glob('*.json'))
            
            logger.info(f"Found {len(json_files)} JSON result files to process")
            
            for json_file in json_files:
                try:
                    # Find corresponding PDF file
                    pdf_name = json_file.stem + '.pdf'
                    pdf_file = pdf_dir / pdf_name
                    
                    if not pdf_file.exists():
                        logger.warning(f"PDF file not found for {json_file.name}: {pdf_file}")
                        continue
                    
                    # Store document
                    document_id = self.store_processed_document(
                        str(json_file), str(pdf_file)
                    )
                    document_ids.append(document_id)
                    
                except Exception as e:
                    logger.error(f"Error processing {json_file.name}: {e}")
                    continue
            
            logger.info(f"Successfully stored {len(document_ids)} documents")
            return document_ids
            
        except Exception as e:
            logger.error(f"Error in batch store operation: {e}")
            raise
    
    def store_comprehensive_test_results(self, 
                                       test_results_file: str,
                                       pdf_directory: str) -> Dict[str, Any]:
        """
        Store results from comprehensive test summary
        
        Args:
            test_results_file: Path to comprehensive test summary JSON
            pdf_directory: Directory containing PDF files
            
        Returns:
            Storage summary
        """
        try:
            # Load comprehensive test results
            with open(test_results_file, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
            
            pdf_dir = Path(pdf_directory)
            stored_documents = []
            failed_documents = []
            
            # Process detailed results
            detailed_results = test_data.get('detailed_results', {})
            
            for filename, result_data in detailed_results.items():
                try:
                    pdf_file = pdf_dir / filename
                    
                    if not pdf_file.exists():
                        logger.warning(f"PDF file not found: {pdf_file}")
                        failed_documents.append({
                            'filename': filename,
                            'error': 'PDF file not found'
                        })
                        continue
                    
                    # Prepare data for storage
                    document_data = {
                        'processing_time': result_data.get('processing_time', 0),
                        'confidence_score': result_data.get('confidence_score', 0.0)
                    }
                    
                    legal_analysis = result_data.get('legal_analysis', {})
                    text_analysis = result_data.get('text_analysis', {})
                    raw_text = result_data.get('extracted_text', '')
                    is_scanned = result_data.get('is_scanned', False)
                    ocr_confidence = result_data.get('ocr_confidence')
                    
                    # Store document
                    document_id = self.kb.store_document(
                        filename=filename,
                        file_path=str(pdf_file),
                        document_data=document_data,
                        legal_analysis=legal_analysis,
                        text_analysis=text_analysis,
                        raw_text=raw_text,
                        is_scanned=is_scanned,
                        ocr_confidence=ocr_confidence
                    )
                    
                    stored_documents.append({
                        'filename': filename,
                        'document_id': document_id,
                        'processing_time': document_data['processing_time'],
                        'confidence_score': document_data['confidence_score']
                    })
                    
                except Exception as e:
                    logger.error(f"Error storing document {filename}: {e}")
                    failed_documents.append({
                        'filename': filename,
                        'error': str(e)
                    })
            
            # Create storage summary
            summary = {
                'total_processed': len(detailed_results),
                'successfully_stored': len(stored_documents),
                'failed_storage': len(failed_documents),
                'success_rate': len(stored_documents) / len(detailed_results) * 100 if detailed_results else 0,
                'stored_documents': stored_documents,
                'failed_documents': failed_documents,
                'storage_timestamp': datetime.now().isoformat(),
                'test_summary': test_data.get('test_summary', {}),
                'overall_statistics': test_data.get('overall_statistics', {})
            }
            
            logger.info(f"Storage completed: {summary['successfully_stored']}/{summary['total_processed']} documents stored")
            return summary
            
        except Exception as e:
            logger.error(f"Error storing comprehensive test results: {e}")
            raise
    
    def export_knowledge_base_summary(self, output_file: str) -> Dict[str, Any]:
        """
        Export Knowledge Base summary to JSON file
        
        Args:
            output_file: Output file path
            
        Returns:
            Export summary
        """
        try:
            # Get Knowledge Base statistics
            stats = self.kb.get_statistics()
            
            # Get recent documents
            recent_docs = self.kb.search_documents(limit=10)
            
            # Create export data
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'knowledge_base_statistics': stats,
                'recent_documents': recent_docs,
                'export_summary': {
                    'total_documents': stats['documents']['total'],
                    'total_entities': stats['entities']['total'],
                    'document_types': stats['documents']['by_type'],
                    'entity_types': stats['entities']['by_type']
                }
            }
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Knowledge Base summary exported to {output_file}")
            return export_data
            
        except Exception as e:
            logger.error(f"Error exporting Knowledge Base summary: {e}")
            raise
    
    def validate_stored_documents(self) -> Dict[str, Any]:
        """
        Validate all stored documents in Knowledge Base
        
        Returns:
            Validation report
        """
        try:
            # Get all documents
            all_docs = self.kb.search_documents(limit=1000)
            
            validation_results = {
                'total_documents': len(all_docs),
                'valid_documents': 0,
                'invalid_documents': 0,
                'validation_errors': [],
                'validation_timestamp': datetime.now().isoformat()
            }
            
            for doc in all_docs:
                try:
                    # Get full document details
                    full_doc = self.kb.get_document_by_id(doc['id'])
                    
                    if not full_doc:
                        validation_results['validation_errors'].append({
                            'document_id': doc['id'],
                            'error': 'Document not found'
                        })
                        validation_results['invalid_documents'] += 1
                        continue
                    
                    # Validate required fields
                    required_fields = ['filename', 'raw_text', 'legal_analysis']
                    missing_fields = [
                        field for field in required_fields 
                        if not full_doc.get(field)
                    ]
                    
                    if missing_fields:
                        validation_results['validation_errors'].append({
                            'document_id': doc['id'],
                            'filename': doc['filename'],
                            'error': f'Missing required fields: {missing_fields}'
                        })
                        validation_results['invalid_documents'] += 1
                    else:
                        validation_results['valid_documents'] += 1
                
                except Exception as e:
                    validation_results['validation_errors'].append({
                        'document_id': doc['id'],
                        'error': f'Validation error: {str(e)}'
                    })
                    validation_results['invalid_documents'] += 1
            
            # Calculate validation rate
            if validation_results['total_documents'] > 0:
                validation_results['validation_rate'] = (
                    validation_results['valid_documents'] / 
                    validation_results['total_documents'] * 100
                )
            else:
                validation_results['validation_rate'] = 0
            
            logger.info(f"Validation completed: {validation_results['valid_documents']}/{validation_results['total_documents']} documents valid")
            return validation_results
            
        except Exception as e:
            logger.error(f"Error validating stored documents: {e}")
            raise


def create_integration_instance() -> PDFProcessingIntegration:
    """Create and return a PDF processing integration instance"""
    try:
        db = get_database_connection()
        kb = KnowledgeBase(db)
        return PDFProcessingIntegration(kb)
    except Exception as e:
        logger.error(f"Error creating integration instance: {e}")
        raise


# Convenience functions for common operations
def store_comprehensive_test_results(test_results_file: str, pdf_directory: str) -> Dict[str, Any]:
    """Convenience function to store comprehensive test results"""
    integration = create_integration_instance()
    return integration.store_comprehensive_test_results(test_results_file, pdf_directory)


def batch_store_documents(results_directory: str, pdf_directory: str) -> List[int]:
    """Convenience function to batch store documents"""
    integration = create_integration_instance()
    return integration.batch_store_from_directory(results_directory, pdf_directory)


def export_knowledge_base(output_file: str) -> Dict[str, Any]:
    """Convenience function to export Knowledge Base summary"""
    integration = create_integration_instance()
    return integration.export_knowledge_base_summary(output_file)