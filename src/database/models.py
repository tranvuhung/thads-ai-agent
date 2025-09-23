"""
Database models for Legal Document Knowledge Base

This module defines the database schema for storing legal documents,
entities, and their relationships extracted from PDF processing.
"""

from sqlalchemy import (
    Column, Integer, String, Text, DateTime, Float, Boolean, 
    ForeignKey, Table, JSON, Enum, Index
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
import enum

Base = declarative_base()

# Association table for many-to-many relationship between documents and entities
document_entity_association = Table(
    'document_entities',
    Base.metadata,
    Column('document_id', Integer, ForeignKey('documents.id'), primary_key=True),
    Column('entity_id', Integer, ForeignKey('legal_entities.id'), primary_key=True),
    Column('confidence_score', Float, default=0.0),
    Column('context', Text),  # Context where entity appears in document
    Column('created_at', DateTime, default=func.now())
)

# Association table for entity relationships
entity_relationship_association = Table(
    'entity_relationships',
    Base.metadata,
    Column('source_entity_id', Integer, ForeignKey('legal_entities.id'), primary_key=True),
    Column('target_entity_id', Integer, ForeignKey('legal_entities.id'), primary_key=True),
    Column('relationship_type', String(100)),  # e.g., 'defendant_in_case', 'judge_of_case'
    Column('confidence_score', Float, default=0.0),
    Column('created_at', DateTime, default=func.now())
)


class DocumentType(enum.Enum):
    """Document type enumeration"""
    CRIMINAL_JUDGMENT = "Bản án hình sự"
    CIVIL_JUDGMENT = "Bản án dân sự"
    EXECUTION_DECISION = "Quyết định thi hành án"
    COURT_DECISION = "Quyết định"
    OTHER = "Khác"


class EntityType(enum.Enum):
    """Legal entity type enumeration"""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    AMOUNT = "amount"
    CASE_NUMBER = "case_number"
    LEGAL_REFERENCE = "legal_reference"
    CHARGE = "charge"
    VERDICT = "verdict"
    SENTENCE = "sentence"


class ProcessingStatus(enum.Enum):
    """Document processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class Document(Base):
    """Main document table storing legal document metadata and content"""
    __tablename__ = 'documents'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False, index=True)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), unique=True, index=True)  # SHA-256 hash for deduplication
    
    # Document metadata
    document_type = Column(Enum(DocumentType), nullable=False, index=True)
    case_number = Column(String(100), index=True)
    court_name = Column(String(255))
    date_issued = Column(DateTime)
    
    # Processing information
    processing_status = Column(Enum(ProcessingStatus), default=ProcessingStatus.PENDING, index=True)
    processing_start = Column(DateTime)
    processing_end = Column(DateTime)
    processing_time = Column(Float)  # seconds
    error_message = Column(Text)
    
    # Content and analysis
    raw_text = Column(Text)
    text_length = Column(Integer)
    word_count = Column(Integer)
    sentence_count = Column(Integer)
    
    # Analysis results
    legal_analysis = Column(JSON)  # Store complete legal analysis JSON
    text_analysis = Column(JSON)   # Store text analysis metrics
    confidence_score = Column(Float, default=0.0)
    
    # OCR information
    is_scanned = Column(Boolean, default=False)
    ocr_confidence = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    entities = relationship(
        "LegalEntity",
        secondary=document_entity_association,
        back_populates="documents"
    )
    cases = relationship("Case", back_populates="document")
    
    # Indexes for performance
    __table_args__ = (
        Index('idx_document_type_date', 'document_type', 'date_issued'),
        Index('idx_processing_status_created', 'processing_status', 'created_at'),
        Index('idx_case_number_court', 'case_number', 'court_name'),
    )


class LegalEntity(Base):
    """Table for storing extracted legal entities"""
    __tablename__ = 'legal_entities'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(Enum(EntityType), nullable=False, index=True)
    name = Column(String(500), nullable=False, index=True)
    normalized_name = Column(String(500), index=True)  # Normalized for matching
    
    # Entity-specific attributes stored as JSON
    attributes = Column(JSON)  # Flexible storage for entity-specific data
    
    # Confidence and frequency
    confidence_score = Column(Float, default=0.0)
    frequency = Column(Integer, default=1)  # How many times this entity appears
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    documents = relationship(
        "Document",
        secondary=document_entity_association,
        back_populates="entities"
    )
    
    # Self-referential relationships for entity connections
    source_relationships = relationship(
        "LegalEntity",
        secondary=entity_relationship_association,
        primaryjoin=id == entity_relationship_association.c.source_entity_id,
        secondaryjoin=id == entity_relationship_association.c.target_entity_id,
        back_populates="target_relationships"
    )
    target_relationships = relationship(
        "LegalEntity",
        secondary=entity_relationship_association,
        primaryjoin=id == entity_relationship_association.c.target_entity_id,
        secondaryjoin=id == entity_relationship_association.c.source_entity_id,
        back_populates="source_relationships"
    )
    
    # Indexes
    __table_args__ = (
        Index('idx_entity_type_name', 'entity_type', 'name'),
        Index('idx_normalized_name', 'normalized_name'),
        Index('idx_entity_frequency', 'frequency'),
    )


class Person(Base):
    """Specialized table for person entities with detailed attributes"""
    __tablename__ = 'persons'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey('legal_entities.id'), unique=True)
    
    # Personal information
    full_name = Column(String(255), nullable=False, index=True)
    birth_year = Column(Integer)
    gender = Column(String(10))
    nationality = Column(String(100))
    ethnicity = Column(String(100))
    religion = Column(String(100))
    
    # Address information
    address = Column(Text)
    ward = Column(String(100))
    district = Column(String(100))
    city = Column(String(100))
    
    # Professional information
    occupation = Column(String(255))
    education_level = Column(String(100))
    
    # Legal information
    id_number = Column(String(50), index=True)
    criminal_record = Column(JSON)  # Store criminal history
    
    # Relationships in legal context
    roles = Column(JSON)  # Roles in different cases (defendant, plaintiff, judge, etc.)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship to base entity
    entity = relationship("LegalEntity", backref="person_details")


class Organization(Base):
    """Specialized table for organization entities"""
    __tablename__ = 'organizations'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey('legal_entities.id'), unique=True)
    
    # Organization information
    name = Column(String(500), nullable=False, index=True)
    organization_type = Column(String(100))  # Court, Police, Company, etc.
    registration_number = Column(String(100), index=True)
    
    # Address information
    address = Column(Text)
    district = Column(String(100))
    city = Column(String(100))
    
    # Contact information
    phone = Column(String(50))
    email = Column(String(255))
    website = Column(String(255))
    
    # Legal information
    legal_representative = Column(String(255))
    establishment_date = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship to base entity
    entity = relationship("LegalEntity", backref="organization_details")


class LegalReference(Base):
    """Table for storing legal references (laws, articles, etc.)"""
    __tablename__ = 'legal_references'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_id = Column(Integer, ForeignKey('legal_entities.id'), unique=True)
    
    # Reference information
    law_name = Column(String(500), nullable=False, index=True)
    article_number = Column(String(50))
    section_number = Column(String(50))
    paragraph_number = Column(String(50))
    point_letter = Column(String(10))
    
    # Full reference text
    full_reference = Column(Text)
    year_enacted = Column(Integer)
    year_amended = Column(Integer)
    
    # Usage statistics
    citation_count = Column(Integer, default=1)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationship to base entity
    entity = relationship("LegalEntity", backref="legal_reference_details")
    
    # Indexes
    __table_args__ = (
        Index('idx_law_article', 'law_name', 'article_number'),
        Index('idx_citation_count', 'citation_count'),
    )


class Case(Base):
    """Table for storing case information extracted from documents"""
    __tablename__ = 'cases'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    
    # Case identification
    case_number = Column(String(100), nullable=False, index=True)
    case_type = Column(String(100))  # Criminal, Civil, Administrative
    
    # Court information
    court_name = Column(String(255))
    court_level = Column(String(100))  # First instance, Appeal, etc.
    
    # Case timeline
    filing_date = Column(DateTime)
    hearing_date = Column(DateTime)
    decision_date = Column(DateTime)
    
    # Case parties (stored as JSON for flexibility)
    defendants = Column(JSON)
    plaintiffs = Column(JSON)
    judges = Column(JSON)
    prosecutors = Column(JSON)
    lawyers = Column(JSON)
    
    # Case details
    charges = Column(JSON)
    verdict = Column(Text)
    sentence = Column(Text)
    fine_amount = Column(Float)
    
    # Legal references used in case
    legal_references = Column(JSON)
    
    # Case outcome
    case_status = Column(String(100))  # Closed, Appealed, etc.
    appeal_deadline = Column(DateTime)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    document = relationship("Document", back_populates="cases")
    
    # Indexes
    __table_args__ = (
        Index('idx_case_number_type', 'case_number', 'case_type'),
        Index('idx_court_date', 'court_name', 'decision_date'),
        Index('idx_case_status', 'case_status'),
    )


class SearchIndex(Base):
    """Full-text search index for efficient document and entity search"""
    __tablename__ = 'search_index'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    entity_id = Column(Integer, ForeignKey('legal_entities.id'))
    
    # Search content
    content = Column(Text, nullable=False)
    content_type = Column(String(50))  # 'document', 'entity', 'case'
    
    # Search metadata
    keywords = Column(JSON)  # Extracted keywords
    search_vector = Column(Text)  # For full-text search
    
    # Relevance scoring
    relevance_score = Column(Float, default=0.0)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Indexes for search performance
    __table_args__ = (
        Index('idx_content_type', 'content_type'),
        Index('idx_relevance_score', 'relevance_score'),
    )


class ProcessingLog(Base):
    """Table for logging document processing activities"""
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'))
    
    # Processing information
    processing_stage = Column(String(100))  # 'ocr', 'extraction', 'analysis'
    status = Column(String(50))  # 'started', 'completed', 'failed'
    message = Column(Text)
    
    # Performance metrics
    processing_time = Column(Float)
    memory_usage = Column(Float)
    
    # Error information
    error_type = Column(String(100))
    error_details = Column(JSON)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now(), index=True)
    
    # Relationship
    document = relationship("Document", backref="processing_logs")


class ChunkingStrategy(enum.Enum):
    """Strategies for text chunking"""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    LEGAL_SECTION = "legal_section"


class TextChunk(Base):
    """Model for storing text chunks from documents"""
    __tablename__ = 'text_chunks'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    document_id = Column(Integer, ForeignKey('documents.id'), nullable=False)
    
    # Chunk content and metadata
    content = Column(Text, nullable=False)
    chunk_index = Column(Integer, nullable=False)  # Order within document
    chunk_size = Column(Integer)  # Character count
    word_count = Column(Integer)
    sentence_count = Column(Integer)
    
    # Chunking strategy and parameters
    chunking_strategy = Column(Enum(ChunkingStrategy), nullable=False)
    chunk_overlap = Column(Integer, default=0)  # Overlap with adjacent chunks
    
    # Position information
    start_position = Column(Integer)  # Character position in original text
    end_position = Column(Integer)
    page_number = Column(Integer)  # If applicable
    
    # Content type and context
    content_type = Column(String(100))  # 'header', 'body', 'footer', 'table', etc.
    section_title = Column(String(500))  # Legal section title if applicable
    
    # Quality metrics
    coherence_score = Column(Float, default=0.0)  # Semantic coherence
    completeness_score = Column(Float, default=0.0)  # Sentence/paragraph completeness
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    document = relationship("Document", backref="text_chunks")
    embeddings = relationship("ChunkEmbedding", back_populates="chunk", cascade="all, delete-orphan")
    
    __table_args__ = (
        Index('idx_document_chunk_index', 'document_id', 'chunk_index'),
        Index('idx_chunking_strategy', 'chunking_strategy'),
        Index('idx_content_type', 'content_type'),
        Index('idx_coherence_score', 'coherence_score'),
    )


class EmbeddingModel(enum.Enum):
    """Available embedding models"""
    SENTENCE_TRANSFORMERS_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    SENTENCE_TRANSFORMERS_VIETNAMESE = "sentence-transformers/distiluse-base-multilingual-cased"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    CUSTOM_VIETNAMESE = "custom-vietnamese-legal"


class ChunkEmbedding(Base):
    """Model for storing embeddings of text chunks"""
    __tablename__ = 'chunk_embeddings'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    chunk_id = Column(Integer, ForeignKey('text_chunks.id'), nullable=False)
    
    # Embedding information
    embedding_model = Column(Enum(EmbeddingModel), nullable=False)
    embedding_vector = Column(JSON, nullable=False)  # Store as JSON array
    embedding_dimension = Column(Integer, nullable=False)
    
    # Model parameters
    model_version = Column(String(100))
    normalization_applied = Column(Boolean, default=True)
    
    # Quality metrics
    embedding_quality_score = Column(Float, default=0.0)
    
    # Processing information
    processing_time = Column(Float)  # Time to generate embedding
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    chunk = relationship("TextChunk", back_populates="embeddings")
    
    __table_args__ = (
        Index('idx_chunk_model', 'chunk_id', 'embedding_model'),
        Index('idx_embedding_model', 'embedding_model'),
        Index('idx_embedding_dimension', 'embedding_dimension'),
    )


class VectorSearchIndex(Base):
    """Model for vector similarity search optimization"""
    __tablename__ = 'vector_search_index'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    embedding_id = Column(Integer, ForeignKey('chunk_embeddings.id'), nullable=False)
    
    # Search optimization
    index_type = Column(String(50), default='flat')  # 'flat', 'ivf', 'hnsw'
    index_parameters = Column(JSON)  # Store index-specific parameters
    
    # Clustering information for faster search
    cluster_id = Column(Integer)
    cluster_centroid = Column(JSON)  # Cluster center vector
    
    # Search performance metrics
    average_search_time = Column(Float)
    search_accuracy = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    embedding = relationship("ChunkEmbedding", backref="search_indices")
    
    __table_args__ = (
        Index('idx_embedding_index_type', 'embedding_id', 'index_type'),
        Index('idx_cluster_id', 'cluster_id'),
    )