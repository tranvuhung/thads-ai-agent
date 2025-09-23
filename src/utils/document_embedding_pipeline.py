"""
Document Embedding Pipeline

This module integrates text chunking and embedding generation into the
document processing pipeline for legal documents.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_

# Import our modules
from ..database.models import (
    Document, TextChunk, ChunkEmbedding, ProcessingStatus,
    ChunkingStrategy, EmbeddingModel
)
from ..database.connection import get_database_connection
from .text_chunking import TextChunkingService, TextChunk as ChunkData
from .embedding import EmbeddingService, EmbeddingModel as EmbeddingModelEnum
from .vector_database import VectorDatabaseService

logger = logging.getLogger(__name__)


@dataclass
class ChunkingConfig:
    """Configuration for text chunking"""
    strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH
    chunk_size: int = 1000
    overlap: int = 100
    min_chunk_size: int = 50
    max_chunk_size: int = 2000
    sentences_per_chunk: int = 3  # For sentence chunking


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL
    normalize: bool = True
    batch_size: int = 32


@dataclass
class PipelineConfig:
    """Configuration for the complete pipeline"""
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    store_embeddings: bool = True
    update_search_index: bool = True
    overwrite_existing: bool = False


@dataclass
class ProcessingResult:
    """Result of document processing"""
    document_id: int
    chunks_created: int
    embeddings_created: int
    processing_time: float
    success: bool
    error_message: Optional[str] = None
    chunk_ids: List[int] = None
    embedding_ids: List[int] = None


class DocumentEmbeddingPipeline:
    """Pipeline for processing documents with chunking and embedding"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session or get_database_connection()
        self.chunking_service = TextChunkingService()
        self.embedding_service = EmbeddingService()
        self.vector_db_service = VectorDatabaseService(self.db_session)
    
    def process_document(
        self,
        document_id: int,
        config: Optional[PipelineConfig] = None
    ) -> ProcessingResult:
        """
        Process a single document through the complete pipeline
        
        Args:
            document_id: ID of the document to process
            config: Pipeline configuration
            
        Returns:
            ProcessingResult object
        """
        if config is None:
            config = PipelineConfig(
                chunking=ChunkingConfig(),
                embedding=EmbeddingConfig()
            )
        
        start_time = time.time()
        
        try:
            # Get document
            document = self.db_session.query(Document).filter(Document.id == document_id).first()
            if not document:
                raise ValueError(f"Document with ID {document_id} not found")
            
            if not document.raw_text:
                raise ValueError(f"Document {document_id} has no raw text to process")
            
            # Check if already processed and not overwriting
            if not config.overwrite_existing:
                existing_chunks = self.db_session.query(TextChunk).filter(
                    TextChunk.document_id == document_id
                ).count()
                
                if existing_chunks > 0:
                    logger.info(f"Document {document_id} already has chunks, skipping")
                    return ProcessingResult(
                        document_id=document_id,
                        chunks_created=0,
                        embeddings_created=0,
                        processing_time=time.time() - start_time,
                        success=True,
                        error_message="Already processed"
                    )
            
            # Step 1: Text Chunking
            logger.info(f"Starting chunking for document {document_id}")
            chunk_ids = self._chunk_document(document, config.chunking)
            
            # Step 2: Generate Embeddings
            logger.info(f"Starting embedding generation for {len(chunk_ids)} chunks")
            embedding_ids = []
            if config.store_embeddings:
                embedding_ids = self._generate_embeddings(chunk_ids, config.embedding)
            
            # Step 3: Update Search Index
            if config.update_search_index and embedding_ids:
                logger.info(f"Updating search index for {len(embedding_ids)} embeddings")
                self.vector_db_service.update_search_index(embedding_ids)
            
            processing_time = time.time() - start_time
            
            # Update document processing status
            document.processing_status = ProcessingStatus.COMPLETED
            document.processing_end = time.time()
            document.processing_time = processing_time
            self.db_session.commit()
            
            logger.info(f"Successfully processed document {document_id}: {len(chunk_ids)} chunks, {len(embedding_ids)} embeddings")
            
            return ProcessingResult(
                document_id=document_id,
                chunks_created=len(chunk_ids),
                embeddings_created=len(embedding_ids),
                processing_time=processing_time,
                success=True,
                chunk_ids=chunk_ids,
                embedding_ids=embedding_ids
            )
            
        except Exception as e:
            # Update document with error status
            try:
                document = self.db_session.query(Document).filter(Document.id == document_id).first()
                if document:
                    document.processing_status = ProcessingStatus.FAILED
                    document.error_message = str(e)
                    document.processing_end = time.time()
                self.db_session.commit()
            except:
                pass
            
            processing_time = time.time() - start_time
            logger.error(f"Error processing document {document_id}: {str(e)}")
            
            return ProcessingResult(
                document_id=document_id,
                chunks_created=0,
                embeddings_created=0,
                processing_time=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _chunk_document(self, document: Document, config: ChunkingConfig) -> List[int]:
        """
        Chunk a document and store chunks in database
        
        Args:
            document: Document object to chunk
            config: Chunking configuration
            
        Returns:
            List of created chunk IDs
        """
        try:
            # Delete existing chunks if overwriting
            existing_chunks = self.db_session.query(TextChunk).filter(
                TextChunk.document_id == document.id
            ).all()
            
            if existing_chunks:
                for chunk in existing_chunks:
                    self.db_session.delete(chunk)
                self.db_session.commit()
                logger.info(f"Deleted {len(existing_chunks)} existing chunks for document {document.id}")
            
            # Generate chunks
            chunker_kwargs = {
                'overlap': config.overlap,
                'min_chunk_size': config.min_chunk_size,
                'max_chunk_size': config.max_chunk_size
            }
            
            if config.strategy == ChunkingStrategy.SENTENCE:
                chunker_kwargs['sentences_per_chunk'] = config.sentences_per_chunk
            elif config.strategy == ChunkingStrategy.FIXED_SIZE:
                chunker_kwargs['chunk_size'] = config.chunk_size
            
            chunks = self.chunking_service.chunk_document(
                text=document.raw_text,
                strategy=config.strategy,
                **chunker_kwargs
            )
            
            # Store chunks in database
            chunk_records = []
            for chunk_data in chunks:
                chunk_record = TextChunk(
                    document_id=document.id,
                    content=chunk_data.content,
                    chunk_index=chunk_data.metadata.chunk_index,
                    chunk_size=chunk_data.metadata.chunk_size,
                    word_count=chunk_data.metadata.word_count,
                    sentence_count=chunk_data.metadata.sentence_count,
                    chunking_strategy=config.strategy,
                    chunk_overlap=config.overlap,
                    start_position=chunk_data.metadata.start_position,
                    end_position=chunk_data.metadata.end_position,
                    content_type=chunk_data.metadata.content_type,
                    section_title=chunk_data.metadata.section_title,
                    page_number=chunk_data.metadata.page_number,
                    coherence_score=chunk_data.metadata.coherence_score,
                    completeness_score=chunk_data.metadata.completeness_score
                )
                chunk_records.append(chunk_record)
            
            self.db_session.add_all(chunk_records)
            self.db_session.commit()
            
            chunk_ids = [chunk.id for chunk in chunk_records]
            logger.info(f"Created {len(chunk_ids)} chunks for document {document.id}")
            
            return chunk_ids
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error chunking document {document.id}: {str(e)}")
            raise
    
    def _generate_embeddings(self, chunk_ids: List[int], config: EmbeddingConfig) -> List[int]:
        """
        Generate embeddings for chunks
        
        Args:
            chunk_ids: List of chunk IDs to process
            config: Embedding configuration
            
        Returns:
            List of created embedding IDs
        """
        try:
            # Get chunks
            chunks = self.db_session.query(TextChunk).filter(
                TextChunk.id.in_(chunk_ids)
            ).all()
            
            if not chunks:
                logger.warning("No chunks found for embedding generation")
                return []
            
            # Process in batches
            embedding_ids = []
            batch_size = config.batch_size
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                batch_texts = [chunk.content for chunk in batch_chunks]
                
                # Generate embeddings for batch
                embedding_results = self.embedding_service.generate_embeddings_batch(
                    texts=batch_texts,
                    model=config.model
                )
                
                # Store embeddings
                batch_embeddings = []
                for j, embedding_result in enumerate(embedding_results):
                    chunk = batch_chunks[j]
                    batch_embeddings.append((chunk.id, embedding_result, config.model))
                
                batch_embedding_ids = self.vector_db_service.store_chunk_embeddings_batch(
                    batch_embeddings
                )
                embedding_ids.extend(batch_embedding_ids)
                
                logger.info(f"Processed embedding batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
            
            logger.info(f"Generated {len(embedding_ids)} embeddings for {len(chunk_ids)} chunks")
            return embedding_ids
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def process_documents_batch(
        self,
        document_ids: List[int],
        config: Optional[PipelineConfig] = None
    ) -> List[ProcessingResult]:
        """
        Process multiple documents in batch
        
        Args:
            document_ids: List of document IDs to process
            config: Pipeline configuration
            
        Returns:
            List of ProcessingResult objects
        """
        results = []
        
        for document_id in document_ids:
            try:
                result = self.process_document(document_id, config)
                results.append(result)
                
                # Log progress
                if len(results) % 10 == 0:
                    logger.info(f"Processed {len(results)}/{len(document_ids)} documents")
                    
            except Exception as e:
                logger.error(f"Error processing document {document_id} in batch: {str(e)}")
                results.append(ProcessingResult(
                    document_id=document_id,
                    chunks_created=0,
                    embeddings_created=0,
                    processing_time=0,
                    success=False,
                    error_message=str(e)
                ))
        
        # Summary statistics
        successful = sum(1 for r in results if r.success)
        total_chunks = sum(r.chunks_created for r in results)
        total_embeddings = sum(r.embeddings_created for r in results)
        total_time = sum(r.processing_time for r in results)
        
        logger.info(f"Batch processing complete: {successful}/{len(document_ids)} successful, "
                   f"{total_chunks} chunks, {total_embeddings} embeddings, "
                   f"{total_time:.2f}s total time")
        
        return results
    
    def reprocess_failed_documents(self, config: Optional[PipelineConfig] = None) -> List[ProcessingResult]:
        """
        Reprocess documents that failed processing
        
        Args:
            config: Pipeline configuration
            
        Returns:
            List of ProcessingResult objects
        """
        # Find failed documents
        failed_documents = self.db_session.query(Document.id).filter(
            Document.processing_status == ProcessingStatus.FAILED
        ).all()
        
        failed_ids = [doc.id for doc in failed_documents]
        
        if not failed_ids:
            logger.info("No failed documents found to reprocess")
            return []
        
        logger.info(f"Reprocessing {len(failed_ids)} failed documents")
        
        # Force overwrite for failed documents
        if config is None:
            config = PipelineConfig(
                chunking=ChunkingConfig(),
                embedding=EmbeddingConfig()
            )
        config.overwrite_existing = True
        
        return self.process_documents_batch(failed_ids, config)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about document processing
        
        Returns:
            Dictionary with processing statistics
        """
        try:
            # Document statistics
            total_documents = self.db_session.query(Document).count()
            processed_documents = self.db_session.query(Document).filter(
                Document.processing_status == ProcessingStatus.COMPLETED
            ).count()
            failed_documents = self.db_session.query(Document).filter(
                Document.processing_status == ProcessingStatus.FAILED
            ).count()
            pending_documents = self.db_session.query(Document).filter(
                Document.processing_status == ProcessingStatus.PENDING
            ).count()
            
            # Chunk statistics
            total_chunks = self.db_session.query(TextChunk).count()
            chunks_by_strategy = {}
            for strategy in ChunkingStrategy:
                count = self.db_session.query(TextChunk).filter(
                    TextChunk.chunking_strategy == strategy
                ).count()
                if count > 0:
                    chunks_by_strategy[strategy.value] = count
            
            # Get embedding statistics from vector database service
            embedding_stats = self.vector_db_service.get_embedding_statistics()
            
            return {
                'documents': {
                    'total': total_documents,
                    'processed': processed_documents,
                    'failed': failed_documents,
                    'pending': pending_documents,
                    'success_rate': processed_documents / total_documents if total_documents > 0 else 0
                },
                'chunks': {
                    'total': total_chunks,
                    'by_strategy': chunks_by_strategy
                },
                'embeddings': embedding_stats
            }
            
        except Exception as e:
            logger.error(f"Error getting processing statistics: {str(e)}")
            raise


# Utility functions for pipeline management
def create_default_config() -> PipelineConfig:
    """Create default pipeline configuration"""
    return PipelineConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.PARAGRAPH,
            chunk_size=1000,
            overlap=100,
            min_chunk_size=50,
            max_chunk_size=2000
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL,
            normalize=True,
            batch_size=32
        ),
        store_embeddings=True,
        update_search_index=True,
        overwrite_existing=False
    )


def create_legal_document_config() -> PipelineConfig:
    """Create configuration optimized for legal documents"""
    return PipelineConfig(
        chunking=ChunkingConfig(
            strategy=ChunkingStrategy.LEGAL_SECTION,
            overlap=50,
            min_chunk_size=100,
            max_chunk_size=1500
        ),
        embedding=EmbeddingConfig(
            model=EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL,
            normalize=True,
            batch_size=16  # Smaller batch for potentially longer legal sections
        ),
        store_embeddings=True,
        update_search_index=True,
        overwrite_existing=False
    )