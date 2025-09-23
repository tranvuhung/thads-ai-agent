"""
Vector Database Service for Embedding Storage and Similarity Search

This module provides functionality to store embeddings in a vector database
and perform efficient similarity searches.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc
import json

# Import our models
from ..database.models import (
    TextChunk, ChunkEmbedding, VectorSearchIndex,
    EmbeddingModel, ChunkingStrategy
)
from ..database.connection import get_database_connection
from .embedding import EmbeddingResult, EmbeddingService

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Result of vector similarity search"""
    chunk_id: int
    document_id: int
    content: str
    similarity_score: float
    chunk_metadata: Dict[str, Any]
    embedding_metadata: Dict[str, Any]


@dataclass
class VectorSearchQuery:
    """Query parameters for vector search"""
    query_embedding: List[float]
    top_k: int = 10
    similarity_threshold: float = 0.5
    document_ids: Optional[List[int]] = None
    content_types: Optional[List[str]] = None
    chunking_strategies: Optional[List[ChunkingStrategy]] = None
    embedding_models: Optional[List[EmbeddingModel]] = None
    date_range: Optional[Tuple[str, str]] = None


class VectorDatabaseService:
    """Service for vector database operations"""
    
    def __init__(self, db_session: Optional[Session] = None):
        self.db_session = db_session or get_database_connection()
        self.embedding_service = EmbeddingService()
    
    def store_chunk_embedding(
        self,
        chunk_id: int,
        embedding_result: EmbeddingResult,
        embedding_model: EmbeddingModel
    ) -> int:
        """
        Store embedding for a text chunk
        
        Args:
            chunk_id: ID of the text chunk
            embedding_result: Result from embedding generation
            embedding_model: Model used for embedding
            
        Returns:
            ID of the created embedding record
        """
        try:
            # Create embedding record
            chunk_embedding = ChunkEmbedding(
                chunk_id=chunk_id,
                embedding_model=embedding_model,
                embedding_vector=embedding_result.embedding_vector,
                embedding_dimension=embedding_result.embedding_dimension,
                model_version=embedding_result.model_version,
                normalization_applied=embedding_result.normalization_applied,
                embedding_quality_score=embedding_result.quality_score,
                processing_time=embedding_result.processing_time
            )
            
            self.db_session.add(chunk_embedding)
            self.db_session.commit()
            
            logger.info(f"Stored embedding for chunk {chunk_id} with model {embedding_model.value}")
            return chunk_embedding.id
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing embedding for chunk {chunk_id}: {str(e)}")
            raise
    
    def store_chunk_embeddings_batch(
        self,
        chunk_embeddings: List[Tuple[int, EmbeddingResult, EmbeddingModel]]
    ) -> List[int]:
        """
        Store embeddings for multiple chunks in batch
        
        Args:
            chunk_embeddings: List of (chunk_id, embedding_result, embedding_model) tuples
            
        Returns:
            List of created embedding record IDs
        """
        try:
            embedding_records = []
            
            for chunk_id, embedding_result, embedding_model in chunk_embeddings:
                chunk_embedding = ChunkEmbedding(
                    chunk_id=chunk_id,
                    embedding_model=embedding_model,
                    embedding_vector=embedding_result.embedding_vector,
                    embedding_dimension=embedding_result.embedding_dimension,
                    model_version=embedding_result.model_version,
                    normalization_applied=embedding_result.normalization_applied,
                    embedding_quality_score=embedding_result.quality_score,
                    processing_time=embedding_result.processing_time
                )
                embedding_records.append(chunk_embedding)
            
            self.db_session.add_all(embedding_records)
            self.db_session.commit()
            
            embedding_ids = [record.id for record in embedding_records]
            logger.info(f"Stored {len(embedding_ids)} embeddings in batch")
            return embedding_ids
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error storing batch embeddings: {str(e)}")
            raise
    
    def get_chunk_embeddings(
        self,
        chunk_ids: Optional[List[int]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        document_ids: Optional[List[int]] = None
    ) -> List[Tuple[int, List[float]]]:
        """
        Retrieve embeddings for chunks
        
        Args:
            chunk_ids: Specific chunk IDs to retrieve
            embedding_model: Filter by embedding model
            document_ids: Filter by document IDs
            
        Returns:
            List of (chunk_id, embedding_vector) tuples
        """
        try:
            query = self.db_session.query(ChunkEmbedding.chunk_id, ChunkEmbedding.embedding_vector)
            
            # Join with TextChunk if filtering by document_ids
            if document_ids:
                query = query.join(TextChunk).filter(TextChunk.document_id.in_(document_ids))
            
            # Apply filters
            if chunk_ids:
                query = query.filter(ChunkEmbedding.chunk_id.in_(chunk_ids))
            
            if embedding_model:
                query = query.filter(ChunkEmbedding.embedding_model == embedding_model)
            
            results = query.all()
            return [(chunk_id, embedding_vector) for chunk_id, embedding_vector in results]
            
        except Exception as e:
            logger.error(f"Error retrieving chunk embeddings: {str(e)}")
            raise
    
    def similarity_search(self, search_query: VectorSearchQuery) -> List[SearchResult]:
        """
        Perform similarity search using embeddings
        
        Args:
            search_query: Search query parameters
            
        Returns:
            List of SearchResult objects sorted by similarity
        """
        try:
            # Get candidate embeddings based on filters
            candidates = self._get_candidate_embeddings(search_query)
            
            if not candidates:
                logger.info("No candidate embeddings found for search")
                return []
            
            # Calculate similarities
            similarities = self.embedding_service.find_similar_embeddings(
                query_embedding=search_query.query_embedding,
                candidate_embeddings=[(chunk_id, embedding) for chunk_id, embedding, _ in candidates],
                top_k=search_query.top_k,
                similarity_threshold=search_query.similarity_threshold
            )
            
            # Build search results
            results = []
            candidate_map = {chunk_id: (embedding, metadata) for chunk_id, embedding, metadata in candidates}
            
            for chunk_id, similarity_score in similarities:
                if chunk_id in candidate_map:
                    _, metadata = candidate_map[chunk_id]
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=metadata['document_id'],
                        content=metadata['content'],
                        similarity_score=similarity_score,
                        chunk_metadata={
                            'chunk_index': metadata['chunk_index'],
                            'chunk_size': metadata['chunk_size'],
                            'word_count': metadata['word_count'],
                            'sentence_count': metadata['sentence_count'],
                            'chunking_strategy': metadata['chunking_strategy'],
                            'content_type': metadata['content_type'],
                            'section_title': metadata['section_title'],
                            'coherence_score': metadata['coherence_score'],
                            'completeness_score': metadata['completeness_score']
                        },
                        embedding_metadata={
                            'embedding_model': metadata['embedding_model'],
                            'embedding_dimension': metadata['embedding_dimension'],
                            'quality_score': metadata['embedding_quality_score']
                        }
                    )
                    results.append(result)
            
            logger.info(f"Found {len(results)} similar chunks")
            return results
            
        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise
    
    def _get_candidate_embeddings(self, search_query: VectorSearchQuery) -> List[Tuple[int, List[float], Dict[str, Any]]]:
        """
        Get candidate embeddings based on search filters
        
        Args:
            search_query: Search query parameters
            
        Returns:
            List of (chunk_id, embedding_vector, metadata) tuples
        """
        try:
            # Build query with joins
            query = self.db_session.query(
                ChunkEmbedding.chunk_id,
                ChunkEmbedding.embedding_vector,
                ChunkEmbedding.embedding_model,
                ChunkEmbedding.embedding_dimension,
                ChunkEmbedding.embedding_quality_score,
                TextChunk.document_id,
                TextChunk.content,
                TextChunk.chunk_index,
                TextChunk.chunk_size,
                TextChunk.word_count,
                TextChunk.sentence_count,
                TextChunk.chunking_strategy,
                TextChunk.content_type,
                TextChunk.section_title,
                TextChunk.coherence_score,
                TextChunk.completeness_score
            ).join(TextChunk)
            
            # Apply filters
            if search_query.document_ids:
                query = query.filter(TextChunk.document_id.in_(search_query.document_ids))
            
            if search_query.content_types:
                query = query.filter(TextChunk.content_type.in_(search_query.content_types))
            
            if search_query.chunking_strategies:
                query = query.filter(TextChunk.chunking_strategy.in_(search_query.chunking_strategies))
            
            if search_query.embedding_models:
                query = query.filter(ChunkEmbedding.embedding_model.in_(search_query.embedding_models))
            
            # Execute query
            results = query.all()
            
            # Format results
            candidates = []
            for result in results:
                metadata = {
                    'document_id': result.document_id,
                    'content': result.content,
                    'chunk_index': result.chunk_index,
                    'chunk_size': result.chunk_size,
                    'word_count': result.word_count,
                    'sentence_count': result.sentence_count,
                    'chunking_strategy': result.chunking_strategy.value if result.chunking_strategy else None,
                    'content_type': result.content_type,
                    'section_title': result.section_title,
                    'coherence_score': result.coherence_score,
                    'completeness_score': result.completeness_score,
                    'embedding_model': result.embedding_model.value if result.embedding_model else None,
                    'embedding_dimension': result.embedding_dimension,
                    'embedding_quality_score': result.embedding_quality_score
                }
                
                candidates.append((result.chunk_id, result.embedding_vector, metadata))
            
            return candidates
            
        except Exception as e:
            logger.error(f"Error getting candidate embeddings: {str(e)}")
            raise
    
    def search_by_text(
        self,
        query_text: str,
        embedding_model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL,
        top_k: int = 10,
        similarity_threshold: float = 0.5,
        **search_filters
    ) -> List[SearchResult]:
        """
        Search by text query (generates embedding automatically)
        
        Args:
            query_text: Text query to search for
            embedding_model: Model to use for query embedding
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            **search_filters: Additional search filters
            
        Returns:
            List of SearchResult objects
        """
        try:
            # Generate embedding for query text
            embedding_result = self.embedding_service.generate_embedding(query_text, embedding_model)
            
            # Create search query
            search_query = VectorSearchQuery(
                query_embedding=embedding_result.embedding_vector,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                embedding_models=[embedding_model],
                **search_filters
            )
            
            return self.similarity_search(search_query)
            
        except Exception as e:
            logger.error(f"Error searching by text: {str(e)}")
            raise
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about stored embeddings
        
        Returns:
            Dictionary with embedding statistics
        """
        try:
            # Count embeddings by model
            model_counts = {}
            for model in EmbeddingModel:
                count = self.db_session.query(ChunkEmbedding).filter(
                    ChunkEmbedding.embedding_model == model
                ).count()
                if count > 0:
                    model_counts[model.value] = count
            
            # Total embeddings
            total_embeddings = self.db_session.query(ChunkEmbedding).count()
            
            # Average quality scores by model
            quality_scores = {}
            for model in EmbeddingModel:
                avg_quality = self.db_session.query(
                    ChunkEmbedding.embedding_quality_score
                ).filter(
                    ChunkEmbedding.embedding_model == model
                ).filter(
                    ChunkEmbedding.embedding_quality_score.isnot(None)
                ).all()
                
                if avg_quality:
                    scores = [score[0] for score in avg_quality if score[0] is not None]
                    if scores:
                        quality_scores[model.value] = {
                            'average': sum(scores) / len(scores),
                            'min': min(scores),
                            'max': max(scores)
                        }
            
            # Embedding dimensions
            dimensions = {}
            for model in EmbeddingModel:
                dim_result = self.db_session.query(ChunkEmbedding.embedding_dimension).filter(
                    ChunkEmbedding.embedding_model == model
                ).first()
                if dim_result:
                    dimensions[model.value] = dim_result[0]
            
            return {
                'total_embeddings': total_embeddings,
                'embeddings_by_model': model_counts,
                'quality_scores': quality_scores,
                'embedding_dimensions': dimensions
            }
            
        except Exception as e:
            logger.error(f"Error getting embedding statistics: {str(e)}")
            raise
    
    def delete_embeddings(
        self,
        chunk_ids: Optional[List[int]] = None,
        document_ids: Optional[List[int]] = None,
        embedding_model: Optional[EmbeddingModel] = None
    ) -> int:
        """
        Delete embeddings based on filters
        
        Args:
            chunk_ids: Specific chunk IDs to delete
            document_ids: Delete embeddings for chunks from these documents
            embedding_model: Delete embeddings from specific model
            
        Returns:
            Number of deleted embeddings
        """
        try:
            query = self.db_session.query(ChunkEmbedding)
            
            # Join with TextChunk if filtering by document_ids
            if document_ids:
                query = query.join(TextChunk).filter(TextChunk.document_id.in_(document_ids))
            
            # Apply filters
            if chunk_ids:
                query = query.filter(ChunkEmbedding.chunk_id.in_(chunk_ids))
            
            if embedding_model:
                query = query.filter(ChunkEmbedding.embedding_model == embedding_model)
            
            # Count before deletion
            count = query.count()
            
            # Delete
            query.delete(synchronize_session=False)
            self.db_session.commit()
            
            logger.info(f"Deleted {count} embeddings")
            return count
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error deleting embeddings: {str(e)}")
            raise
    
    def update_search_index(self, embedding_ids: List[int], index_type: str = "flat"):
        """
        Update vector search index for better performance
        
        Args:
            embedding_ids: List of embedding IDs to index
            index_type: Type of index to create
        """
        try:
            # Simple implementation - in production, you might use specialized vector databases
            # like Faiss, Pinecone, or Weaviate for better performance
            
            for embedding_id in embedding_ids:
                # Check if index already exists
                existing_index = self.db_session.query(VectorSearchIndex).filter(
                    VectorSearchIndex.embedding_id == embedding_id,
                    VectorSearchIndex.index_type == index_type
                ).first()
                
                if not existing_index:
                    search_index = VectorSearchIndex(
                        embedding_id=embedding_id,
                        index_type=index_type,
                        index_parameters={'created_at': time.time()}
                    )
                    self.db_session.add(search_index)
            
            self.db_session.commit()
            logger.info(f"Updated search index for {len(embedding_ids)} embeddings")
            
        except Exception as e:
            self.db_session.rollback()
            logger.error(f"Error updating search index: {str(e)}")
            raise


# Utility functions for vector operations
def cosine_similarity_batch(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between query vector and batch of vectors
    
    Args:
        query_vector: Query vector (1D array)
        vectors: Batch of vectors (2D array, each row is a vector)
        
    Returns:
        Array of similarity scores
    """
    # Normalize vectors
    query_norm = query_vector / np.linalg.norm(query_vector)
    vectors_norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Calculate cosine similarity
    similarities = np.dot(vectors_norm, query_norm)
    return similarities


def euclidean_distance_batch(query_vector: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    """
    Calculate Euclidean distance between query vector and batch of vectors
    
    Args:
        query_vector: Query vector (1D array)
        vectors: Batch of vectors (2D array, each row is a vector)
        
    Returns:
        Array of distances
    """
    distances = np.linalg.norm(vectors - query_vector, axis=1)
    return distances