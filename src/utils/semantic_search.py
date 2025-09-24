"""
Semantic Search Engine for THADS AI Agent

This module provides comprehensive semantic search functionality including:
- Advanced query processing and expansion
- Multiple similarity search algorithms
- Intelligent ranking and scoring
- Flexible filtering and result refinement
- Performance optimization and caching
"""

import logging
import time
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import json
import hashlib
from datetime import datetime, timedelta

# Import our existing services
from .embedding import EmbeddingService, EmbeddingModel, EmbeddingResult
from .vector_database import VectorDatabaseService, SearchResult, VectorSearchQuery
from .text_chunking import TextChunkingService, ChunkingStrategy
from ..database.models import TextChunk, ChunkEmbedding, Document
from ..database.connection import get_database_connection

logger = logging.getLogger(__name__)


class SearchMode(Enum):
    """Search modes for different use cases"""
    EXACT = "exact"              # Exact semantic matching
    FUZZY = "fuzzy"              # Fuzzy semantic matching
    HYBRID = "hybrid"            # Combination of exact and fuzzy
    EXPLORATORY = "exploratory"  # Broad exploration with query expansion
    LEGAL = "legal"              # Legal document specific search


class RankingMethod(Enum):
    """Ranking methods for search results"""
    SIMILARITY_ONLY = "similarity_only"
    RELEVANCE_SCORE = "relevance_score"
    TEMPORAL_BOOST = "temporal_boost"
    DOCUMENT_AUTHORITY = "document_authority"
    HYBRID_RANKING = "hybrid_ranking"


class FilterType(Enum):
    """Types of filters for search results"""
    DOCUMENT_TYPE = "document_type"
    DATE_RANGE = "date_range"
    SIMILARITY_THRESHOLD = "similarity_threshold"
    CONTENT_LENGTH = "content_length"
    EMBEDDING_MODEL = "embedding_model"
    CHUNKING_STRATEGY = "chunking_strategy"


@dataclass
class QueryExpansion:
    """Query expansion configuration"""
    enabled: bool = True
    synonyms: bool = True
    related_terms: bool = True
    legal_terms: bool = True
    max_expansions: int = 5
    expansion_weight: float = 0.7


@dataclass
class SearchFilter:
    """Individual search filter"""
    filter_type: FilterType
    values: List[Any]
    operator: str = "in"  # in, not_in, range, equals, contains


@dataclass
class RankingConfig:
    """Configuration for result ranking"""
    method: RankingMethod = RankingMethod.HYBRID_RANKING
    similarity_weight: float = 0.6
    relevance_weight: float = 0.3
    temporal_weight: float = 0.1
    authority_weight: float = 0.0
    boost_recent: bool = True
    recent_days: int = 30


@dataclass
class SemanticSearchQuery:
    """Comprehensive semantic search query"""
    query_text: str
    search_mode: SearchMode = SearchMode.HYBRID
    embedding_models: List[EmbeddingModel] = field(default_factory=lambda: [EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL])
    top_k: int = 20
    similarity_threshold: float = 0.3
    filters: List[SearchFilter] = field(default_factory=list)
    ranking_config: RankingConfig = field(default_factory=RankingConfig)
    query_expansion: QueryExpansion = field(default_factory=QueryExpansion)
    include_metadata: bool = True
    cache_results: bool = True


@dataclass
class EnhancedSearchResult:
    """Enhanced search result with additional metadata"""
    chunk_id: int
    document_id: int
    content: str
    similarity_score: float
    relevance_score: float
    final_score: float
    rank: int
    chunk_metadata: Dict[str, Any]
    embedding_metadata: Dict[str, Any]
    document_metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchAnalytics:
    """Analytics for search performance"""
    query_text: str
    total_candidates: int
    filtered_candidates: int
    final_results: int
    processing_time: float
    embedding_time: float
    search_time: float
    ranking_time: float
    cache_hit: bool = False


class SemanticSearchEngine:
    """
    Advanced Semantic Search Engine
    
    Provides comprehensive semantic search functionality with:
    - Multi-model embedding support
    - Advanced query processing
    - Intelligent ranking and filtering
    - Performance optimization
    """
    
    def __init__(self, 
                 embedding_service: Optional[EmbeddingService] = None,
                 vector_db_service: Optional[VectorDatabaseService] = None,
                 enable_caching: bool = True,
                 cache_ttl: int = 3600):
        """
        Initialize the semantic search engine
        
        Args:
            embedding_service: Service for generating embeddings
            vector_db_service: Service for vector database operations
            enable_caching: Whether to enable result caching
            cache_ttl: Cache time-to-live in seconds
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.vector_db_service = vector_db_service or VectorDatabaseService()
        self.enable_caching = enable_caching
        self.cache_ttl = cache_ttl
        
        # Initialize caches
        self._result_cache: Dict[str, Tuple[List[EnhancedSearchResult], datetime]] = {}
        self._query_expansion_cache: Dict[str, List[str]] = {}
        
        # Legal domain specific terms for expansion
        self.legal_synonyms = {
            "luật": ["pháp luật", "quy định", "điều luật", "văn bản pháp luật"],
            "quyết định": ["nghị quyết", "chỉ thị", "thông tư"],
            "hợp đồng": ["thỏa thuận", "giao kèo", "cam kết"],
            "vi phạm": ["phạm pháp", "trái luật", "bất hợp pháp"],
            "trách nhiệm": ["nghĩa vụ", "bổn phận", "cam kết"],
        }
        
        logger.info("Semantic Search Engine initialized successfully")
    
    def search(self, query: SemanticSearchQuery) -> Tuple[List[EnhancedSearchResult], SearchAnalytics]:
        """
        Perform comprehensive semantic search
        
        Args:
            query: Semantic search query with all parameters
            
        Returns:
            Tuple of (search results, analytics)
        """
        start_time = time.time()
        
        # Check cache first
        if query.cache_results and self.enable_caching:
            cache_key = self._generate_cache_key(query)
            cached_results = self._get_cached_results(cache_key)
            if cached_results:
                analytics = SearchAnalytics(
                    query_text=query.query_text,
                    total_candidates=len(cached_results),
                    filtered_candidates=len(cached_results),
                    final_results=len(cached_results),
                    processing_time=time.time() - start_time,
                    embedding_time=0,
                    search_time=0,
                    ranking_time=0,
                    cache_hit=True
                )
                return cached_results, analytics
        
        # Process query
        processed_queries = self._process_query(query)
        embedding_start = time.time()
        
        # Generate embeddings for all query variants
        query_embeddings = []
        for processed_query in processed_queries:
            for model in query.embedding_models:
                try:
                    embedding_result = self.embedding_service.generate_embedding(
                        processed_query, model
                    )
                    query_embeddings.append((embedding_result.embedding_vector, model, processed_query))
                except Exception as e:
                    logger.warning(f"Failed to generate embedding with model {model}: {e}")
        
        embedding_time = time.time() - embedding_start
        search_start = time.time()
        
        # Perform vector searches
        all_candidates = []
        for embedding_vector, model, processed_query in query_embeddings:
            vector_query = VectorSearchQuery(
                query_embedding=embedding_vector,
                top_k=query.top_k * 2,  # Get more candidates for better ranking
                similarity_threshold=query.similarity_threshold * 0.8,  # Lower threshold for initial search
                embedding_models=[model] if model else None
            )
            
            # Apply filters to vector query
            self._apply_filters_to_vector_query(vector_query, query.filters)
            
            candidates = self.vector_db_service.similarity_search(vector_query)
            all_candidates.extend(candidates)
        
        search_time = time.time() - search_start
        ranking_start = time.time()
        
        # Remove duplicates and enhance results
        unique_candidates = self._deduplicate_results(all_candidates)
        
        # Apply additional filters
        filtered_candidates = self._apply_advanced_filters(unique_candidates, query.filters)
        
        # Rank and score results
        enhanced_results = self._rank_and_score_results(
            filtered_candidates, query, query_embeddings
        )
        
        # Limit to requested number of results
        final_results = enhanced_results[:query.top_k]
        
        ranking_time = time.time() - ranking_start
        
        # Cache results
        if query.cache_results and self.enable_caching:
            self._cache_results(cache_key, final_results)
        
        # Create analytics
        analytics = SearchAnalytics(
            query_text=query.query_text,
            total_candidates=len(all_candidates),
            filtered_candidates=len(filtered_candidates),
            final_results=len(final_results),
            processing_time=time.time() - start_time,
            embedding_time=embedding_time,
            search_time=search_time,
            ranking_time=ranking_time,
            cache_hit=False
        )
        
        logger.info(f"Search completed: {len(final_results)} results in {analytics.processing_time:.3f}s")
        
        return final_results, analytics
    
    def _process_query(self, query: SemanticSearchQuery) -> List[str]:
        """
        Process and expand the search query
        
        Args:
            query: Original search query
            
        Returns:
            List of processed query variants
        """
        queries = [query.query_text.strip()]
        
        if not query.query_expansion.enabled:
            return queries
        
        # Basic text preprocessing
        processed_text = self._preprocess_text(query.query_text)
        if processed_text != query.query_text:
            queries.append(processed_text)
        
        # Query expansion
        if query.query_expansion.synonyms:
            expanded_queries = self._expand_with_synonyms(query.query_text)
            queries.extend(expanded_queries[:query.query_expansion.max_expansions])
        
        if query.query_expansion.legal_terms and query.search_mode == SearchMode.LEGAL:
            legal_expanded = self._expand_with_legal_terms(query.query_text)
            queries.extend(legal_expanded[:query.query_expansion.max_expansions])
        
        return list(set(queries))  # Remove duplicates
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for better search"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize Vietnamese characters if needed
        # Add more preprocessing as needed
        
        return text
    
    def _expand_with_synonyms(self, query: str) -> List[str]:
        """Expand query with synonyms"""
        expanded = []
        words = query.lower().split()
        
        for word in words:
            if word in self.legal_synonyms:
                for synonym in self.legal_synonyms[word]:
                    expanded_query = query.replace(word, synonym)
                    expanded.append(expanded_query)
        
        return expanded
    
    def _expand_with_legal_terms(self, query: str) -> List[str]:
        """Expand query with legal-specific terms"""
        legal_expansions = []
        
        # Add legal context terms
        legal_contexts = [
            f"văn bản pháp luật {query}",
            f"quy định về {query}",
            f"điều khoản {query}",
            f"pháp lý {query}"
        ]
        
        legal_expansions.extend(legal_contexts)
        return legal_expansions
    
    def _apply_filters_to_vector_query(self, vector_query: VectorSearchQuery, filters: List[SearchFilter]):
        """Apply filters to vector database query"""
        for filter_item in filters:
            if filter_item.filter_type == FilterType.DOCUMENT_TYPE:
                # This would be handled in the vector database query
                pass
            elif filter_item.filter_type == FilterType.DATE_RANGE:
                if len(filter_item.values) == 2:
                    vector_query.date_range = (filter_item.values[0], filter_item.values[1])
            elif filter_item.filter_type == FilterType.EMBEDDING_MODEL:
                vector_query.embedding_models = filter_item.values
            elif filter_item.filter_type == FilterType.CHUNKING_STRATEGY:
                vector_query.chunking_strategies = filter_item.values
    
    def _apply_advanced_filters(self, candidates: List[SearchResult], filters: List[SearchFilter]) -> List[SearchResult]:
        """Apply advanced filters to search results"""
        filtered = candidates
        
        for filter_item in filters:
            if filter_item.filter_type == FilterType.SIMILARITY_THRESHOLD:
                threshold = filter_item.values[0] if filter_item.values else 0.5
                filtered = [r for r in filtered if r.similarity_score >= threshold]
            
            elif filter_item.filter_type == FilterType.CONTENT_LENGTH:
                if len(filter_item.values) == 2:
                    min_len, max_len = filter_item.values
                    filtered = [r for r in filtered if min_len <= len(r.content) <= max_len]
        
        return filtered
    
    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results based on chunk_id"""
        seen_chunks = set()
        unique_results = []
        
        for result in results:
            if result.chunk_id not in seen_chunks:
                seen_chunks.add(result.chunk_id)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_and_score_results(self, 
                               results: List[SearchResult], 
                               query: SemanticSearchQuery,
                               query_embeddings: List[Tuple[List[float], EmbeddingModel, str]]) -> List[EnhancedSearchResult]:
        """
        Rank and score search results using multiple factors
        """
        enhanced_results = []
        
        for result in results:
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(result, query)
            
            # Calculate final score based on ranking method
            final_score = self._calculate_final_score(
                result.similarity_score, 
                relevance_score, 
                result, 
                query.ranking_config
            )
            
            # Create enhanced result
            enhanced_result = EnhancedSearchResult(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                content=result.content,
                similarity_score=result.similarity_score,
                relevance_score=relevance_score,
                final_score=final_score,
                rank=0,  # Will be set after sorting
                chunk_metadata=result.chunk_metadata,
                embedding_metadata=result.embedding_metadata,
                explanation={
                    "similarity_score": result.similarity_score,
                    "relevance_score": relevance_score,
                    "final_score": final_score,
                    "ranking_method": query.ranking_config.method.value
                }
            )
            
            enhanced_results.append(enhanced_result)
        
        # Sort by final score
        enhanced_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Set ranks
        for i, result in enumerate(enhanced_results):
            result.rank = i + 1
        
        return enhanced_results
    
    def _calculate_relevance_score(self, result: SearchResult, query: SemanticSearchQuery) -> float:
        """Calculate relevance score based on content analysis"""
        relevance_score = 0.0
        
        # Text overlap score
        query_words = set(query.query_text.lower().split())
        content_words = set(result.content.lower().split())
        overlap = len(query_words.intersection(content_words))
        if query_words:
            text_overlap_score = overlap / len(query_words)
            relevance_score += text_overlap_score * 0.3
        
        # Content length normalization
        content_length = len(result.content)
        if 100 <= content_length <= 1000:  # Optimal length range
            length_score = 1.0
        elif content_length < 100:
            length_score = content_length / 100
        else:
            length_score = max(0.5, 1000 / content_length)
        
        relevance_score += length_score * 0.2
        
        # Metadata relevance
        metadata_score = self._calculate_metadata_relevance(result.chunk_metadata, query)
        relevance_score += metadata_score * 0.5
        
        return min(1.0, relevance_score)
    
    def _calculate_metadata_relevance(self, metadata: Dict[str, Any], query: SemanticSearchQuery) -> float:
        """Calculate relevance based on metadata"""
        score = 0.0
        
        # Document type relevance
        if query.search_mode == SearchMode.LEGAL:
            legal_types = ["law", "regulation", "decree", "decision"]
            doc_type = metadata.get("document_type", "").lower()
            if any(legal_type in doc_type for legal_type in legal_types):
                score += 0.5
        
        # Recency boost
        created_date = metadata.get("created_date")
        if created_date:
            try:
                date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                days_old = (datetime.now() - date_obj).days
                if days_old <= 30:
                    score += 0.3
                elif days_old <= 90:
                    score += 0.2
                elif days_old <= 365:
                    score += 0.1
            except:
                pass
        
        return min(1.0, score)
    
    def _calculate_final_score(self, 
                              similarity_score: float, 
                              relevance_score: float, 
                              result: SearchResult,
                              ranking_config: RankingConfig) -> float:
        """Calculate final ranking score"""
        
        if ranking_config.method == RankingMethod.SIMILARITY_ONLY:
            return similarity_score
        
        elif ranking_config.method == RankingMethod.RELEVANCE_SCORE:
            return relevance_score
        
        elif ranking_config.method == RankingMethod.HYBRID_RANKING:
            final_score = (
                similarity_score * ranking_config.similarity_weight +
                relevance_score * ranking_config.relevance_weight
            )
            
            # Temporal boost
            if ranking_config.boost_recent:
                temporal_boost = self._calculate_temporal_boost(result, ranking_config.recent_days)
                final_score += temporal_boost * ranking_config.temporal_weight
            
            return min(1.0, final_score)
        
        else:
            # Default to hybrid
            return (similarity_score + relevance_score) / 2
    
    def _calculate_temporal_boost(self, result: SearchResult, recent_days: int) -> float:
        """Calculate temporal boost for recent content"""
        created_date = result.chunk_metadata.get("created_date")
        if not created_date:
            return 0.0
        
        try:
            date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            days_old = (datetime.now() - date_obj).days
            
            if days_old <= recent_days:
                return 1.0 - (days_old / recent_days)
            else:
                return 0.0
        except:
            return 0.0
    
    def _generate_cache_key(self, query: SemanticSearchQuery) -> str:
        """Generate cache key for query"""
        query_dict = {
            "query_text": query.query_text,
            "search_mode": query.search_mode.value,
            "embedding_models": [m.value for m in query.embedding_models],
            "top_k": query.top_k,
            "similarity_threshold": query.similarity_threshold,
            "filters": [(f.filter_type.value, tuple(f.values), f.operator) for f in query.filters],
            "ranking_method": query.ranking_config.method.value
        }
        
        query_str = json.dumps(query_dict, sort_keys=True)
        return hashlib.md5(query_str.encode()).hexdigest()
    
    def _get_cached_results(self, cache_key: str) -> Optional[List[EnhancedSearchResult]]:
        """Get cached results if available and not expired"""
        if cache_key in self._result_cache:
            results, timestamp = self._result_cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return results
            else:
                del self._result_cache[cache_key]
        return None
    
    def _cache_results(self, cache_key: str, results: List[EnhancedSearchResult]):
        """Cache search results"""
        self._result_cache[cache_key] = (results, datetime.now())
        
        # Clean old cache entries
        if len(self._result_cache) > 1000:  # Limit cache size
            oldest_key = min(self._result_cache.keys(), 
                           key=lambda k: self._result_cache[k][1])
            del self._result_cache[oldest_key]
    
    def get_search_suggestions(self, partial_query: str, limit: int = 5) -> List[str]:
        """Get search suggestions based on partial query"""
        # This could be enhanced with a proper suggestion system
        suggestions = []
        
        # Basic suggestions based on legal terms
        legal_terms = [
            "luật doanh nghiệp", "luật lao động", "luật dân sự", 
            "luật hình sự", "luật thuế", "quyết định", "thông tư",
            "nghị định", "chỉ thị", "hợp đồng", "trách nhiệm"
        ]
        
        partial_lower = partial_query.lower()
        for term in legal_terms:
            if partial_lower in term or term.startswith(partial_lower):
                suggestions.append(term)
                if len(suggestions) >= limit:
                    break
        
        return suggestions
    
    def clear_cache(self):
        """Clear all cached results"""
        self._result_cache.clear()
        self._query_expansion_cache.clear()
        logger.info("Search cache cleared")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            "result_cache_size": len(self._result_cache),
            "query_expansion_cache_size": len(self._query_expansion_cache),
            "cache_enabled": self.enable_caching,
            "cache_ttl": self.cache_ttl
        }


# Convenience functions for common search patterns

def quick_search(query_text: str, 
                top_k: int = 10, 
                similarity_threshold: float = 0.5) -> List[EnhancedSearchResult]:
    """
    Quick semantic search with default settings
    
    Args:
        query_text: Search query
        top_k: Number of results to return
        similarity_threshold: Minimum similarity threshold
        
    Returns:
        List of search results
    """
    engine = SemanticSearchEngine()
    query = SemanticSearchQuery(
        query_text=query_text,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )
    
    results, _ = engine.search(query)
    return results


def legal_search(query_text: str, 
                top_k: int = 10,
                include_recent_only: bool = False) -> List[EnhancedSearchResult]:
    """
    Legal document specific search
    
    Args:
        query_text: Search query
        top_k: Number of results to return
        include_recent_only: Whether to boost recent documents
        
    Returns:
        List of search results
    """
    engine = SemanticSearchEngine()
    
    ranking_config = RankingConfig(
        method=RankingMethod.HYBRID_RANKING,
        boost_recent=include_recent_only,
        recent_days=90
    )
    
    query = SemanticSearchQuery(
        query_text=query_text,
        search_mode=SearchMode.LEGAL,
        top_k=top_k,
        ranking_config=ranking_config,
        query_expansion=QueryExpansion(legal_terms=True)
    )
    
    results, _ = engine.search(query)
    return results