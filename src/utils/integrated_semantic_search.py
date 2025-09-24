"""
Integrated Semantic Search System

This module provides a unified interface that integrates all semantic search components:
- Core semantic search engine
- Advanced similarity algorithms
- Ranking system
- Filtering system
- Existing embedding and vector database services
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

# Import existing services
from .embedding import EmbeddingService
from .vector_database import VectorDatabaseService

# Import new semantic search components
from .semantic_search import (
    SemanticSearchEngine, SearchQuery, SearchResult, SearchConfig
)
from .similarity_algorithms import (
    AdvancedSimilaritySearcher, HybridSimilaritySearcher, 
    SimilarityMetric, SearchAlgorithm, SearchConfig as SimilarityConfig
)
from .ranking_system import (
    AdvancedRankingSystem, RankingConfig, RankingAlgorithm, 
    RankedResult, RelevanceFeature
)
from .filtering_system import (
    AdvancedFilteringSystem, FilterConfig, FilterCriterion, 
    FilterGroup, FilterType, FilterOperator, LogicalOperator
)

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSearchConfig:
    """Configuration for integrated semantic search"""
    # Core search settings
    max_results: int = 100
    similarity_threshold: float = 0.3
    enable_reranking: bool = True
    enable_filtering: bool = True
    
    # Search algorithm settings
    similarity_metric: SimilarityMetric = SimilarityMetric.COSINE
    search_algorithm: SearchAlgorithm = SearchAlgorithm.FAISS_FLAT
    
    # Ranking settings
    ranking_algorithm: RankingAlgorithm = RankingAlgorithm.HYBRID_SCORE
    ranking_weights: Dict[RelevanceFeature, float] = field(default_factory=dict)
    
    # Filtering settings
    enable_quality_filter: bool = True
    min_quality_score: float = 0.3
    enable_temporal_filter: bool = False
    temporal_window_days: int = 365
    
    # Performance settings
    enable_caching: bool = True
    cache_ttl: int = 3600
    batch_size: int = 32
    enable_async: bool = True


@dataclass
class IntegratedSearchResult:
    """Comprehensive search result with all metadata"""
    # Original result data
    chunk_id: str
    content: str
    chunk_metadata: Dict[str, Any]
    
    # Similarity scores
    similarity_score: float
    similarity_details: Dict[str, float] = field(default_factory=dict)
    
    # Ranking information
    ranking_score: float
    ranking_features: Dict[str, float] = field(default_factory=dict)
    ranking_explanation: Dict[str, Any] = field(default_factory=dict)
    rank_position: int = 0
    
    # Filtering information
    filter_matches: List[str] = field(default_factory=list)
    quality_score: float = 0.0
    authority_score: float = 0.0
    
    # Additional metadata
    search_timestamp: datetime = field(default_factory=datetime.now)
    processing_time: float = 0.0


class IntegratedSemanticSearchSystem:
    """
    Unified semantic search system integrating all components
    """
    
    def __init__(self, 
                 embedding_service: EmbeddingService,
                 vector_db_service: VectorDatabaseService,
                 config: IntegratedSearchConfig = None):
        """
        Initialize the integrated semantic search system
        
        Args:
            embedding_service: Service for generating embeddings
            vector_db_service: Service for vector database operations
            config: Search configuration
        """
        self.embedding_service = embedding_service
        self.vector_db_service = vector_db_service
        self.config = config or IntegratedSearchConfig()
        
        # Initialize components
        self._initialize_components()
        
        # Performance tracking
        self.search_stats = {
            'total_searches': 0,
            'cache_hits': 0,
            'avg_response_time': 0.0,
            'last_search_time': None
        }
        
        logger.info("IntegratedSemanticSearchSystem initialized")
    
    def _initialize_components(self):
        """Initialize all search components"""
        # Core semantic search engine
        search_config = SearchConfig(
            max_results=self.config.max_results,
            similarity_threshold=self.config.similarity_threshold,
            enable_query_expansion=True,
            enable_preprocessing=True
        )
        self.search_engine = SemanticSearchEngine(
            embedding_service=self.embedding_service,
            vector_db_service=self.vector_db_service,
            config=search_config
        )
        
        # Advanced similarity searcher
        similarity_config = SimilarityConfig(
            metric=self.config.similarity_metric,
            algorithm=self.config.search_algorithm,
            enable_gpu=True,
            batch_size=self.config.batch_size
        )
        self.similarity_searcher = AdvancedSimilaritySearcher(similarity_config)
        
        # Ranking system
        ranking_config = RankingConfig(
            algorithm=self.config.ranking_algorithm,
            feature_weights=self.config.ranking_weights or self._get_default_ranking_weights(),
            normalize_scores=True,
            apply_diversity=True
        )
        self.ranking_system = AdvancedRankingSystem(ranking_config)
        
        # Filtering system
        filter_config = FilterConfig(
            enable_fuzzy_matching=True,
            enable_caching=self.config.enable_caching,
            max_results=self.config.max_results,
            min_score_threshold=self.config.similarity_threshold
        )
        self.filtering_system = AdvancedFilteringSystem(filter_config)
    
    def _get_default_ranking_weights(self) -> Dict[RelevanceFeature, float]:
        """Get default ranking feature weights"""
        return {
            RelevanceFeature.SEMANTIC_SIMILARITY: 0.4,
            RelevanceFeature.LEXICAL_OVERLAP: 0.2,
            RelevanceFeature.DOCUMENT_AUTHORITY: 0.15,
            RelevanceFeature.TEMPORAL_RELEVANCE: 0.1,
            RelevanceFeature.CONTENT_QUALITY: 0.1,
            RelevanceFeature.USER_ENGAGEMENT: 0.05
        }
    
    async def search(self, 
                    query: str,
                    filters: Optional[Union[FilterCriterion, FilterGroup, List[FilterCriterion]]] = None,
                    user_context: Optional[Dict[str, Any]] = None,
                    custom_config: Optional[Dict[str, Any]] = None) -> List[IntegratedSearchResult]:
        """
        Perform comprehensive semantic search
        
        Args:
            query: Search query
            filters: Optional filters to apply
            user_context: User context for personalization
            custom_config: Custom configuration overrides
            
        Returns:
            List of integrated search results
        """
        start_time = datetime.now()
        
        try:
            # Update search statistics
            self.search_stats['total_searches'] += 1
            self.search_stats['last_search_time'] = start_time
            
            # Step 1: Core semantic search
            logger.info(f"Starting semantic search for query: '{query}'")
            search_query = SearchQuery(
                text=query,
                max_results=self.config.max_results,
                similarity_threshold=self.config.similarity_threshold,
                metadata_filters=self._extract_metadata_filters(filters) if filters else None
            )
            
            core_results = await self._perform_core_search(search_query)
            
            if not core_results:
                logger.info("No results found in core search")
                return []
            
            # Step 2: Advanced similarity search (if enabled)
            if self.config.search_algorithm != SearchAlgorithm.BRUTE_FORCE:
                core_results = await self._enhance_similarity_scores(core_results, query)
            
            # Step 3: Apply filtering
            if self.config.enable_filtering and filters:
                core_results = await self._apply_filters(core_results, filters, user_context)
            
            # Step 4: Apply ranking
            if self.config.enable_reranking:
                core_results = await self._apply_ranking(core_results, query, user_context)
            
            # Step 5: Apply quality and temporal filters
            core_results = await self._apply_automatic_filters(core_results)
            
            # Step 6: Convert to integrated results
            integrated_results = await self._convert_to_integrated_results(
                core_results, query, start_time
            )
            
            # Update performance statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            self._update_performance_stats(processing_time)
            
            logger.info(f"Search completed: {len(integrated_results)} results in {processing_time:.3f}s")
            return integrated_results
            
        except Exception as e:
            logger.error(f"Error in integrated search: {e}")
            raise
    
    async def _perform_core_search(self, search_query: SearchQuery) -> List[SearchResult]:
        """Perform core semantic search"""
        if self.config.enable_async:
            return await asyncio.to_thread(self.search_engine.search, search_query)
        else:
            return self.search_engine.search(search_query)
    
    async def _enhance_similarity_scores(self, 
                                       results: List[SearchResult], 
                                       query: str) -> List[SearchResult]:
        """Enhance similarity scores using advanced algorithms"""
        try:
            # Generate query embedding
            query_embedding = await asyncio.to_thread(
                self.embedding_service.generate_embedding, query
            )
            
            # Extract result embeddings
            result_embeddings = []
            for result in results:
                if hasattr(result, 'embedding') and result.embedding is not None:
                    result_embeddings.append(result.embedding)
                else:
                    # Generate embedding for content if not available
                    embedding = await asyncio.to_thread(
                        self.embedding_service.generate_embedding, result.content
                    )
                    result_embeddings.append(embedding.embedding)
            
            if result_embeddings:
                # Calculate enhanced similarity scores
                enhanced_scores = self.similarity_searcher.calculate_similarities(
                    query_embedding.embedding,
                    np.array(result_embeddings)
                )
                
                # Update results with enhanced scores
                for i, result in enumerate(results):
                    if i < len(enhanced_scores):
                        result.similarity_score = float(enhanced_scores[i])
            
            return results
            
        except Exception as e:
            logger.warning(f"Error enhancing similarity scores: {e}")
            return results
    
    async def _apply_filters(self, 
                           results: List[SearchResult],
                           filters: Union[FilterCriterion, FilterGroup, List[FilterCriterion]],
                           user_context: Optional[Dict[str, Any]] = None) -> List[SearchResult]:
        """Apply filtering to search results"""
        try:
            filter_result = await asyncio.to_thread(
                self.filtering_system.apply_filters,
                results, filters, user_context
            )
            
            logger.info(f"Filtering: {len(results)} -> {len(filter_result.filtered_results)} results")
            return filter_result.filtered_results
            
        except Exception as e:
            logger.warning(f"Error applying filters: {e}")
            return results
    
    async def _apply_ranking(self, 
                           results: List[SearchResult],
                           query: str,
                           user_context: Optional[Dict[str, Any]] = None) -> List[RankedResult]:
        """Apply ranking to search results"""
        try:
            # Generate query embedding for ranking
            query_embedding = await asyncio.to_thread(
                self.embedding_service.generate_embedding, query
            )
            
            ranked_results = await asyncio.to_thread(
                self.ranking_system.rank_results,
                results, query, query_embedding.embedding, user_context
            )
            
            logger.info(f"Ranking applied to {len(results)} results")
            return ranked_results
            
        except Exception as e:
            logger.warning(f"Error applying ranking: {e}")
            # Convert to RankedResult format for consistency
            return [
                RankedResult(
                    original_result=result,
                    ranking_score=result.similarity_score,
                    ranking_features=None,
                    rank_position=i+1
                ) for i, result in enumerate(results)
            ]
    
    async def _apply_automatic_filters(self, results: List[Union[SearchResult, RankedResult]]) -> List[Union[SearchResult, RankedResult]]:
        """Apply automatic quality and temporal filters"""
        filtered_results = []
        
        for result in results:
            # Get the original result
            original_result = result.original_result if hasattr(result, 'original_result') else result
            
            # Apply quality filter
            if self.config.enable_quality_filter:
                quality_score = self._calculate_quality_score(original_result)
                if quality_score < self.config.min_quality_score:
                    continue
            
            # Apply temporal filter
            if self.config.enable_temporal_filter:
                if not self._passes_temporal_filter(original_result):
                    continue
            
            filtered_results.append(result)
        
        return filtered_results
    
    def _calculate_quality_score(self, result: Union[SearchResult, Any]) -> float:
        """Calculate quality score for a result"""
        content = getattr(result, 'content', '')
        metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
        
        quality_score = 0.5  # Base score
        
        # Content length quality
        length = len(content)
        if 50 <= length <= 2000:
            quality_score += 0.2
        elif length < 50:
            quality_score -= 0.2
        
        # Language quality
        sentences = content.split('.')
        if len(sentences) > 1:
            quality_score += 0.1
        
        # Metadata completeness
        if metadata.get('title'):
            quality_score += 0.1
        if metadata.get('author'):
            quality_score += 0.1
        
        return min(1.0, max(0.0, quality_score))
    
    def _passes_temporal_filter(self, result: Union[SearchResult, Any]) -> bool:
        """Check if result passes temporal filter"""
        metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
        created_date = metadata.get('created_date') or metadata.get('date')
        
        if not created_date:
            return True  # Pass if no date information
        
        try:
            if isinstance(created_date, str):
                date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            else:
                date_obj = created_date
            
            days_old = (datetime.now() - date_obj).days
            return days_old <= self.config.temporal_window_days
            
        except Exception:
            return True  # Pass if date parsing fails
    
    async def _convert_to_integrated_results(self, 
                                           results: List[Union[SearchResult, RankedResult]],
                                           query: str,
                                           start_time: datetime) -> List[IntegratedSearchResult]:
        """Convert results to integrated format"""
        integrated_results = []
        processing_time = (datetime.now() - start_time).total_seconds()
        
        for i, result in enumerate(results):
            # Extract data based on result type
            if hasattr(result, 'original_result'):
                # RankedResult
                original_result = result.original_result
                ranking_score = result.ranking_score
                ranking_features = result.ranking_features
                ranking_explanation = result.explanation
                rank_position = result.rank_position
            else:
                # SearchResult
                original_result = result
                ranking_score = getattr(result, 'similarity_score', 0.0)
                ranking_features = None
                ranking_explanation = {}
                rank_position = i + 1
            
            # Create integrated result
            integrated_result = IntegratedSearchResult(
                chunk_id=getattr(original_result, 'chunk_id', f'result_{i}'),
                content=getattr(original_result, 'content', ''),
                chunk_metadata=getattr(original_result, 'chunk_metadata', {}) or getattr(original_result, 'metadata', {}),
                similarity_score=getattr(original_result, 'similarity_score', 0.0),
                ranking_score=ranking_score,
                rank_position=rank_position,
                quality_score=self._calculate_quality_score(original_result),
                authority_score=self._calculate_authority_score(original_result),
                search_timestamp=start_time,
                processing_time=processing_time
            )
            
            # Add ranking features if available
            if ranking_features:
                integrated_result.ranking_features = {
                    attr: getattr(ranking_features, attr, 0.0)
                    for attr in dir(ranking_features)
                    if not attr.startswith('_') and not callable(getattr(ranking_features, attr))
                }
            
            # Add ranking explanation
            integrated_result.ranking_explanation = ranking_explanation
            
            integrated_results.append(integrated_result)
        
        return integrated_results
    
    def _calculate_authority_score(self, result: Union[SearchResult, Any]) -> float:
        """Calculate authority score for a result"""
        metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
        
        authority_score = 0.5  # Base score
        
        # Document type authority
        doc_type = metadata.get('document_type', '').lower()
        if 'law' in doc_type or 'regulation' in doc_type:
            authority_score += 0.3
        elif 'official' in doc_type:
            authority_score += 0.2
        
        # Source authority
        source = metadata.get('source', '').lower()
        if 'government' in source or 'official' in source:
            authority_score += 0.2
        
        return min(1.0, max(0.0, authority_score))
    
    def _extract_metadata_filters(self, filters: Union[FilterCriterion, FilterGroup, List[FilterCriterion]]) -> Dict[str, Any]:
        """Extract metadata filters for core search"""
        metadata_filters = {}
        
        # This is a simplified extraction - in practice, you'd want more sophisticated logic
        if isinstance(filters, FilterCriterion) and filters.filter_type == FilterType.METADATA_FILTER:
            metadata_filters[filters.field] = filters.value
        elif isinstance(filters, list):
            for f in filters:
                if isinstance(f, FilterCriterion) and f.filter_type == FilterType.METADATA_FILTER:
                    metadata_filters[f.field] = f.value
        
        return metadata_filters
    
    def _update_performance_stats(self, processing_time: float):
        """Update performance statistics"""
        total_searches = self.search_stats['total_searches']
        current_avg = self.search_stats['avg_response_time']
        
        # Calculate new average response time
        new_avg = ((current_avg * (total_searches - 1)) + processing_time) / total_searches
        self.search_stats['avg_response_time'] = new_avg
    
    # Convenience methods for common search patterns
    
    async def search_by_content(self, 
                              query: str,
                              content_filters: Optional[List[str]] = None,
                              max_results: int = None) -> List[IntegratedSearchResult]:
        """Search by content with optional content filters"""
        filters = []
        
        if content_filters:
            for content_filter in content_filters:
                filters.append(
                    self.filtering_system.create_content_filter(
                        content_filter, FilterOperator.CONTAINS
                    )
                )
        
        # Override max_results if specified
        original_max = self.config.max_results
        if max_results:
            self.config.max_results = max_results
        
        try:
            results = await self.search(query, filters if filters else None)
            return results
        finally:
            self.config.max_results = original_max
    
    async def search_by_metadata(self, 
                               query: str,
                               metadata_filters: Dict[str, Any],
                               max_results: int = None) -> List[IntegratedSearchResult]:
        """Search with metadata filters"""
        filters = []
        
        for field, value in metadata_filters.items():
            filters.append(
                self.filtering_system.create_metadata_filter(field, value)
            )
        
        # Override max_results if specified
        original_max = self.config.max_results
        if max_results:
            self.config.max_results = max_results
        
        try:
            results = await self.search(query, filters if filters else None)
            return results
        finally:
            self.config.max_results = original_max
    
    async def search_recent(self, 
                          query: str,
                          days: int = 30,
                          max_results: int = None) -> List[IntegratedSearchResult]:
        """Search for recent documents"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        temporal_filter = self.filtering_system.create_temporal_filter(
            start_date=start_date,
            end_date=end_date
        )
        
        # Override max_results if specified
        original_max = self.config.max_results
        if max_results:
            self.config.max_results = max_results
        
        try:
            results = await self.search(query, temporal_filter)
            return results
        finally:
            self.config.max_results = original_max
    
    async def search_high_quality(self, 
                                query: str,
                                min_quality: float = 0.7,
                                min_authority: float = 0.6,
                                max_results: int = None) -> List[IntegratedSearchResult]:
        """Search for high-quality, authoritative documents"""
        filters = [
            self.filtering_system.create_quality_filter(min_quality),
            self.filtering_system.create_authority_filter(min_authority)
        ]
        
        filter_group = FilterGroup(criteria=filters, logical_operator=LogicalOperator.AND)
        
        # Override max_results if specified
        original_max = self.config.max_results
        if max_results:
            self.config.max_results = max_results
        
        try:
            results = await self.search(query, filter_group)
            return results
        finally:
            self.config.max_results = original_max
    
    # System management methods
    
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        return {
            'search_stats': self.search_stats,
            'config': {
                'max_results': self.config.max_results,
                'similarity_threshold': self.config.similarity_threshold,
                'similarity_metric': self.config.similarity_metric.value,
                'search_algorithm': self.config.search_algorithm.value,
                'ranking_algorithm': self.config.ranking_algorithm.value
            },
            'component_stats': {
                'ranking_system': self.ranking_system.get_ranking_statistics(),
                'filtering_system': self.filtering_system.get_filter_statistics()
            }
        }
    
    def clear_caches(self):
        """Clear all system caches"""
        self.filtering_system.clear_cache()
        # Add other cache clearing as needed
        logger.info("All caches cleared")
    
    def update_config(self, new_config: IntegratedSearchConfig):
        """Update system configuration"""
        self.config = new_config
        self._initialize_components()
        logger.info("System configuration updated")


# Factory function for easy initialization
def create_integrated_search_system(embedding_service: EmbeddingService,
                                   vector_db_service: VectorDatabaseService,
                                   config: IntegratedSearchConfig = None) -> IntegratedSemanticSearchSystem:
    """
    Factory function to create an integrated semantic search system
    
    Args:
        embedding_service: Service for generating embeddings
        vector_db_service: Service for vector database operations
        config: Optional configuration
        
    Returns:
        Configured IntegratedSemanticSearchSystem
    """
    return IntegratedSemanticSearchSystem(
        embedding_service=embedding_service,
        vector_db_service=vector_db_service,
        config=config
    )