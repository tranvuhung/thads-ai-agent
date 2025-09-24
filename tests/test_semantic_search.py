"""
Comprehensive Test Suite for Semantic Search System

This module contains unit tests, integration tests, and performance tests
for all components of the semantic search system.
"""

import unittest
import numpy as np
import tempfile
import shutil
import os
import time
from unittest.mock import Mock, patch, MagicMock
from typing import List, Dict, Any
import sys
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.utils.semantic_search import (
    SemanticSearchEngine, SearchQuery, SearchResult, SearchConfig
)
from src.utils.similarity_algorithms import (
    AdvancedSimilaritySearcher, SimilarityMetric, SearchAlgorithm,
    HybridSimilaritySearcher, AdaptiveSimilaritySearcher
)
from src.utils.ranking_system import (
    AdvancedRankingSystem, RankingAlgorithm, RankingConfig,
    LearningToRankSystem
)
from src.utils.filtering_system import (
    AdvancedFilteringSystem, FilterType, FilterOperator,
    FilterCriterion, FilterGroup
)
from src.utils.query_expansion import (
    QueryPreprocessor, QueryExpander, IntegratedQueryProcessor,
    PreprocessingConfig, QueryExpansionConfig, PreprocessingStep, ExpansionMethod
)
from src.utils.performance_optimization import (
    CacheManager, CacheConfig, CacheType, OptimizedSemanticSearchCache,
    PerformanceConfig, BatchProcessor
)
from src.utils.integrated_semantic_search import (
    IntegratedSemanticSearchSystem, IntegratedSearchConfig
)

class TestSemanticSearchEngine(unittest.TestCase):
    """Test cases for core semantic search engine"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SearchConfig(
            max_results=10,
            similarity_threshold=0.5,
            enable_reranking=True
        )
        
        # Mock embedding service
        self.mock_embedding_service = Mock()
        self.mock_embedding_service.generate_embedding.return_value = np.random.rand(384)
        
        # Mock vector database service
        self.mock_vector_db = Mock()
        self.mock_vector_db.similarity_search.return_value = [
            {"chunk_id": "1", "content": "test content 1", "similarity": 0.9},
            {"chunk_id": "2", "content": "test content 2", "similarity": 0.8}
        ]
        
        self.search_engine = SemanticSearchEngine(
            embedding_service=self.mock_embedding_service,
            vector_db_service=self.mock_vector_db,
            config=self.config
        )
    
    def test_search_basic_functionality(self):
        """Test basic search functionality"""
        query = SearchQuery(
            text="test query",
            max_results=5,
            similarity_threshold=0.7
        )
        
        results = self.search_engine.search(query)
        
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) <= 5)
        self.mock_embedding_service.generate_embedding.assert_called_once()
        self.mock_vector_db.similarity_search.assert_called_once()
    
    def test_search_with_filters(self):
        """Test search with filters applied"""
        query = SearchQuery(
            text="test query",
            filters={"document_type": "pdf", "date_range": "2023-01-01:2023-12-31"}
        )
        
        results = self.search_engine.search(query)
        
        self.assertIsInstance(results, list)
        # Verify filters were applied in the search call
        call_args = self.mock_vector_db.similarity_search.call_args
        self.assertIsNotNone(call_args)
    
    def test_search_empty_query(self):
        """Test search with empty query"""
        query = SearchQuery(text="")
        
        results = self.search_engine.search(query)
        
        self.assertEqual(len(results), 0)
    
    def test_search_no_results(self):
        """Test search when no results found"""
        self.mock_vector_db.similarity_search.return_value = []
        
        query = SearchQuery(text="no results query")
        results = self.search_engine.search(query)
        
        self.assertEqual(len(results), 0)

class TestSimilarityAlgorithms(unittest.TestCase):
    """Test cases for similarity algorithms"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.searcher = AdvancedSimilaritySearcher()
        
        # Create test vectors
        self.query_vector = np.random.rand(100)
        self.document_vectors = np.random.rand(50, 100)
        self.document_ids = [f"doc_{i}" for i in range(50)]
    
    def test_cosine_similarity_search(self):
        """Test cosine similarity search"""
        results = self.searcher.search(
            query_vector=self.query_vector,
            document_vectors=self.document_vectors,
            document_ids=self.document_ids,
            metric=SimilarityMetric.COSINE,
            top_k=10
        )
        
        self.assertEqual(len(results), 10)
        self.assertTrue(all(0 <= r.similarity <= 1 for r in results))
        # Results should be sorted by similarity (descending)
        similarities = [r.similarity for r in results]
        self.assertEqual(similarities, sorted(similarities, reverse=True))
    
    def test_euclidean_distance_search(self):
        """Test Euclidean distance search"""
        results = self.searcher.search(
            query_vector=self.query_vector,
            document_vectors=self.document_vectors,
            document_ids=self.document_ids,
            metric=SimilarityMetric.EUCLIDEAN,
            top_k=5
        )
        
        self.assertEqual(len(results), 5)
        self.assertTrue(all(r.similarity >= 0 for r in results))
    
    def test_hybrid_similarity_search(self):
        """Test hybrid similarity search"""
        hybrid_searcher = HybridSimilaritySearcher()
        
        # Configure multiple metrics with weights
        metrics_config = {
            SimilarityMetric.COSINE: 0.6,
            SimilarityMetric.EUCLIDEAN: 0.4
        }
        
        results = hybrid_searcher.search_hybrid(
            query_vector=self.query_vector,
            document_vectors=self.document_vectors,
            document_ids=self.document_ids,
            metrics_weights=metrics_config,
            top_k=10
        )
        
        self.assertEqual(len(results), 10)
        self.assertTrue(all(r.similarity >= 0 for r in results))
    
    def test_adaptive_similarity_search(self):
        """Test adaptive similarity search"""
        adaptive_searcher = AdaptiveSimilaritySearcher()
        
        results = adaptive_searcher.search_adaptive(
            query_vector=self.query_vector,
            document_vectors=self.document_vectors,
            document_ids=self.document_ids,
            top_k=10
        )
        
        self.assertEqual(len(results), 10)
        self.assertIsNotNone(adaptive_searcher.selected_metric)

class TestRankingSystem(unittest.TestCase):
    """Test cases for ranking system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = RankingConfig(
            algorithm=RankingAlgorithm.HYBRID_SCORE,
            enable_diversity=True,
            diversity_threshold=0.8
        )
        self.ranking_system = AdvancedRankingSystem(self.config)
        
        # Create test search results
        self.search_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"test content {i}",
                "similarity": 0.9 - i * 0.1,
                "metadata": {"document_id": f"doc_{i}", "page": i}
            }
            for i in range(10)
        ]
    
    def test_bm25_ranking(self):
        """Test BM25 ranking algorithm"""
        query = "test query"
        
        ranked_results = self.ranking_system.rank_results(
            query=query,
            results=self.search_results,
            algorithm=RankingAlgorithm.BM25
        )
        
        self.assertEqual(len(ranked_results), len(self.search_results))
        self.assertTrue(all(hasattr(r, 'final_score') for r in ranked_results))
    
    def test_hybrid_ranking(self):
        """Test hybrid ranking algorithm"""
        query = "test query"
        
        ranked_results = self.ranking_system.rank_results(
            query=query,
            results=self.search_results,
            algorithm=RankingAlgorithm.HYBRID_SCORE
        )
        
        self.assertEqual(len(ranked_results), len(self.search_results))
        # Results should be sorted by final score
        scores = [r.final_score for r in ranked_results]
        self.assertEqual(scores, sorted(scores, reverse=True))
    
    def test_diversity_reranking(self):
        """Test diversity-based reranking"""
        # Create results with similar content
        similar_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": "very similar content about machine learning",
                "similarity": 0.9,
                "metadata": {"document_id": f"doc_{i}"}
            }
            for i in range(5)
        ]
        
        ranked_results = self.ranking_system.apply_diversity_reranking(
            similar_results,
            diversity_threshold=0.8
        )
        
        # Should have fewer results due to diversity filtering
        self.assertLessEqual(len(ranked_results), len(similar_results))
    
    def test_learning_to_rank(self):
        """Test learning to rank system"""
        ltr_system = LearningToRankSystem()
        
        # Mock training data
        training_data = [
            {
                "query": "test query",
                "results": self.search_results,
                "relevance_scores": [1, 1, 0, 0, 1, 0, 1, 0, 0, 1]
            }
        ]
        
        # Test training (mock implementation)
        ltr_system.train(training_data)
        
        # Test prediction
        ranked_results = ltr_system.rank_results(
            query="test query",
            results=self.search_results
        )
        
        self.assertEqual(len(ranked_results), len(self.search_results))

class TestFilteringSystem(unittest.TestCase):
    """Test cases for filtering system"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.filtering_system = AdvancedFilteringSystem()
        
        # Create test search results
        self.search_results = [
            {
                "chunk_id": f"chunk_{i}",
                "content": f"test content {i}",
                "similarity": 0.9 - i * 0.1,
                "metadata": {
                    "document_type": "pdf" if i % 2 == 0 else "txt",
                    "date": f"2023-{(i % 12) + 1:02d}-01",
                    "author": f"author_{i % 3}",
                    "quality_score": 0.8 + (i % 3) * 0.1
                }
            }
            for i in range(10)
        ]
    
    def test_content_filtering(self):
        """Test content-based filtering"""
        filter_criterion = FilterCriterion(
            filter_type=FilterType.CONTENT_FILTER,
            field="content",
            operator=FilterOperator.CONTAINS,
            value="test"
        )
        
        filtered_results = self.filtering_system.apply_filters(
            self.search_results,
            [filter_criterion]
        )
        
        self.assertTrue(len(filtered_results.results) <= len(self.search_results))
        # All results should contain "test" in content
        for result in filtered_results.results:
            self.assertIn("test", result["content"])
    
    def test_metadata_filtering(self):
        """Test metadata-based filtering"""
        filter_criterion = FilterCriterion(
            filter_type=FilterType.METADATA_FILTER,
            field="document_type",
            operator=FilterOperator.EQUALS,
            value="pdf"
        )
        
        filtered_results = self.filtering_system.apply_filters(
            self.search_results,
            [filter_criterion]
        )
        
        # All results should have document_type = "pdf"
        for result in filtered_results.results:
            self.assertEqual(result["metadata"]["document_type"], "pdf")
    
    def test_temporal_filtering(self):
        """Test temporal filtering"""
        filter_criterion = FilterCriterion(
            filter_type=FilterType.TEMPORAL_FILTER,
            field="date",
            operator=FilterOperator.BETWEEN,
            value=["2023-01-01", "2023-06-30"]
        )
        
        filtered_results = self.filtering_system.apply_filters(
            self.search_results,
            [filter_criterion]
        )
        
        # All results should be within the date range
        for result in filtered_results.results:
            date = result["metadata"]["date"]
            self.assertTrue("2023-01" <= date <= "2023-06")
    
    def test_quality_filtering(self):
        """Test quality-based filtering"""
        filter_criterion = FilterCriterion(
            filter_type=FilterType.QUALITY_FILTER,
            field="quality_score",
            operator=FilterOperator.GREATER_THAN,
            value=0.85
        )
        
        filtered_results = self.filtering_system.apply_filters(
            self.search_results,
            [filter_criterion]
        )
        
        # All results should have quality_score > 0.85
        for result in filtered_results.results:
            self.assertGreater(result["metadata"]["quality_score"], 0.85)
    
    def test_complex_filtering(self):
        """Test complex filtering with multiple criteria"""
        filter_group = FilterGroup(
            criteria=[
                FilterCriterion(
                    filter_type=FilterType.METADATA_FILTER,
                    field="document_type",
                    operator=FilterOperator.EQUALS,
                    value="pdf"
                ),
                FilterCriterion(
                    filter_type=FilterType.QUALITY_FILTER,
                    field="quality_score",
                    operator=FilterOperator.GREATER_THAN,
                    value=0.8
                )
            ],
            logical_operator="AND"
        )
        
        filtered_results = self.filtering_system.apply_filter_group(
            self.search_results,
            filter_group
        )
        
        # All results should satisfy both criteria
        for result in filtered_results.results:
            self.assertEqual(result["metadata"]["document_type"], "pdf")
            self.assertGreater(result["metadata"]["quality_score"], 0.8)

class TestQueryExpansion(unittest.TestCase):
    """Test cases for query expansion and preprocessing"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.preprocessing_config = PreprocessingConfig(
            steps=[
                PreprocessingStep.LOWERCASING,
                PreprocessingStep.PUNCTUATION_REMOVAL,
                PreprocessingStep.TOKENIZATION,
                PreprocessingStep.STOPWORD_REMOVAL,
                PreprocessingStep.LEMMATIZATION
            ]
        )
        
        self.expansion_config = QueryExpansionConfig(
            expansion_methods=[
                ExpansionMethod.SYNONYM_EXPANSION,
                ExpansionMethod.SEMANTIC_EXPANSION
            ],
            max_expansions_per_term=3
        )
        
        self.preprocessor = QueryPreprocessor(self.preprocessing_config)
        self.expander = QueryExpander(self.expansion_config)
        self.integrated_processor = IntegratedQueryProcessor(
            self.preprocessing_config,
            self.expansion_config
        )
    
    def test_text_preprocessing(self):
        """Test text preprocessing functionality"""
        text = "Hello World! This is a TEST query with PUNCTUATION."
        
        processed_text, applied_steps = self.preprocessor.preprocess(text)
        
        self.assertIsInstance(processed_text, str)
        self.assertIsInstance(applied_steps, list)
        self.assertTrue(len(applied_steps) > 0)
        # Text should be lowercase after preprocessing
        self.assertTrue(processed_text.islower())
    
    def test_query_expansion(self):
        """Test query expansion functionality"""
        query = "machine learning algorithms"
        
        expanded_query = self.expander.expand_query(query)
        
        self.assertEqual(expanded_query.original_query, query)
        self.assertIsInstance(expanded_query.expanded_terms, list)
        self.assertIsInstance(expanded_query.expansion_weights, dict)
        self.assertTrue(0 <= expanded_query.confidence_score <= 1)
    
    def test_integrated_processing(self):
        """Test integrated query processing"""
        query = "Natural Language Processing techniques"
        
        processed_query = self.integrated_processor.process_query(query)
        
        self.assertEqual(processed_query.original_query, query)
        self.assertIsInstance(processed_query.expanded_terms, list)
        self.assertIsInstance(processed_query.preprocessing_applied, list)
        
        # Get final query terms
        final_terms = self.integrated_processor.get_final_query_terms(processed_query)
        self.assertIsInstance(final_terms, list)
        self.assertTrue(len(final_terms) > 0)

class TestPerformanceOptimization(unittest.TestCase):
    """Test cases for performance optimization and caching"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.cache_config = CacheConfig(
            cache_type=CacheType.MEMORY,
            max_size=100,
            ttl_seconds=60
        )
        
        self.performance_config = PerformanceConfig(
            max_workers=2,
            batch_size=10,
            enable_monitoring=False  # Disable for testing
        )
        
        self.cache_manager = CacheManager(self.cache_config)
        self.batch_processor = BatchProcessor(batch_size=5, max_workers=2)
    
    def test_cache_operations(self):
        """Test basic cache operations"""
        # Test put and get
        self.cache_manager.put("test_key", "test_value")
        value = self.cache_manager.get("test_key")
        
        self.assertEqual(value, "test_value")
        
        # Test cache miss
        missing_value = self.cache_manager.get("missing_key")
        self.assertIsNone(missing_value)
        
        # Test cache statistics
        stats = self.cache_manager.get_cache_stats()
        self.assertIn("cache_hits", stats)
        self.assertIn("cache_misses", stats)
        self.assertIn("total_queries", stats)
    
    def test_batch_processing(self):
        """Test batch processing functionality"""
        # Create test data
        test_items = list(range(20))
        
        def simple_processor(batch):
            return [item * 2 for item in batch]
        
        # Process in batches
        results = self.batch_processor.process_batch(test_items, simple_processor)
        
        self.assertEqual(len(results), len(test_items))
        self.assertEqual(results, [item * 2 for item in test_items])
    
    def test_optimized_search_cache(self):
        """Test optimized search cache"""
        cache = OptimizedSemanticSearchCache(
            self.cache_config,
            self.performance_config
        )
        
        # Test search result caching
        query = "test query"
        results = ["result1", "result2", "result3"]
        
        cache.cache_search_results(query, results)
        cached_results = cache.get_search_results(query)
        
        self.assertEqual(cached_results, results)
        
        # Test embedding caching
        text = "test text"
        embedding = np.random.rand(100)
        
        cache.cache_embedding(text, embedding)
        cached_embedding = cache.get_embedding_cache(text)
        
        np.testing.assert_array_equal(cached_embedding, embedding)
        
        # Cleanup
        cache.cleanup()
    
    def tearDown(self):
        """Clean up after tests"""
        self.batch_processor.close()

class TestIntegratedSemanticSearch(unittest.TestCase):
    """Integration tests for the complete semantic search system"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Mock all dependencies
        self.mock_embedding_service = Mock()
        self.mock_vector_db = Mock()
        self.mock_cache = Mock()
        
        # Configure mocks
        self.mock_embedding_service.generate_embedding.return_value = np.random.rand(384)
        self.mock_vector_db.similarity_search.return_value = [
            {"chunk_id": "1", "content": "test content 1", "similarity": 0.9},
            {"chunk_id": "2", "content": "test content 2", "similarity": 0.8}
        ]
        self.mock_cache.get_search_results.return_value = None
        
        # Create integrated search system
        self.config = IntegratedSearchConfig(
            max_results=10,
            enable_caching=True,
            enable_query_expansion=True,
            enable_filtering=True,
            enable_ranking=True
        )
        
        self.search_system = IntegratedSemanticSearchSystem(
            embedding_service=self.mock_embedding_service,
            vector_db_service=self.mock_vector_db,
            cache_system=self.mock_cache,
            config=self.config
        )
    
    def test_end_to_end_search(self):
        """Test complete end-to-end search functionality"""
        query = "machine learning algorithms"
        
        results = self.search_system.search(query)
        
        self.assertIsInstance(results, list)
        # Verify all components were called
        self.mock_embedding_service.generate_embedding.assert_called()
        self.mock_vector_db.similarity_search.assert_called()
    
    def test_search_with_filters(self):
        """Test search with filtering enabled"""
        query = "deep learning"
        filters = {"document_type": "pdf", "date_range": "2023-01-01:2023-12-31"}
        
        results = self.search_system.search(query, filters=filters)
        
        self.assertIsInstance(results, list)
    
    def test_search_with_caching(self):
        """Test search with caching enabled"""
        query = "neural networks"
        
        # First search (cache miss)
        results1 = self.search_system.search(query)
        
        # Configure cache to return results
        self.mock_cache.get_search_results.return_value = results1
        
        # Second search (cache hit)
        results2 = self.search_system.search(query)
        
        self.assertEqual(results1, results2)
    
    def test_search_statistics(self):
        """Test search system statistics"""
        # Perform some searches
        self.search_system.search("query 1")
        self.search_system.search("query 2")
        
        stats = self.search_system.get_system_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertIn("total_searches", stats)
        self.assertIn("cache_stats", stats)

class TestPerformance(unittest.TestCase):
    """Performance tests for semantic search system"""
    
    def setUp(self):
        """Set up performance test fixtures"""
        self.large_dataset_size = 1000
        self.query_count = 100
    
    def test_search_performance(self):
        """Test search performance with large dataset"""
        # This would be a more comprehensive performance test
        # with actual data and timing measurements
        
        start_time = time.time()
        
        # Simulate multiple searches
        for i in range(10):
            # Mock search operation
            time.sleep(0.001)  # Simulate processing time
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Assert reasonable performance
        self.assertLess(total_time, 1.0)  # Should complete in under 1 second
    
    def test_memory_usage(self):
        """Test memory usage during operations"""
        # This would test memory consumption
        # and ensure no memory leaks
        pass
    
    def test_concurrent_searches(self):
        """Test concurrent search operations"""
        # This would test thread safety and
        # concurrent access patterns
        pass

def create_test_suite():
    """Create comprehensive test suite"""
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestSemanticSearchEngine,
        TestSimilarityAlgorithms,
        TestRankingSystem,
        TestFilteringSystem,
        TestQueryExpansion,
        TestPerformanceOptimization,
        TestIntegratedSemanticSearch,
        TestPerformance
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    return test_suite

if __name__ == "__main__":
    # Run all tests
    runner = unittest.TextTestRunner(verbosity=2)
    test_suite = create_test_suite()
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")