"""
Advanced Similarity Search Algorithms

This module provides various similarity search algorithms and metrics
for enhanced semantic search capabilities.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import math
from scipy.spatial.distance import cosine, euclidean, manhattan, chebyshev
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import faiss
import time

logger = logging.getLogger(__name__)


class SimilarityMetric(Enum):
    """Available similarity metrics"""
    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    CHEBYSHEV = "chebyshev"
    PEARSON = "pearson"
    SPEARMAN = "spearman"
    JACCARD = "jaccard"
    HAMMING = "hamming"
    MINKOWSKI = "minkowski"
    MAHALANOBIS = "mahalanobis"
    ANGULAR = "angular"
    DOT_PRODUCT = "dot_product"
    NORMALIZED_DOT_PRODUCT = "normalized_dot_product"


class SearchAlgorithm(Enum):
    """Available search algorithms"""
    BRUTE_FORCE = "brute_force"
    KD_TREE = "kd_tree"
    BALL_TREE = "ball_tree"
    LSH = "lsh"  # Locality Sensitive Hashing
    FAISS_FLAT = "faiss_flat"
    FAISS_IVF = "faiss_ivf"
    FAISS_HNSW = "faiss_hnsw"
    ANNOY = "annoy"


@dataclass
class SimilarityResult:
    """Result of similarity calculation"""
    index: int
    similarity_score: float
    distance: float
    metadata: Dict[str, Any] = None


@dataclass
class SearchConfig:
    """Configuration for similarity search"""
    metric: SimilarityMetric = SimilarityMetric.COSINE
    algorithm: SearchAlgorithm = SearchAlgorithm.BRUTE_FORCE
    top_k: int = 10
    threshold: float = 0.0
    normalize_vectors: bool = True
    use_gpu: bool = False
    batch_size: int = 1000


class AdvancedSimilaritySearcher:
    """
    Advanced similarity search with multiple algorithms and metrics
    """
    
    def __init__(self, config: SearchConfig = None):
        """
        Initialize the similarity searcher
        
        Args:
            config: Search configuration
        """
        self.config = config or SearchConfig()
        self.vectors = None
        self.index = None
        self.metadata = None
        self.is_fitted = False
        
        # Initialize FAISS if available
        self.faiss_available = True
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            self.faiss_available = False
            logger.warning("FAISS not available. Install with: pip install faiss-cpu")
        
        logger.info(f"AdvancedSimilaritySearcher initialized with {self.config.metric.value} metric")
    
    def fit(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None):
        """
        Fit the searcher with a collection of vectors
        
        Args:
            vectors: Array of vectors to index
            metadata: Optional metadata for each vector
        """
        if vectors.ndim != 2:
            raise ValueError("Vectors must be a 2D array")
        
        self.vectors = vectors.astype(np.float32)
        self.metadata = metadata or [{}] * len(vectors)
        
        if self.config.normalize_vectors:
            self.vectors = self._normalize_vectors(self.vectors)
        
        # Build index based on algorithm
        self._build_index()
        self.is_fitted = True
        
        logger.info(f"Fitted searcher with {len(vectors)} vectors using {self.config.algorithm.value}")
    
    def search(self, query_vector: np.ndarray, top_k: int = None) -> List[SimilarityResult]:
        """
        Search for similar vectors
        
        Args:
            query_vector: Query vector
            top_k: Number of results to return
            
        Returns:
            List of similarity results
        """
        if not self.is_fitted:
            raise ValueError("Searcher must be fitted before searching")
        
        top_k = top_k or self.config.top_k
        
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        query_vector = query_vector.astype(np.float32)
        
        if self.config.normalize_vectors:
            query_vector = self._normalize_vectors(query_vector)
        
        # Perform search based on algorithm
        if self.config.algorithm == SearchAlgorithm.BRUTE_FORCE:
            return self._brute_force_search(query_vector[0], top_k)
        elif self.config.algorithm in [SearchAlgorithm.FAISS_FLAT, SearchAlgorithm.FAISS_IVF, SearchAlgorithm.FAISS_HNSW]:
            return self._faiss_search(query_vector, top_k)
        else:
            # Fallback to brute force
            return self._brute_force_search(query_vector[0], top_k)
    
    def batch_search(self, query_vectors: np.ndarray, top_k: int = None) -> List[List[SimilarityResult]]:
        """
        Batch search for multiple query vectors
        
        Args:
            query_vectors: Array of query vectors
            top_k: Number of results per query
            
        Returns:
            List of result lists
        """
        if not self.is_fitted:
            raise ValueError("Searcher must be fitted before searching")
        
        top_k = top_k or self.config.top_k
        results = []
        
        batch_size = self.config.batch_size
        for i in range(0, len(query_vectors), batch_size):
            batch = query_vectors[i:i + batch_size]
            
            if self.config.algorithm in [SearchAlgorithm.FAISS_FLAT, SearchAlgorithm.FAISS_IVF, SearchAlgorithm.FAISS_HNSW]:
                batch_results = self._faiss_batch_search(batch, top_k)
                results.extend(batch_results)
            else:
                # Process individually for other algorithms
                for query_vector in batch:
                    result = self.search(query_vector, top_k)
                    results.append(result)
        
        return results
    
    def _build_index(self):
        """Build search index based on algorithm"""
        if self.config.algorithm == SearchAlgorithm.FAISS_FLAT and self.faiss_available:
            self._build_faiss_flat_index()
        elif self.config.algorithm == SearchAlgorithm.FAISS_IVF and self.faiss_available:
            self._build_faiss_ivf_index()
        elif self.config.algorithm == SearchAlgorithm.FAISS_HNSW and self.faiss_available:
            self._build_faiss_hnsw_index()
        else:
            # No index needed for brute force
            self.index = None
    
    def _build_faiss_flat_index(self):
        """Build FAISS flat index"""
        dimension = self.vectors.shape[1]
        
        if self.config.metric == SimilarityMetric.COSINE:
            self.index = self.faiss.IndexFlatIP(dimension)  # Inner product for cosine
        else:
            self.index = self.faiss.IndexFlatL2(dimension)  # L2 distance
        
        if self.config.use_gpu and self.faiss.get_num_gpus() > 0:
            self.index = self.faiss.index_cpu_to_gpu(self.faiss.StandardGpuResources(), 0, self.index)
        
        self.index.add(self.vectors)
    
    def _build_faiss_ivf_index(self):
        """Build FAISS IVF index for faster search"""
        dimension = self.vectors.shape[1]
        nlist = min(100, int(math.sqrt(len(self.vectors))))  # Number of clusters
        
        if self.config.metric == SimilarityMetric.COSINE:
            quantizer = self.faiss.IndexFlatIP(dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        else:
            quantizer = self.faiss.IndexFlatL2(dimension)
            self.index = self.faiss.IndexIVFFlat(quantizer, dimension, nlist)
        
        if self.config.use_gpu and self.faiss.get_num_gpus() > 0:
            self.index = self.faiss.index_cpu_to_gpu(self.faiss.StandardGpuResources(), 0, self.index)
        
        self.index.train(self.vectors)
        self.index.add(self.vectors)
        self.index.nprobe = min(10, nlist)  # Number of clusters to search
    
    def _build_faiss_hnsw_index(self):
        """Build FAISS HNSW index for high-dimensional data"""
        dimension = self.vectors.shape[1]
        M = 32  # Number of connections per node
        
        self.index = self.faiss.IndexHNSWFlat(dimension, M)
        self.index.hnsw.efConstruction = 200
        self.index.hnsw.efSearch = 100
        
        self.index.add(self.vectors)
    
    def _brute_force_search(self, query_vector: np.ndarray, top_k: int) -> List[SimilarityResult]:
        """Brute force similarity search"""
        similarities = []
        
        for i, vector in enumerate(self.vectors):
            similarity = self._calculate_similarity(query_vector, vector)
            
            if similarity >= self.config.threshold:
                similarities.append(SimilarityResult(
                    index=i,
                    similarity_score=similarity,
                    distance=1.0 - similarity if similarity <= 1.0 else 0.0,
                    metadata=self.metadata[i] if self.metadata else None
                ))
        
        # Sort by similarity score (descending)
        similarities.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return similarities[:top_k]
    
    def _faiss_search(self, query_vectors: np.ndarray, top_k: int) -> List[SimilarityResult]:
        """FAISS-based search"""
        if not self.faiss_available or self.index is None:
            return self._brute_force_search(query_vectors[0], top_k)
        
        distances, indices = self.index.search(query_vectors, top_k)
        
        results = []
        for i, (dist_row, idx_row) in enumerate(zip(distances, indices)):
            query_results = []
            for dist, idx in zip(dist_row, idx_row):
                if idx >= 0:  # Valid index
                    # Convert distance to similarity
                    if self.config.metric == SimilarityMetric.COSINE:
                        similarity = float(dist)  # FAISS returns inner product for cosine
                    else:
                        similarity = 1.0 / (1.0 + float(dist))  # Convert L2 distance to similarity
                    
                    if similarity >= self.config.threshold:
                        query_results.append(SimilarityResult(
                            index=int(idx),
                            similarity_score=similarity,
                            distance=float(dist),
                            metadata=self.metadata[idx] if self.metadata else None
                        ))
            
            results.append(query_results)
        
        return results[0] if len(results) == 1 else results
    
    def _faiss_batch_search(self, query_vectors: np.ndarray, top_k: int) -> List[List[SimilarityResult]]:
        """FAISS batch search"""
        if not self.faiss_available or self.index is None:
            return [self._brute_force_search(qv, top_k) for qv in query_vectors]
        
        if query_vectors.ndim == 1:
            query_vectors = query_vectors.reshape(1, -1)
        
        query_vectors = query_vectors.astype(np.float32)
        
        if self.config.normalize_vectors:
            query_vectors = self._normalize_vectors(query_vectors)
        
        distances, indices = self.index.search(query_vectors, top_k)
        
        results = []
        for dist_row, idx_row in zip(distances, indices):
            query_results = []
            for dist, idx in zip(dist_row, idx_row):
                if idx >= 0:  # Valid index
                    # Convert distance to similarity
                    if self.config.metric == SimilarityMetric.COSINE:
                        similarity = float(dist)
                    else:
                        similarity = 1.0 / (1.0 + float(dist))
                    
                    if similarity >= self.config.threshold:
                        query_results.append(SimilarityResult(
                            index=int(idx),
                            similarity_score=similarity,
                            distance=float(dist),
                            metadata=self.metadata[idx] if self.metadata else None
                        ))
            
            results.append(query_results)
        
        return results
    
    def _calculate_similarity(self, vector1: np.ndarray, vector2: np.ndarray) -> float:
        """Calculate similarity between two vectors"""
        try:
            if self.config.metric == SimilarityMetric.COSINE:
                return float(1 - cosine(vector1, vector2))
            
            elif self.config.metric == SimilarityMetric.EUCLIDEAN:
                dist = euclidean(vector1, vector2)
                return 1.0 / (1.0 + dist)
            
            elif self.config.metric == SimilarityMetric.MANHATTAN:
                dist = manhattan(vector1, vector2)
                return 1.0 / (1.0 + dist)
            
            elif self.config.metric == SimilarityMetric.CHEBYSHEV:
                dist = chebyshev(vector1, vector2)
                return 1.0 / (1.0 + dist)
            
            elif self.config.metric == SimilarityMetric.PEARSON:
                corr, _ = pearsonr(vector1, vector2)
                return float(corr) if not np.isnan(corr) else 0.0
            
            elif self.config.metric == SimilarityMetric.SPEARMAN:
                corr, _ = spearmanr(vector1, vector2)
                return float(corr) if not np.isnan(corr) else 0.0
            
            elif self.config.metric == SimilarityMetric.DOT_PRODUCT:
                return float(np.dot(vector1, vector2))
            
            elif self.config.metric == SimilarityMetric.NORMALIZED_DOT_PRODUCT:
                norm1 = np.linalg.norm(vector1)
                norm2 = np.linalg.norm(vector2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(vector1, vector2) / (norm1 * norm2))
            
            elif self.config.metric == SimilarityMetric.ANGULAR:
                cos_sim = 1 - cosine(vector1, vector2)
                return 1.0 - (2.0 * np.arccos(np.clip(cos_sim, -1, 1)) / np.pi)
            
            else:
                # Default to cosine
                return float(1 - cosine(vector1, vector2))
                
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return 0.0
    
    def _normalize_vectors(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length"""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        return vectors / norms
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get searcher statistics"""
        if not self.is_fitted:
            return {"fitted": False}
        
        return {
            "fitted": True,
            "num_vectors": len(self.vectors),
            "vector_dimension": self.vectors.shape[1],
            "metric": self.config.metric.value,
            "algorithm": self.config.algorithm.value,
            "normalized": self.config.normalize_vectors,
            "faiss_available": self.faiss_available,
            "gpu_enabled": self.config.use_gpu
        }


class HybridSimilaritySearcher:
    """
    Hybrid searcher that combines multiple similarity metrics
    """
    
    def __init__(self, metrics: List[SimilarityMetric], weights: List[float] = None):
        """
        Initialize hybrid searcher
        
        Args:
            metrics: List of similarity metrics to combine
            weights: Weights for each metric (default: equal weights)
        """
        self.metrics = metrics
        self.weights = weights or [1.0 / len(metrics)] * len(metrics)
        
        if len(self.weights) != len(self.metrics):
            raise ValueError("Number of weights must match number of metrics")
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        self.searchers = {}
        self.is_fitted = False
    
    def fit(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None):
        """Fit all searchers"""
        for metric in self.metrics:
            config = SearchConfig(metric=metric)
            searcher = AdvancedSimilaritySearcher(config)
            searcher.fit(vectors, metadata)
            self.searchers[metric] = searcher
        
        self.is_fitted = True
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[SimilarityResult]:
        """
        Hybrid search combining multiple metrics
        """
        if not self.is_fitted:
            raise ValueError("Searcher must be fitted before searching")
        
        # Get results from each searcher
        all_results = {}
        for metric, weight in zip(self.metrics, self.weights):
            searcher = self.searchers[metric]
            results = searcher.search(query_vector, top_k * 2)  # Get more candidates
            
            for result in results:
                if result.index not in all_results:
                    all_results[result.index] = {
                        'index': result.index,
                        'metadata': result.metadata,
                        'scores': {},
                        'weighted_score': 0.0
                    }
                
                all_results[result.index]['scores'][metric] = result.similarity_score
                all_results[result.index]['weighted_score'] += result.similarity_score * weight
        
        # Convert to SimilarityResult objects and sort
        final_results = []
        for data in all_results.values():
            result = SimilarityResult(
                index=data['index'],
                similarity_score=data['weighted_score'],
                distance=1.0 - data['weighted_score'],
                metadata=data['metadata']
            )
            final_results.append(result)
        
        # Sort by weighted score
        final_results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return final_results[:top_k]


class AdaptiveSimilaritySearcher:
    """
    Adaptive searcher that selects the best metric based on data characteristics
    """
    
    def __init__(self):
        self.searchers = {}
        self.best_metric = None
        self.is_fitted = False
    
    def fit(self, vectors: np.ndarray, metadata: List[Dict[str, Any]] = None, 
            validation_queries: np.ndarray = None):
        """
        Fit and select best metric based on validation performance
        """
        metrics_to_test = [
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.MANHATTAN,
            SimilarityMetric.DOT_PRODUCT
        ]
        
        # Fit all searchers
        for metric in metrics_to_test:
            config = SearchConfig(metric=metric)
            searcher = AdvancedSimilaritySearcher(config)
            searcher.fit(vectors, metadata)
            self.searchers[metric] = searcher
        
        # Select best metric based on validation if provided
        if validation_queries is not None:
            self.best_metric = self._select_best_metric(validation_queries)
        else:
            # Default to cosine similarity
            self.best_metric = SimilarityMetric.COSINE
        
        self.is_fitted = True
        logger.info(f"Selected best metric: {self.best_metric.value}")
    
    def _select_best_metric(self, validation_queries: np.ndarray) -> SimilarityMetric:
        """Select best metric based on validation performance"""
        # This is a simplified selection - in practice, you'd use labeled data
        # For now, we'll use the metric that gives the most diverse results
        
        diversity_scores = {}
        
        for metric, searcher in self.searchers.items():
            total_diversity = 0.0
            
            for query in validation_queries[:10]:  # Sample validation queries
                results = searcher.search(query, top_k=10)
                if len(results) > 1:
                    # Calculate diversity as variance in similarity scores
                    scores = [r.similarity_score for r in results]
                    diversity = np.var(scores)
                    total_diversity += diversity
            
            diversity_scores[metric] = total_diversity
        
        # Select metric with highest diversity (better discrimination)
        best_metric = max(diversity_scores.keys(), key=lambda k: diversity_scores[k])
        return best_metric
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[SimilarityResult]:
        """Search using the best selected metric"""
        if not self.is_fitted:
            raise ValueError("Searcher must be fitted before searching")
        
        return self.searchers[self.best_metric].search(query_vector, top_k)


# Utility functions

def compare_similarity_metrics(vectors: np.ndarray, 
                             query_vector: np.ndarray,
                             metrics: List[SimilarityMetric] = None,
                             top_k: int = 10) -> Dict[str, List[SimilarityResult]]:
    """
    Compare different similarity metrics on the same data
    
    Args:
        vectors: Collection of vectors
        query_vector: Query vector
        metrics: Metrics to compare (default: all available)
        top_k: Number of results per metric
        
    Returns:
        Dictionary mapping metric names to results
    """
    if metrics is None:
        metrics = [
            SimilarityMetric.COSINE,
            SimilarityMetric.EUCLIDEAN,
            SimilarityMetric.MANHATTAN,
            SimilarityMetric.DOT_PRODUCT
        ]
    
    results = {}
    
    for metric in metrics:
        config = SearchConfig(metric=metric)
        searcher = AdvancedSimilaritySearcher(config)
        searcher.fit(vectors)
        
        search_results = searcher.search(query_vector, top_k)
        results[metric.value] = search_results
    
    return results


def benchmark_search_algorithms(vectors: np.ndarray,
                               query_vectors: np.ndarray,
                               algorithms: List[SearchAlgorithm] = None,
                               top_k: int = 10) -> Dict[str, Dict[str, float]]:
    """
    Benchmark different search algorithms
    
    Args:
        vectors: Collection of vectors to index
        query_vectors: Query vectors for testing
        algorithms: Algorithms to benchmark
        top_k: Number of results per query
        
    Returns:
        Dictionary with performance metrics for each algorithm
    """
    if algorithms is None:
        algorithms = [SearchAlgorithm.BRUTE_FORCE, SearchAlgorithm.FAISS_FLAT]
    
    results = {}
    
    for algorithm in algorithms:
        config = SearchConfig(algorithm=algorithm)
        searcher = AdvancedSimilaritySearcher(config)
        
        # Measure fit time
        start_time = time.time()
        searcher.fit(vectors)
        fit_time = time.time() - start_time
        
        # Measure search time
        start_time = time.time()
        for query_vector in query_vectors:
            searcher.search(query_vector, top_k)
        search_time = time.time() - start_time
        
        results[algorithm.value] = {
            "fit_time": fit_time,
            "search_time": search_time,
            "avg_search_time": search_time / len(query_vectors),
            "throughput": len(query_vectors) / search_time if search_time > 0 else float('inf')
        }
    
    return results