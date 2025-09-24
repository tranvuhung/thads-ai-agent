"""
Performance Optimization and Caching System for Semantic Search

This module provides comprehensive performance optimization features including
caching, indexing, batch processing, and monitoring for semantic search systems.
"""

import time
import hashlib
import pickle
import json
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import numpy as np
from datetime import datetime, timedelta
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import redis
import sqlite3
import psutil
import gc

class CacheType(Enum):
    """Types of caching strategies"""
    MEMORY = "memory"
    REDIS = "redis"
    DISK = "disk"
    HYBRID = "hybrid"

class OptimizationStrategy(Enum):
    """Performance optimization strategies"""
    LAZY_LOADING = "lazy_loading"
    BATCH_PROCESSING = "batch_processing"
    PARALLEL_PROCESSING = "parallel_processing"
    INDEX_OPTIMIZATION = "index_optimization"
    MEMORY_OPTIMIZATION = "memory_optimization"
    QUERY_OPTIMIZATION = "query_optimization"

@dataclass
class CacheConfig:
    """Configuration for caching system"""
    cache_type: CacheType = CacheType.MEMORY
    max_size: int = 1000
    ttl_seconds: int = 3600
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    disk_cache_dir: str = "./cache"
    compression_enabled: bool = True
    auto_cleanup: bool = True

@dataclass
class PerformanceConfig:
    """Configuration for performance optimization"""
    strategies: List[OptimizationStrategy] = field(default_factory=lambda: [
        OptimizationStrategy.BATCH_PROCESSING,
        OptimizationStrategy.PARALLEL_PROCESSING,
        OptimizationStrategy.MEMORY_OPTIMIZATION
    ])
    max_workers: int = 4
    batch_size: int = 100
    enable_monitoring: bool = True
    memory_threshold: float = 0.8
    cpu_threshold: float = 0.9

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    timestamp: datetime
    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    size_bytes: int = 0

@dataclass
class PerformanceMetrics:
    """Performance monitoring metrics"""
    cache_hits: int = 0
    cache_misses: int = 0
    total_queries: int = 0
    avg_response_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    active_connections: int = 0
    error_count: int = 0

class LRUCache:
    """Thread-safe LRU cache implementation"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        with self.lock:
            if key not in self.cache:
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if self._is_expired(entry):
                del self.cache[key]
                return None
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        with self.lock:
            # Calculate size
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=datetime.now(),
                size_bytes=size_bytes
            )
            
            # Remove if exists
            if key in self.cache:
                del self.cache[key]
            
            # Add new entry
            self.cache[key] = entry
            
            # Evict if necessary
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        with self.lock:
            self.cache.clear()
    
    def size(self) -> int:
        """Get cache size"""
        with self.lock:
            return len(self.cache)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired"""
        return (datetime.now() - entry.timestamp).total_seconds() > self.ttl_seconds
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return 0

class RedisCache:
    """Redis-based cache implementation"""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, ttl_seconds: int = 3600):
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=False)
            self.ttl_seconds = ttl_seconds
            self.available = True
        except:
            self.available = False
            logging.warning("Redis not available, falling back to memory cache")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from Redis cache"""
        if not self.available:
            return None
        
        try:
            data = self.redis_client.get(key)
            if data:
                return pickle.loads(data)
            return None
        except Exception as e:
            logging.error(f"Redis get error: {e}")
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in Redis cache"""
        if not self.available:
            return
        
        try:
            data = pickle.dumps(value)
            self.redis_client.setex(key, self.ttl_seconds, data)
        except Exception as e:
            logging.error(f"Redis put error: {e}")
    
    def clear(self) -> None:
        """Clear all cache entries"""
        if not self.available:
            return
        
        try:
            self.redis_client.flushdb()
        except Exception as e:
            logging.error(f"Redis clear error: {e}")

class HybridCache:
    """Hybrid cache combining memory and Redis"""
    
    def __init__(self, cache_config: CacheConfig):
        self.memory_cache = LRUCache(
            max_size=cache_config.max_size // 2,
            ttl_seconds=cache_config.ttl_seconds
        )
        self.redis_cache = RedisCache(
            host=cache_config.redis_host,
            port=cache_config.redis_port,
            db=cache_config.redis_db,
            ttl_seconds=cache_config.ttl_seconds
        )
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from hybrid cache"""
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try Redis cache
        value = self.redis_cache.get(key)
        if value is not None:
            # Store in memory cache for faster access
            self.memory_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any) -> None:
        """Put value in hybrid cache"""
        # Store in both caches
        self.memory_cache.put(key, value)
        self.redis_cache.put(key, value)
    
    def clear(self) -> None:
        """Clear all cache entries"""
        self.memory_cache.clear()
        self.redis_cache.clear()

class CacheManager:
    """Centralized cache management system"""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache = self._create_cache()
        self.metrics = PerformanceMetrics()
        
        # Start cleanup thread if auto cleanup is enabled
        if config.auto_cleanup:
            self._start_cleanup_thread()
    
    def _create_cache(self):
        """Create cache based on configuration"""
        if self.config.cache_type == CacheType.MEMORY:
            return LRUCache(self.config.max_size, self.config.ttl_seconds)
        elif self.config.cache_type == CacheType.REDIS:
            return RedisCache(
                self.config.redis_host,
                self.config.redis_port,
                self.config.redis_db,
                self.config.ttl_seconds
            )
        elif self.config.cache_type == CacheType.HYBRID:
            return HybridCache(self.config)
        else:
            return LRUCache(self.config.max_size, self.config.ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache with metrics tracking"""
        start_time = time.time()
        
        value = self.cache.get(key)
        
        # Update metrics
        if value is not None:
            self.metrics.cache_hits += 1
        else:
            self.metrics.cache_misses += 1
        
        self.metrics.total_queries += 1
        response_time = time.time() - start_time
        self._update_avg_response_time(response_time)
        
        return value
    
    def put(self, key: str, value: Any) -> None:
        """Put value in cache"""
        self.cache.put(key, value)
    
    def clear(self) -> None:
        """Clear cache"""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0.0
        if self.metrics.total_queries > 0:
            hit_rate = self.metrics.cache_hits / self.metrics.total_queries
        
        return {
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "total_queries": self.metrics.total_queries,
            "hit_rate": hit_rate,
            "avg_response_time": self.metrics.avg_response_time,
            "cache_size": self.cache.size() if hasattr(self.cache, 'size') else 0
        }
    
    def _update_avg_response_time(self, response_time: float):
        """Update average response time"""
        if self.metrics.total_queries == 1:
            self.metrics.avg_response_time = response_time
        else:
            # Exponential moving average
            alpha = 0.1
            self.metrics.avg_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.metrics.avg_response_time
            )
    
    def _start_cleanup_thread(self):
        """Start background cleanup thread"""
        def cleanup_worker():
            while True:
                time.sleep(300)  # Run every 5 minutes
                try:
                    # Force garbage collection
                    gc.collect()
                    
                    # Log cache statistics
                    stats = self.get_cache_stats()
                    logging.info(f"Cache stats: {stats}")
                    
                except Exception as e:
                    logging.error(f"Cleanup error: {e}")
        
        cleanup_thread = threading.Thread(target=cleanup_worker, daemon=True)
        cleanup_thread.start()

class QueryHasher:
    """Generate consistent hashes for queries"""
    
    @staticmethod
    def hash_query(query: str, filters: Dict = None, config: Dict = None) -> str:
        """Generate hash for query with filters and config"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Create hash input
        hash_input = {
            "query": normalized_query,
            "filters": filters or {},
            "config": config or {}
        }
        
        # Convert to string and hash
        hash_string = json.dumps(hash_input, sort_keys=True)
        return hashlib.md5(hash_string.encode()).hexdigest()

class BatchProcessor:
    """Batch processing for improved performance"""
    
    def __init__(self, batch_size: int = 100, max_workers: int = 4):
        self.batch_size = batch_size
        self.max_workers = max_workers
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
    
    def process_batch(self, items: List[Any], processor_func: Callable, use_processes: bool = False) -> List[Any]:
        """Process items in batches"""
        results = []
        
        # Split into batches
        batches = [items[i:i + self.batch_size] for i in range(0, len(items), self.batch_size)]
        
        # Choose executor
        executor = self.process_pool if use_processes else self.thread_pool
        
        # Process batches
        futures = []
        for batch in batches:
            future = executor.submit(processor_func, batch)
            futures.append(future)
        
        # Collect results
        for future in futures:
            try:
                batch_results = future.result(timeout=30)
                results.extend(batch_results)
            except Exception as e:
                logging.error(f"Batch processing error: {e}")
        
        return results
    
    def close(self):
        """Close thread and process pools"""
        self.thread_pool.shutdown(wait=True)
        self.process_pool.shutdown(wait=True)

class PerformanceMonitor:
    """System performance monitoring"""
    
    def __init__(self):
        self.metrics = PerformanceMetrics()
        self.monitoring_active = False
        self.monitor_thread = None
    
    def start_monitoring(self, interval: int = 60):
        """Start performance monitoring"""
        self.monitoring_active = True
        
        def monitor_worker():
            while self.monitoring_active:
                try:
                    # Update system metrics
                    self.metrics.memory_usage = psutil.virtual_memory().percent / 100.0
                    self.metrics.cpu_usage = psutil.cpu_percent() / 100.0
                    
                    # Log metrics if thresholds exceeded
                    if self.metrics.memory_usage > 0.8:
                        logging.warning(f"High memory usage: {self.metrics.memory_usage:.2%}")
                    
                    if self.metrics.cpu_usage > 0.9:
                        logging.warning(f"High CPU usage: {self.metrics.cpu_usage:.2%}")
                    
                    time.sleep(interval)
                    
                except Exception as e:
                    logging.error(f"Monitoring error: {e}")
        
        self.monitor_thread = threading.Thread(target=monitor_worker, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop performance monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def get_metrics(self) -> PerformanceMetrics:
        """Get current performance metrics"""
        return self.metrics

class OptimizedSemanticSearchCache:
    """Optimized caching layer for semantic search"""
    
    def __init__(self, cache_config: CacheConfig, performance_config: PerformanceConfig):
        self.cache_manager = CacheManager(cache_config)
        self.performance_config = performance_config
        self.batch_processor = BatchProcessor(
            batch_size=performance_config.batch_size,
            max_workers=performance_config.max_workers
        )
        self.performance_monitor = PerformanceMonitor()
        
        # Start monitoring if enabled
        if performance_config.enable_monitoring:
            self.performance_monitor.start_monitoring()
    
    def get_search_results(self, query: str, filters: Dict = None, config: Dict = None) -> Optional[Any]:
        """Get cached search results"""
        cache_key = QueryHasher.hash_query(query, filters, config)
        return self.cache_manager.get(cache_key)
    
    def cache_search_results(self, query: str, results: Any, filters: Dict = None, config: Dict = None):
        """Cache search results"""
        cache_key = QueryHasher.hash_query(query, filters, config)
        self.cache_manager.put(cache_key, results)
    
    def get_embedding_cache(self, text: str) -> Optional[np.ndarray]:
        """Get cached embedding"""
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        return self.cache_manager.get(cache_key)
    
    def cache_embedding(self, text: str, embedding: np.ndarray):
        """Cache embedding"""
        cache_key = f"embedding:{hashlib.md5(text.encode()).hexdigest()}"
        self.cache_manager.put(cache_key, embedding)
    
    def process_batch_embeddings(self, texts: List[str], embedding_func: Callable) -> List[np.ndarray]:
        """Process embeddings in batches with caching"""
        results = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached_embedding = self.get_embedding_cache(text)
            if cached_embedding is not None:
                results.append((i, cached_embedding))
            else:
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Process uncached texts in batches
        if uncached_texts:
            def batch_embedding_processor(batch_texts):
                return [embedding_func(text) for text in batch_texts]
            
            uncached_embeddings = self.batch_processor.process_batch(
                uncached_texts, batch_embedding_processor
            )
            
            # Cache new embeddings and add to results
            for i, (text, embedding) in enumerate(zip(uncached_texts, uncached_embeddings)):
                self.cache_embedding(text, embedding)
                original_index = uncached_indices[i]
                results.append((original_index, embedding))
        
        # Sort results by original index
        results.sort(key=lambda x: x[0])
        return [embedding for _, embedding in results]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        cache_stats = self.cache_manager.get_cache_stats()
        performance_metrics = self.performance_monitor.get_metrics()
        
        return {
            "cache": cache_stats,
            "performance": {
                "memory_usage": performance_metrics.memory_usage,
                "cpu_usage": performance_metrics.cpu_usage,
                "error_count": performance_metrics.error_count
            }
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.performance_monitor.stop_monitoring()
        self.batch_processor.close()
        self.cache_manager.clear()

# Utility functions
def create_default_cache_config() -> CacheConfig:
    """Create default cache configuration"""
    return CacheConfig(
        cache_type=CacheType.HYBRID,
        max_size=1000,
        ttl_seconds=3600,
        compression_enabled=True,
        auto_cleanup=True
    )

def create_default_performance_config() -> PerformanceConfig:
    """Create default performance configuration"""
    return PerformanceConfig(
        strategies=[
            OptimizationStrategy.BATCH_PROCESSING,
            OptimizationStrategy.PARALLEL_PROCESSING,
            OptimizationStrategy.MEMORY_OPTIMIZATION
        ],
        max_workers=4,
        batch_size=100,
        enable_monitoring=True
    )

def create_optimized_search_cache() -> OptimizedSemanticSearchCache:
    """Create optimized search cache with default configurations"""
    cache_config = create_default_cache_config()
    performance_config = create_default_performance_config()
    
    return OptimizedSemanticSearchCache(cache_config, performance_config)