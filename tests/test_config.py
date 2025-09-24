"""
Test Configuration and Utilities for Semantic Search Tests

This module provides configuration, fixtures, and utility functions
for testing the semantic search system.
"""

import os
import tempfile
import shutil
import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import json
import sqlite3
from unittest.mock import Mock

@dataclass
class TestConfig:
    """Configuration for test environment"""
    test_data_dir: str = "./test_data"
    temp_dir: str = "./temp_test"
    mock_embedding_dim: int = 384
    test_document_count: int = 100
    test_query_count: int = 50
    enable_performance_tests: bool = False
    enable_integration_tests: bool = True

class TestDataGenerator:
    """Generate test data for semantic search testing"""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.temp_dir = None
    
    def setup_test_environment(self):
        """Set up test environment with temporary directories"""
        self.temp_dir = tempfile.mkdtemp(prefix="semantic_search_test_")
        
        # Create test data directory
        test_data_dir = os.path.join(self.temp_dir, "test_data")
        os.makedirs(test_data_dir, exist_ok=True)
        
        return self.temp_dir
    
    def cleanup_test_environment(self):
        """Clean up test environment"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def generate_test_documents(self, count: int = None) -> List[Dict[str, Any]]:
        """Generate test documents for testing"""
        if count is None:
            count = self.config.test_document_count
        
        documents = []
        
        # Sample document templates
        templates = [
            "This is a document about {topic}. It contains information about {subtopic} and {detail}.",
            "Research paper on {topic}: {subtopic} analysis and {detail} implementation.",
            "Technical guide for {topic} focusing on {subtopic} with practical {detail}.",
            "Introduction to {topic} covering {subtopic} fundamentals and {detail} applications.",
            "Advanced {topic} concepts including {subtopic} theory and {detail} examples."
        ]
        
        topics = [
            "machine learning", "artificial intelligence", "data science", "natural language processing",
            "computer vision", "deep learning", "neural networks", "reinforcement learning",
            "data mining", "big data", "cloud computing", "software engineering"
        ]
        
        subtopics = [
            "algorithms", "models", "frameworks", "techniques", "methods", "approaches",
            "architectures", "systems", "tools", "libraries", "platforms", "services"
        ]
        
        details = [
            "optimization", "evaluation", "training", "testing", "deployment", "monitoring",
            "scaling", "performance", "accuracy", "efficiency", "robustness", "reliability"
        ]
        
        for i in range(count):
            template = templates[i % len(templates)]
            topic = topics[i % len(topics)]
            subtopic = subtopics[i % len(subtopics)]
            detail = details[i % len(details)]
            
            content = template.format(topic=topic, subtopic=subtopic, detail=detail)
            
            document = {
                "chunk_id": f"test_chunk_{i}",
                "content": content,
                "metadata": {
                    "document_id": f"test_doc_{i // 5}",  # 5 chunks per document
                    "chunk_index": i % 5,
                    "document_type": "pdf" if i % 2 == 0 else "txt",
                    "author": f"author_{i % 10}",
                    "date": f"2023-{(i % 12) + 1:02d}-{((i % 28) + 1):02d}",
                    "topic": topic,
                    "subtopic": subtopic,
                    "quality_score": 0.5 + (i % 5) * 0.1,
                    "page_number": (i % 5) + 1,
                    "word_count": len(content.split()),
                    "language": "en"
                }
            }
            
            documents.append(document)
        
        return documents
    
    def generate_test_queries(self, count: int = None) -> List[Dict[str, Any]]:
        """Generate test queries for testing"""
        if count is None:
            count = self.config.test_query_count
        
        queries = []
        
        query_templates = [
            "What is {topic}?",
            "How to implement {topic} {subtopic}?",
            "Best practices for {topic} {detail}",
            "Comparison of {topic} {subtopic} methods",
            "Tutorial on {topic} {subtopic} and {detail}",
            "{topic} {subtopic} examples and use cases",
            "Advanced {topic} techniques for {detail}",
            "Introduction to {topic} {subtopic}",
            "{topic} {subtopic} optimization strategies",
            "Common challenges in {topic} {detail}"
        ]
        
        topics = [
            "machine learning", "deep learning", "neural networks", "data science",
            "artificial intelligence", "natural language processing", "computer vision"
        ]
        
        subtopics = [
            "algorithms", "models", "training", "evaluation", "deployment", "optimization"
        ]
        
        details = [
            "performance", "accuracy", "efficiency", "scalability", "robustness"
        ]
        
        for i in range(count):
            template = query_templates[i % len(query_templates)]
            topic = topics[i % len(topics)]
            subtopic = subtopics[i % len(subtopics)]
            detail = details[i % len(details)]
            
            query_text = template.format(topic=topic, subtopic=subtopic, detail=detail)
            
            query = {
                "query_id": f"test_query_{i}",
                "text": query_text,
                "expected_topics": [topic, subtopic],
                "filters": {
                    "document_type": "pdf" if i % 3 == 0 else None,
                    "author": f"author_{i % 10}" if i % 4 == 0 else None,
                    "date_range": "2023-01-01:2023-06-30" if i % 5 == 0 else None
                },
                "expected_result_count": min(10, 5 + (i % 10)),
                "relevance_threshold": 0.5 + (i % 3) * 0.1
            }
            
            queries.append(query)
        
        return queries
    
    def generate_test_embeddings(self, documents: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
        """Generate mock embeddings for test documents"""
        embeddings = {}
        
        for doc in documents:
            # Generate deterministic but varied embeddings based on content
            content_hash = hash(doc["content"]) % 1000000
            np.random.seed(content_hash)
            
            # Create embedding with some structure based on content
            embedding = np.random.rand(self.config.mock_embedding_dim)
            
            # Add some topic-based clustering
            topic = doc["metadata"].get("topic", "")
            if "machine learning" in topic:
                embedding[0:50] += 0.3
            elif "deep learning" in topic:
                embedding[50:100] += 0.3
            elif "data science" in topic:
                embedding[100:150] += 0.3
            
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            
            embeddings[doc["chunk_id"]] = embedding
        
        return embeddings

class MockServiceFactory:
    """Factory for creating mock services for testing"""
    
    @staticmethod
    def create_mock_embedding_service(embeddings: Dict[str, np.ndarray] = None):
        """Create mock embedding service"""
        mock_service = Mock()
        
        if embeddings:
            def mock_generate_embedding(text: str) -> np.ndarray:
                # Generate deterministic embedding based on text hash
                text_hash = hash(text) % 1000000
                np.random.seed(text_hash)
                embedding = np.random.rand(384)
                return embedding / np.linalg.norm(embedding)
            
            mock_service.generate_embedding.side_effect = mock_generate_embedding
        else:
            mock_service.generate_embedding.return_value = np.random.rand(384)
        
        def mock_generate_embeddings_batch(texts: List[str]) -> List[np.ndarray]:
            return [mock_service.generate_embedding(text) for text in texts]
        
        mock_service.generate_embeddings_batch.side_effect = mock_generate_embeddings_batch
        
        return mock_service
    
    @staticmethod
    def create_mock_vector_database(documents: List[Dict[str, Any]] = None, 
                                  embeddings: Dict[str, np.ndarray] = None):
        """Create mock vector database service"""
        mock_db = Mock()
        
        if documents and embeddings:
            def mock_similarity_search(query_embedding: np.ndarray, 
                                     top_k: int = 10, 
                                     filters: Dict = None) -> List[Dict[str, Any]]:
                results = []
                
                for doc in documents:
                    chunk_id = doc["chunk_id"]
                    if chunk_id in embeddings:
                        doc_embedding = embeddings[chunk_id]
                        
                        # Calculate cosine similarity
                        similarity = np.dot(query_embedding, doc_embedding)
                        
                        # Apply filters if provided
                        if filters:
                            if not MockServiceFactory._passes_filters(doc, filters):
                                continue
                        
                        result = {
                            **doc,
                            "similarity": float(similarity)
                        }
                        results.append(result)
                
                # Sort by similarity and return top_k
                results.sort(key=lambda x: x["similarity"], reverse=True)
                return results[:top_k]
            
            mock_db.similarity_search.side_effect = mock_similarity_search
        else:
            # Default mock behavior
            mock_db.similarity_search.return_value = [
                {
                    "chunk_id": "mock_chunk_1",
                    "content": "mock content 1",
                    "similarity": 0.9,
                    "metadata": {"document_id": "mock_doc_1"}
                },
                {
                    "chunk_id": "mock_chunk_2", 
                    "content": "mock content 2",
                    "similarity": 0.8,
                    "metadata": {"document_id": "mock_doc_2"}
                }
            ]
        
        return mock_db
    
    @staticmethod
    def _passes_filters(document: Dict[str, Any], filters: Dict[str, Any]) -> bool:
        """Check if document passes filter criteria"""
        metadata = document.get("metadata", {})
        
        for filter_key, filter_value in filters.items():
            if filter_value is None:
                continue
            
            if filter_key == "document_type":
                if metadata.get("document_type") != filter_value:
                    return False
            elif filter_key == "author":
                if metadata.get("author") != filter_value:
                    return False
            elif filter_key == "date_range":
                doc_date = metadata.get("date", "")
                if ":" in filter_value:
                    start_date, end_date = filter_value.split(":")
                    if not (start_date <= doc_date <= end_date):
                        return False
            elif filter_key in metadata:
                if metadata[filter_key] != filter_value:
                    return False
        
        return True

class TestDatabase:
    """In-memory test database for testing"""
    
    def __init__(self):
        self.connection = sqlite3.connect(":memory:")
        self.setup_tables()
    
    def setup_tables(self):
        """Set up test database tables"""
        cursor = self.connection.cursor()
        
        # Documents table
        cursor.execute("""
            CREATE TABLE documents (
                chunk_id TEXT PRIMARY KEY,
                content TEXT,
                metadata TEXT,
                embedding BLOB
            )
        """)
        
        # Queries table
        cursor.execute("""
            CREATE TABLE test_queries (
                query_id TEXT PRIMARY KEY,
                text TEXT,
                expected_results TEXT,
                filters TEXT
            )
        """)
        
        # Test results table
        cursor.execute("""
            CREATE TABLE test_results (
                test_id TEXT PRIMARY KEY,
                query_id TEXT,
                results TEXT,
                metrics TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.connection.commit()
    
    def insert_documents(self, documents: List[Dict[str, Any]], 
                        embeddings: Dict[str, np.ndarray] = None):
        """Insert test documents into database"""
        cursor = self.connection.cursor()
        
        for doc in documents:
            chunk_id = doc["chunk_id"]
            content = doc["content"]
            metadata = json.dumps(doc["metadata"])
            
            embedding_blob = None
            if embeddings and chunk_id in embeddings:
                embedding_blob = embeddings[chunk_id].tobytes()
            
            cursor.execute("""
                INSERT INTO documents (chunk_id, content, metadata, embedding)
                VALUES (?, ?, ?, ?)
            """, (chunk_id, content, metadata, embedding_blob))
        
        self.connection.commit()
    
    def insert_test_queries(self, queries: List[Dict[str, Any]]):
        """Insert test queries into database"""
        cursor = self.connection.cursor()
        
        for query in queries:
            cursor.execute("""
                INSERT INTO test_queries (query_id, text, expected_results, filters)
                VALUES (?, ?, ?, ?)
            """, (
                query["query_id"],
                query["text"],
                json.dumps(query.get("expected_results", [])),
                json.dumps(query.get("filters", {}))
            ))
        
        self.connection.commit()
    
    def save_test_results(self, test_id: str, query_id: str, 
                         results: List[Dict[str, Any]], metrics: Dict[str, Any]):
        """Save test results to database"""
        cursor = self.connection.cursor()
        
        cursor.execute("""
            INSERT INTO test_results (test_id, query_id, results, metrics)
            VALUES (?, ?, ?, ?)
        """, (test_id, query_id, json.dumps(results), json.dumps(metrics)))
        
        self.connection.commit()
    
    def close(self):
        """Close database connection"""
        self.connection.close()

# Test configuration instance
TEST_CONFIG = TestConfig()

# Utility functions
def setup_test_data():
    """Set up comprehensive test data"""
    generator = TestDataGenerator(TEST_CONFIG)
    
    # Generate test documents and queries
    documents = generator.generate_test_documents()
    queries = generator.generate_test_queries()
    embeddings = generator.generate_test_embeddings(documents)
    
    return documents, queries, embeddings

def create_test_services(documents: List[Dict[str, Any]] = None,
                        embeddings: Dict[str, np.ndarray] = None):
    """Create mock services for testing"""
    embedding_service = MockServiceFactory.create_mock_embedding_service(embeddings)
    vector_db_service = MockServiceFactory.create_mock_vector_database(documents, embeddings)
    
    return embedding_service, vector_db_service

def calculate_test_metrics(expected_results: List[str], 
                          actual_results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate test metrics for search results"""
    if not actual_results:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    actual_ids = {result.get("chunk_id", "") for result in actual_results}
    expected_ids = set(expected_results)
    
    # Calculate precision, recall, F1
    true_positives = len(actual_ids.intersection(expected_ids))
    precision = true_positives / len(actual_ids) if actual_ids else 0.0
    recall = true_positives / len(expected_ids) if expected_ids else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positives": true_positives,
        "total_results": len(actual_results),
        "expected_results": len(expected_results)
    }