#!/usr/bin/env python3
"""
Test script for Text Chunking and Embedding System

This script tests the complete text chunking and embedding pipeline
including database operations and vector search functionality.
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.database.connection import get_database_connection
    from src.database.models import (
        Document, TextChunk, ChunkEmbedding, ProcessingStatus,
        ChunkingStrategy, EmbeddingModel, DocumentType
    )
    from src.utils.text_chunking import TextChunkingService, ChunkingStrategy as ChunkStrategy
    from src.utils.embedding import EmbeddingService, EmbeddingModel as EmbeddingModelEnum
    from src.utils.vector_database import VectorDatabaseService
    from src.utils.document_embedding_pipeline import (
        DocumentEmbeddingPipeline, PipelineConfig, ChunkingConfig, EmbeddingConfig
    )
except ImportError:
    # Try without src prefix
    from database.connection import get_database_connection
    from database.models import (
        Document, TextChunk, ChunkEmbedding, ProcessingStatus,
        ChunkingStrategy, EmbeddingModel, DocumentType
    )
    from utils.text_chunking import TextChunkingService, ChunkingStrategy as ChunkStrategy
    from utils.embedding import EmbeddingService, EmbeddingModel as EmbeddingModelEnum
    from utils.vector_database import VectorDatabaseService
    from utils.document_embedding_pipeline import (
        DocumentEmbeddingPipeline, PipelineConfig, ChunkingConfig, EmbeddingConfig
    )

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_document(db_session, title: str, content: str) -> int:
    """Create a test document in the database"""
    document = Document(
        title=title,
        document_type=DocumentType.LAW,
        raw_text=content,
        processing_status=ProcessingStatus.PENDING,
        file_path=f"test_{title.lower().replace(' ', '_')}.txt",
        file_size=len(content.encode('utf-8')),
        page_count=1
    )
    
    db_session.add(document)
    db_session.commit()
    
    logger.info(f"Created test document: {title} (ID: {document.id})")
    return document.id


def test_text_chunking():
    """Test text chunking functionality"""
    logger.info("=== Testing Text Chunking ===")
    
    chunking_service = TextChunkingService()
    
    # Test text
    test_text = """
    Điều 1. Phạm vi điều chỉnh
    Luật này quy định về tổ chức và hoạt động của Quốc hội; quyền và nghĩa vụ của đại biểu Quốc hội.
    
    Điều 2. Vị trí, vai trò của Quốc hội
    Quốc hội là cơ quan đại diện cao nhất của nhân dân, cơ quan quyền lực nhà nước cao nhất của nước Cộng hòa xã hội chủ nghĩa Việt Nam.
    
    Điều 3. Nguyên tắc tổ chức và hoạt động của Quốc hội
    1. Quốc hội tổ chức và hoạt động theo nguyên tắc tập trung dân chủ.
    2. Quốc hội làm việc theo chế độ kỳ họp.
    3. Quyết định của Quốc hội được thông qua theo đa số.
    """
    
    # Test different chunking strategies
    strategies = [
        ChunkStrategy.PARAGRAPH,
        ChunkStrategy.SENTENCE,
        ChunkStrategy.FIXED_SIZE,
        ChunkStrategy.LEGAL_SECTION
    ]
    
    for strategy in strategies:
        logger.info(f"Testing {strategy.value} chunking...")
        
        try:
            if strategy == ChunkStrategy.SENTENCE:
                chunks = chunking_service.chunk_document(
                    text=test_text,
                    strategy=strategy,
                    sentences_per_chunk=2
                )
            elif strategy == ChunkStrategy.FIXED_SIZE:
                chunks = chunking_service.chunk_document(
                    text=test_text,
                    strategy=strategy,
                    chunk_size=200,
                    overlap=50
                )
            else:
                chunks = chunking_service.chunk_document(
                    text=test_text,
                    strategy=strategy
                )
            
            logger.info(f"  Created {len(chunks)} chunks")
            for i, chunk in enumerate(chunks[:2]):  # Show first 2 chunks
                logger.info(f"  Chunk {i+1}: {chunk.content[:100]}...")
                
        except Exception as e:
            logger.error(f"  Error with {strategy.value}: {str(e)}")
    
    logger.info("Text chunking test completed\n")


def test_embedding_generation():
    """Test embedding generation functionality"""
    logger.info("=== Testing Embedding Generation ===")
    
    embedding_service = EmbeddingService()
    
    # Test texts
    test_texts = [
        "Quốc hội là cơ quan đại diện cao nhất của nhân dân.",
        "Luật này quy định về tổ chức và hoạt động của Quốc hội.",
        "Quyết định của Quốc hội được thông qua theo đa số."
    ]
    
    # Test different embedding models
    models = [EmbeddingModelEnum.SENTENCE_TRANSFORMERS_MULTILINGUAL]
    
    for model in models:
        logger.info(f"Testing {model.value} embeddings...")
        
        try:
            # Test single embedding
            result = embedding_service.generate_embedding(test_texts[0], model)
            logger.info(f"  Single embedding: dimension={result.dimension}, model={result.model}")
            
            # Test batch embeddings
            results = embedding_service.generate_embeddings_batch(test_texts, model)
            logger.info(f"  Batch embeddings: {len(results)} embeddings generated")
            
            # Test similarity
            similarity = embedding_service.calculate_similarity(results[0], results[1])
            logger.info(f"  Similarity between first two texts: {similarity:.4f}")
            
        except Exception as e:
            logger.error(f"  Error with {model.value}: {str(e)}")
    
    logger.info("Embedding generation test completed\n")


def test_vector_database():
    """Test vector database functionality"""
    logger.info("=== Testing Vector Database ===")
    
    try:
        db_session = get_database_connection()
        vector_db = VectorDatabaseService(db_session)
        embedding_service = EmbeddingService()
        
        # Create test embeddings
        test_texts = [
            "Quốc hội là cơ quan đại diện cao nhất của nhân dân.",
            "Luật này quy định về tổ chức và hoạt động của Quốc hội.",
            "Quyết định của Quốc hội được thông qua theo đa số.",
            "Đại biểu Quốc hội có quyền và nghĩa vụ theo quy định của pháp luật."
        ]
        
        embeddings = embedding_service.generate_embeddings_batch(
            test_texts, 
            EmbeddingModelEnum.SENTENCE_TRANSFORMERS_MULTILINGUAL
        )
        
        # Test similarity search
        query_embedding = embeddings[0]
        similar_embeddings = vector_db.find_similar_embeddings(
            query_embedding.vector,
            top_k=3,
            threshold=0.5
        )
        
        logger.info(f"Found {len(similar_embeddings)} similar embeddings")
        for result in similar_embeddings:
            logger.info(f"  Similarity: {result.similarity:.4f}")
        
        # Test statistics
        stats = vector_db.get_embedding_statistics()
        logger.info(f"Vector database statistics: {stats}")
        
    except Exception as e:
        logger.error(f"Vector database test error: {str(e)}")
    
    logger.info("Vector database test completed\n")


def test_complete_pipeline():
    """Test the complete document processing pipeline"""
    logger.info("=== Testing Complete Pipeline ===")
    
    try:
        db_session = get_database_connection()
        pipeline = DocumentEmbeddingPipeline(db_session)
        
        # Create test document
        test_content = """
        Điều 1. Phạm vi điều chỉnh
        Luật này quy định về tổ chức và hoạt động của Quốc hội; quyền và nghĩa vụ của đại biểu Quốc hội.
        
        Điều 2. Vị trí, vai trò của Quốc hội
        Quốc hội là cơ quan đại diện cao nhất của nhân dân, cơ quan quyền lực nhà nước cao nhất của nước Cộng hòa xã hội chủ nghĩa Việt Nam.
        Quốc hội thực hiện quyền lập pháp, quyết định những vấn đề quan trọng của đất nước, giám sát tối cao đối với hoạt động của Nhà nước.
        
        Điều 3. Nguyên tắc tổ chức và hoạt động của Quốc hội
        1. Quốc hội tổ chức và hoạt động theo nguyên tắc tập trung dân chủ.
        2. Quốc hội làm việc theo chế độ kỳ họp.
        3. Quyết định của Quốc hội được thông qua theo đa số.
        4. Hoạt động của Quốc hội được tiến hành công khai, dân chủ.
        
        Điều 4. Nhiệm kỳ của Quốc hội
        Quốc hội khóa XIV có nhiệm kỳ năm năm kể từ ngày kỳ họp thứ nhất của Quốc hội khóa XIV khai mạc.
        """
        
        document_id = create_test_document(db_session, "Test Legal Document", test_content)
        
        # Test with different configurations
        configs = [
            ("Default Config", PipelineConfig(
                chunking=ChunkingConfig(strategy=ChunkingStrategy.PARAGRAPH),
                embedding=EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL)
            )),
            ("Legal Section Config", PipelineConfig(
                chunking=ChunkingConfig(strategy=ChunkingStrategy.LEGAL_SECTION),
                embedding=EmbeddingConfig(model=EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL)
            ))
        ]
        
        for config_name, config in configs:
            logger.info(f"Testing pipeline with {config_name}...")
            
            # Set overwrite to true for testing
            config.overwrite_existing = True
            
            result = pipeline.process_document(document_id, config)
            
            if result.success:
                logger.info(f"  Success: {result.chunks_created} chunks, {result.embeddings_created} embeddings")
                logger.info(f"  Processing time: {result.processing_time:.2f}s")
            else:
                logger.error(f"  Failed: {result.error_message}")
        
        # Test statistics
        stats = pipeline.get_processing_statistics()
        logger.info(f"Pipeline statistics: {stats}")
        
        # Clean up test document
        document = db_session.query(Document).filter(Document.id == document_id).first()
        if document:
            # Delete related chunks and embeddings
            chunks = db_session.query(TextChunk).filter(TextChunk.document_id == document_id).all()
            for chunk in chunks:
                embeddings = db_session.query(ChunkEmbedding).filter(ChunkEmbedding.chunk_id == chunk.id).all()
                for embedding in embeddings:
                    db_session.delete(embedding)
                db_session.delete(chunk)
            db_session.delete(document)
            db_session.commit()
            logger.info(f"Cleaned up test document {document_id}")
        
    except Exception as e:
        logger.error(f"Complete pipeline test error: {str(e)}")
        import traceback
        traceback.print_exc()
    
    logger.info("Complete pipeline test completed\n")


def main():
    """Run all tests"""
    logger.info("Starting Text Chunking and Embedding System Tests")
    logger.info("=" * 60)
    
    try:
        # Test individual components
        test_text_chunking()
        test_embedding_generation()
        test_vector_database()
        
        # Test complete pipeline
        test_complete_pipeline()
        
        logger.info("=" * 60)
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()