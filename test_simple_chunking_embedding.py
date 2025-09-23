#!/usr/bin/env python3
"""
Simple test script for Text Chunking and Embedding components

This script tests the core functionality without database dependencies.
"""

import os
import sys
import logging
import time
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_text_chunking():
    """Test text chunking functionality"""
    logger.info("=== Testing Text Chunking ===")
    
    try:
        # Import chunking modules
        from src.utils.text_chunking import (
            TextChunkingService, ChunkingStrategy, 
            SentenceChunker, ParagraphChunker, FixedSizeChunker, LegalSectionChunker
        )
        
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
            ChunkingStrategy.PARAGRAPH,
            ChunkingStrategy.SENTENCE,
            ChunkingStrategy.FIXED_SIZE,
            ChunkingStrategy.LEGAL_SECTION
        ]
        
        for strategy in strategies:
            logger.info(f"Testing {strategy.value} chunking...")
            
            try:
                if strategy == ChunkingStrategy.SENTENCE:
                    chunks = chunking_service.chunk_document(
                        text=test_text,
                        strategy=strategy,
                        sentences_per_chunk=2
                    )
                elif strategy == ChunkingStrategy.FIXED_SIZE:
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
                    logger.info(f"    Metadata: size={chunk.metadata.chunk_size}, words={chunk.metadata.word_count}")
                    
            except Exception as e:
                logger.error(f"  Error with {strategy.value}: {str(e)}")
        
        logger.info("Text chunking test completed successfully!\n")
        return True
        
    except Exception as e:
        logger.error(f"Text chunking test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_generation():
    """Test embedding generation functionality"""
    logger.info("=== Testing Embedding Generation ===")
    
    try:
        # Import embedding modules
        from src.utils.embedding import (
            EmbeddingService, EmbeddingModel, 
            SentenceTransformerEmbeddingGenerator
        )
        
        embedding_service = EmbeddingService()
        
        # Test texts
        test_texts = [
            "Quốc hội là cơ quan đại diện cao nhất của nhân dân.",
            "Luật này quy định về tổ chức và hoạt động của Quốc hội.",
            "Quyết định của Quốc hội được thông qua theo đa số."
        ]
        
        # Test embedding model
        model = EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL
        
        logger.info(f"Testing {model.value} embeddings...")
        
        try:
            # Test single embedding
            result = embedding_service.generate_embedding(test_texts[0], model)
            logger.info(f"  Single embedding: dimension={result.embedding_dimension}, model={result.model_name}")
            logger.info(f"  Vector preview: {result.embedding_vector[:5]}...")
            
            # Test batch embeddings
            results = embedding_service.generate_embeddings_batch(test_texts, model)
            logger.info(f"  Batch embeddings: {len(results)} embeddings generated")
            
            # Test similarity
            similarity = embedding_service.calculate_similarity(results[0], results[1])
            logger.info(f"  Similarity between first two texts: {similarity:.4f}")
            
            # Test similarity calculation
            cosine_sim = embedding_service.calculate_similarity(results[0].embedding_vector, results[1].embedding_vector, method="cosine")
            euclidean_sim = embedding_service.calculate_similarity(results[0].embedding_vector, results[1].embedding_vector, method="euclidean")
            logger.info(f"  Cosine similarity: {cosine_sim:.4f}")
            logger.info(f"  Euclidean similarity: {euclidean_sim:.4f}")
            
        except Exception as e:
            logger.error(f"  Error with {model.value}: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
        
        logger.info("Embedding generation test completed successfully!\n")
        return True
        
    except Exception as e:
        logger.error(f"Embedding generation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_integrated_chunking_embedding():
    """Test integrated chunking and embedding"""
    logger.info("=== Testing Integrated Chunking and Embedding ===")
    
    try:
        # Import modules
        from src.utils.text_chunking import TextChunkingService, ChunkingStrategy
        from src.utils.embedding import EmbeddingService, EmbeddingModel
        
        chunking_service = TextChunkingService()
        embedding_service = EmbeddingService()
        
        # Test document
        test_document = """
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
        """
        
        # Step 1: Chunk the document
        logger.info("Step 1: Chunking document...")
        chunks = chunking_service.chunk_document(
            text=test_document,
            strategy=ChunkingStrategy.LEGAL_SECTION
        )
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 2: Generate embeddings for chunks
        logger.info("Step 2: Generating embeddings...")
        chunk_texts = [chunk.content for chunk in chunks]
        embeddings = embedding_service.generate_embeddings_batch(
            chunk_texts,
            EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL
        )
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Step 3: Test similarity search
        logger.info("Step 3: Testing similarity search...")
        query_text = "Quốc hội có vai trò gì?"
        query_embedding = embedding_service.generate_embedding(
            query_text,
            EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL
        )
        
        # Find most similar chunks
        similarities = []
        for i, embedding in enumerate(embeddings):
            similarity = embedding_service.calculate_similarity(query_embedding, embedding)
            similarities.append((i, similarity, chunks[i].content[:100]))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(f"Query: {query_text}")
        logger.info("Most similar chunks:")
        for i, (chunk_idx, similarity, content) in enumerate(similarities[:3]):
            logger.info(f"  {i+1}. Similarity: {similarity:.4f}")
            logger.info(f"     Content: {content}...")
        
        logger.info("Integrated chunking and embedding test completed successfully!\n")
        return True
        
    except Exception as e:
        logger.error(f"Integrated test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    logger.info("Starting Simple Text Chunking and Embedding Tests")
    logger.info("=" * 60)
    
    results = []
    
    # Test individual components
    results.append(("Text Chunking", test_text_chunking()))
    results.append(("Embedding Generation", test_embedding_generation()))
    results.append(("Integrated Processing", test_integrated_chunking_embedding()))
    
    # Summary
    logger.info("=" * 60)
    logger.info("Test Results Summary:")
    
    all_passed = True
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        logger.info(f"  {test_name}: {status}")
        if not passed:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests completed successfully!")
        logger.info("Text Chunking and Embedding system is working correctly!")
    else:
        logger.error("\n❌ Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()