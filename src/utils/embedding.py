"""
Embedding Generation Module for Text Chunks

This module provides functionality to generate embeddings for text chunks
using various embedding models, with focus on Vietnamese and multilingual support.
"""

import logging
import time
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json

# Sentence Transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logging.warning("sentence-transformers not available. Install with: pip install sentence-transformers")

# OpenAI for embeddings (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

logger = logging.getLogger(__name__)


class EmbeddingModel(Enum):
    """Available embedding models"""
    SENTENCE_TRANSFORMERS_MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    SENTENCE_TRANSFORMERS_VIETNAMESE = "sentence-transformers/distiluse-base-multilingual-cased"
    OPENAI_ADA_002 = "text-embedding-ada-002"
    CUSTOM_VIETNAMESE = "custom-vietnamese-legal"


@dataclass
class EmbeddingResult:
    """Result of embedding generation"""
    embedding_vector: List[float]
    embedding_dimension: int
    model_name: str
    model_version: str
    processing_time: float
    quality_score: float = 0.0
    normalization_applied: bool = True


class BaseEmbeddingGenerator:
    """Base class for embedding generators"""
    
    def __init__(self, model_name: str, normalize: bool = True):
        self.model_name = model_name
        self.normalize = normalize
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the embedding model"""
        raise NotImplementedError("Subclasses must implement _load_model method")
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            
        Returns:
            EmbeddingResult object
        """
        raise NotImplementedError("Subclasses must implement generate_embedding method")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts to embed
            
        Returns:
            List of EmbeddingResult objects
        """
        results = []
        for text in texts:
            result = self.generate_embedding(text)
            results.append(result)
        return results
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize embedding vector to unit length"""
        if self.normalize:
            norm = np.linalg.norm(vector)
            if norm > 0:
                return vector / norm
        return vector
    
    def _calculate_quality_score(self, text: str, embedding: np.ndarray) -> float:
        """
        Calculate quality score for the embedding
        Simple implementation based on text length and embedding variance
        """
        if len(text.strip()) == 0:
            return 0.0
        
        # Check embedding variance (higher variance usually indicates better representation)
        variance = np.var(embedding)
        
        # Normalize based on text length (longer texts might have better embeddings)
        text_length_factor = min(1.0, len(text) / 100)
        
        # Simple quality score combining variance and text length
        quality = min(1.0, variance * 10 + text_length_factor * 0.3)
        return max(0.0, quality)


class SentenceTransformerEmbeddingGenerator(BaseEmbeddingGenerator):
    """Embedding generator using Sentence Transformers"""
    
    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2", **kwargs):
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required but not installed")
        
        self.full_model_name = model_name
        super().__init__(model_name, **kwargs)
    
    def _load_model(self):
        """Load Sentence Transformer model"""
        try:
            self.model = SentenceTransformer(self.full_model_name)
            logger.info(f"Loaded Sentence Transformer model: {self.full_model_name}")
        except Exception as e:
            logger.error(f"Failed to load Sentence Transformer model {self.full_model_name}: {str(e)}")
            raise
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding using Sentence Transformers"""
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        start_time = time.time()
        
        try:
            # Generate embedding
            embedding = self.model.encode(text, convert_to_numpy=True)
            
            # Normalize if requested
            if self.normalize:
                embedding = self._normalize_vector(embedding)
            
            processing_time = time.time() - start_time
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(text, embedding)
            
            return EmbeddingResult(
                embedding_vector=embedding.tolist(),
                embedding_dimension=len(embedding),
                model_name=self.model_name,
                model_version=self.full_model_name,
                processing_time=processing_time,
                quality_score=quality_score,
                normalization_applied=self.normalize
            )
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for batch of texts (more efficient)"""
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [(i, text) for i, text in enumerate(texts) if text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")
        
        indices, valid_text_list = zip(*valid_texts)
        
        start_time = time.time()
        
        try:
            # Generate embeddings in batch
            embeddings = self.model.encode(valid_text_list, convert_to_numpy=True)
            
            # Normalize if requested
            if self.normalize:
                embeddings = np.array([self._normalize_vector(emb) for emb in embeddings])
            
            processing_time = time.time() - start_time
            avg_processing_time = processing_time / len(valid_text_list)
            
            # Create results
            results = []
            for i, (original_idx, text) in enumerate(valid_texts):
                embedding = embeddings[i]
                quality_score = self._calculate_quality_score(text, embedding)
                
                result = EmbeddingResult(
                    embedding_vector=embedding.tolist(),
                    embedding_dimension=len(embedding),
                    model_name=self.model_name,
                    model_version=self.full_model_name,
                    processing_time=avg_processing_time,
                    quality_score=quality_score,
                    normalization_applied=self.normalize
                )
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Error generating batch embeddings: {str(e)}")
            raise


class OpenAIEmbeddingGenerator(BaseEmbeddingGenerator):
    """Embedding generator using OpenAI API"""
    
    def __init__(self, model_name: str = "text-embedding-ada-002", api_key: str = None, **kwargs):
        if not OPENAI_AVAILABLE:
            raise ImportError("openai is required but not installed")
        
        self.api_key = api_key
        if api_key:
            openai.api_key = api_key
        
        super().__init__(model_name, **kwargs)
    
    def _load_model(self):
        """OpenAI models don't need explicit loading"""
        logger.info(f"Using OpenAI embedding model: {self.model_name}")
    
    def generate_embedding(self, text: str) -> EmbeddingResult:
        """Generate embedding using OpenAI API"""
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        
        start_time = time.time()
        
        try:
            response = openai.Embedding.create(
                input=text,
                model=self.model_name
            )
            
            embedding = np.array(response['data'][0]['embedding'])
            
            # Normalize if requested
            if self.normalize:
                embedding = self._normalize_vector(embedding)
            
            processing_time = time.time() - start_time
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(text, embedding)
            
            return EmbeddingResult(
                embedding_vector=embedding.tolist(),
                embedding_dimension=len(embedding),
                model_name=self.model_name,
                model_version=self.model_name,
                processing_time=processing_time,
                quality_score=quality_score,
                normalization_applied=self.normalize
            )
            
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            raise


class EmbeddingService:
    """Service for managing embedding generation operations"""
    
    def __init__(self):
        self.generators = {}
        self._initialize_default_generators()
    
    def _initialize_default_generators(self):
        """Initialize default embedding generators"""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Multilingual model good for Vietnamese
                self.generators[EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL] = \
                    SentenceTransformerEmbeddingGenerator("paraphrase-multilingual-MiniLM-L12-v2")
                
                # Another good multilingual model
                self.generators[EmbeddingModel.SENTENCE_TRANSFORMERS_VIETNAMESE] = \
                    SentenceTransformerEmbeddingGenerator("distiluse-base-multilingual-cased")
                
                logger.info("Initialized Sentence Transformer generators")
            except Exception as e:
                logger.warning(f"Failed to initialize Sentence Transformer generators: {str(e)}")
    
    def add_generator(self, model: EmbeddingModel, generator: BaseEmbeddingGenerator):
        """Add a custom embedding generator"""
        self.generators[model] = generator
        logger.info(f"Added custom generator for {model.value}")
    
    def generate_embedding(
        self,
        text: str,
        model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL
    ) -> EmbeddingResult:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text to embed
            model: Embedding model to use
            
        Returns:
            EmbeddingResult object
        """
        if model not in self.generators:
            raise ValueError(f"Generator for model {model.value} not available")
        
        generator = self.generators[model]
        return generator.generate_embedding(text)
    
    def generate_embeddings_batch(
        self,
        texts: List[str],
        model: EmbeddingModel = EmbeddingModel.SENTENCE_TRANSFORMERS_MULTILINGUAL
    ) -> List[EmbeddingResult]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of input texts to embed
            model: Embedding model to use
            
        Returns:
            List of EmbeddingResult objects
        """
        if model not in self.generators:
            raise ValueError(f"Generator for model {model.value} not available")
        
        generator = self.generators[model]
        return generator.generate_embeddings_batch(texts)
    
    def get_available_models(self) -> List[EmbeddingModel]:
        """Get list of available embedding models"""
        return list(self.generators.keys())
    
    def calculate_similarity(
        self,
        embedding1: Union[EmbeddingResult, List[float]],
        embedding2: Union[EmbeddingResult, List[float]],
        method: str = "cosine"
    ) -> float:
        """
        Calculate similarity between two embeddings
        
        Args:
            embedding1: First embedding (EmbeddingResult or vector)
            embedding2: Second embedding (EmbeddingResult or vector)
            method: Similarity method ("cosine", "euclidean", "dot")
            
        Returns:
            Similarity score
        """
        # Extract vectors from EmbeddingResult if needed
        vec1 = embedding1.embedding_vector if isinstance(embedding1, EmbeddingResult) else embedding1
        vec2 = embedding2.embedding_vector if isinstance(embedding2, EmbeddingResult) else embedding2
        
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        if method == "cosine":
            # Cosine similarity
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return dot_product / (norm1 * norm2)
        
        elif method == "euclidean":
            # Euclidean distance (converted to similarity)
            distance = np.linalg.norm(vec1 - vec2)
            return 1.0 / (1.0 + distance)
        
        elif method == "dot":
            # Dot product
            return np.dot(vec1, vec2)
        
        else:
            raise ValueError(f"Unsupported similarity method: {method}")
    
    def find_similar_embeddings(
        self,
        query_embedding: List[float],
        candidate_embeddings: List[Tuple[int, List[float]]],
        top_k: int = 10,
        similarity_threshold: float = 0.5
    ) -> List[Tuple[int, float]]:
        """
        Find most similar embeddings to a query embedding
        
        Args:
            query_embedding: Query embedding vector
            candidate_embeddings: List of (id, embedding) tuples
            top_k: Number of top results to return
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of (id, similarity_score) tuples, sorted by similarity
        """
        similarities = []
        
        for candidate_id, candidate_embedding in candidate_embeddings:
            similarity = self.calculate_similarity(query_embedding, candidate_embedding)
            
            if similarity >= similarity_threshold:
                similarities.append((candidate_id, similarity))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


# Utility functions for embedding operations
def save_embeddings_to_json(embeddings: List[EmbeddingResult], filepath: str):
    """Save embeddings to JSON file"""
    data = []
    for embedding in embeddings:
        data.append({
            'embedding_vector': embedding.embedding_vector,
            'embedding_dimension': embedding.embedding_dimension,
            'model_name': embedding.model_name,
            'model_version': embedding.model_version,
            'processing_time': embedding.processing_time,
            'quality_score': embedding.quality_score,
            'normalization_applied': embedding.normalization_applied
        })
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_embeddings_from_json(filepath: str) -> List[EmbeddingResult]:
    """Load embeddings from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    embeddings = []
    for item in data:
        embedding = EmbeddingResult(
            embedding_vector=item['embedding_vector'],
            embedding_dimension=item['embedding_dimension'],
            model_name=item['model_name'],
            model_version=item['model_version'],
            processing_time=item['processing_time'],
            quality_score=item.get('quality_score', 0.0),
            normalization_applied=item.get('normalization_applied', True)
        )
        embeddings.append(embedding)
    
    return embeddings