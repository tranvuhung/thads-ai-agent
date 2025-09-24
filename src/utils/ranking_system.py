"""
Advanced Ranking System for Semantic Search Results

This module provides sophisticated ranking and scoring algorithms
for ordering search results based on multiple relevance factors.
"""

import logging
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import re

logger = logging.getLogger(__name__)


class RankingAlgorithm(Enum):
    """Available ranking algorithms"""
    TF_IDF = "tf_idf"
    BM25 = "bm25"
    COSINE_SIMILARITY = "cosine_similarity"
    LEARNING_TO_RANK = "learning_to_rank"
    PAGERANK = "pagerank"
    HYBRID_SCORE = "hybrid_score"
    NEURAL_RANKING = "neural_ranking"


class RelevanceFeature(Enum):
    """Features used for relevance scoring"""
    SEMANTIC_SIMILARITY = "semantic_similarity"
    LEXICAL_OVERLAP = "lexical_overlap"
    DOCUMENT_FREQUENCY = "document_frequency"
    TERM_FREQUENCY = "term_frequency"
    DOCUMENT_LENGTH = "document_length"
    DOCUMENT_AUTHORITY = "document_authority"
    TEMPORAL_RELEVANCE = "temporal_relevance"
    USER_ENGAGEMENT = "user_engagement"
    CONTENT_QUALITY = "content_quality"
    ENTITY_MATCHING = "entity_matching"


@dataclass
class RankingFeatures:
    """Features extracted for ranking"""
    semantic_similarity: float = 0.0
    lexical_overlap: float = 0.0
    document_frequency: float = 0.0
    term_frequency: float = 0.0
    document_length: float = 0.0
    document_authority: float = 0.0
    temporal_relevance: float = 0.0
    user_engagement: float = 0.0
    content_quality: float = 0.0
    entity_matching: float = 0.0
    custom_features: Dict[str, float] = field(default_factory=dict)


@dataclass
class RankingConfig:
    """Configuration for ranking system"""
    algorithm: RankingAlgorithm = RankingAlgorithm.HYBRID_SCORE
    feature_weights: Dict[RelevanceFeature, float] = field(default_factory=dict)
    normalize_scores: bool = True
    apply_diversity: bool = True
    diversity_lambda: float = 0.5
    temporal_decay: float = 0.1
    authority_boost: float = 0.2
    quality_threshold: float = 0.3


@dataclass
class RankedResult:
    """Result with ranking information"""
    original_result: Any
    ranking_score: float
    ranking_features: RankingFeatures
    rank_position: int
    explanation: Dict[str, Any] = field(default_factory=dict)


class AdvancedRankingSystem:
    """
    Advanced ranking system with multiple algorithms and features
    """
    
    def __init__(self, config: RankingConfig = None):
        """
        Initialize the ranking system
        
        Args:
            config: Ranking configuration
        """
        self.config = config or RankingConfig()
        
        # Default feature weights
        if not self.config.feature_weights:
            self.config.feature_weights = {
                RelevanceFeature.SEMANTIC_SIMILARITY: 0.4,
                RelevanceFeature.LEXICAL_OVERLAP: 0.2,
                RelevanceFeature.DOCUMENT_AUTHORITY: 0.15,
                RelevanceFeature.TEMPORAL_RELEVANCE: 0.1,
                RelevanceFeature.CONTENT_QUALITY: 0.1,
                RelevanceFeature.USER_ENGAGEMENT: 0.05
            }
        
        # Document statistics for TF-IDF and BM25
        self.document_stats = {}
        self.corpus_stats = {}
        self.is_fitted = False
        
        logger.info(f"AdvancedRankingSystem initialized with {self.config.algorithm.value}")
    
    def fit(self, documents: List[Dict[str, Any]], corpus_metadata: Dict[str, Any] = None):
        """
        Fit the ranking system with document corpus
        
        Args:
            documents: List of documents with content and metadata
            corpus_metadata: Global corpus statistics
        """
        self.documents = documents
        self.corpus_stats = corpus_metadata or {}
        
        # Calculate document statistics
        self._calculate_document_stats()
        
        # Calculate corpus-level statistics
        self._calculate_corpus_stats()
        
        self.is_fitted = True
        logger.info(f"Ranking system fitted with {len(documents)} documents")
    
    def rank_results(self, 
                    results: List[Any], 
                    query: str,
                    query_embedding: np.ndarray = None,
                    user_context: Dict[str, Any] = None) -> List[RankedResult]:
        """
        Rank search results using the configured algorithm
        
        Args:
            results: Search results to rank
            query: Original search query
            query_embedding: Query embedding vector
            user_context: User context for personalization
            
        Returns:
            List of ranked results
        """
        if not results:
            return []
        
        # Extract features for each result
        ranked_results = []
        for result in results:
            features = self._extract_features(result, query, query_embedding, user_context)
            
            # Calculate ranking score
            ranking_score = self._calculate_ranking_score(features, result, query)
            
            ranked_result = RankedResult(
                original_result=result,
                ranking_score=ranking_score,
                ranking_features=features,
                rank_position=0,  # Will be set after sorting
                explanation=self._generate_explanation(features, ranking_score)
            )
            
            ranked_results.append(ranked_result)
        
        # Sort by ranking score
        ranked_results.sort(key=lambda x: x.ranking_score, reverse=True)
        
        # Apply diversity if enabled
        if self.config.apply_diversity:
            ranked_results = self._apply_diversity_reranking(ranked_results)
        
        # Set final rank positions
        for i, result in enumerate(ranked_results):
            result.rank_position = i + 1
        
        # Normalize scores if enabled
        if self.config.normalize_scores:
            self._normalize_scores(ranked_results)
        
        return ranked_results
    
    def _extract_features(self, 
                         result: Any, 
                         query: str,
                         query_embedding: np.ndarray = None,
                         user_context: Dict[str, Any] = None) -> RankingFeatures:
        """Extract ranking features from a result"""
        features = RankingFeatures()
        
        # Get result content and metadata
        content = getattr(result, 'content', '')
        metadata = getattr(result, 'chunk_metadata', {}) or getattr(result, 'metadata', {})
        
        # Semantic similarity (if available)
        if hasattr(result, 'similarity_score'):
            features.semantic_similarity = float(result.similarity_score)
        
        # Lexical overlap
        features.lexical_overlap = self._calculate_lexical_overlap(query, content)
        
        # Document frequency features
        features.document_frequency = self._calculate_document_frequency(content)
        features.term_frequency = self._calculate_term_frequency(query, content)
        
        # Document length normalization
        features.document_length = self._calculate_length_score(content)
        
        # Document authority
        features.document_authority = self._calculate_authority_score(metadata)
        
        # Temporal relevance
        features.temporal_relevance = self._calculate_temporal_relevance(metadata)
        
        # User engagement (if available)
        features.user_engagement = self._calculate_user_engagement(metadata, user_context)
        
        # Content quality
        features.content_quality = self._calculate_content_quality(content, metadata)
        
        # Entity matching
        features.entity_matching = self._calculate_entity_matching(query, content)
        
        return features
    
    def _calculate_lexical_overlap(self, query: str, content: str) -> float:
        """Calculate lexical overlap between query and content"""
        query_terms = set(self._tokenize(query.lower()))
        content_terms = set(self._tokenize(content.lower()))
        
        if not query_terms:
            return 0.0
        
        overlap = len(query_terms.intersection(content_terms))
        return overlap / len(query_terms)
    
    def _calculate_document_frequency(self, content: str) -> float:
        """Calculate document frequency score"""
        if not self.is_fitted:
            return 0.5
        
        terms = self._tokenize(content.lower())
        if not terms:
            return 0.0
        
        # Average IDF of terms in the document
        total_idf = 0.0
        for term in set(terms):
            df = self.corpus_stats.get('term_doc_freq', {}).get(term, 1)
            total_docs = self.corpus_stats.get('total_documents', 1)
            idf = math.log(total_docs / df)
            total_idf += idf
        
        return total_idf / len(set(terms)) if terms else 0.0
    
    def _calculate_term_frequency(self, query: str, content: str) -> float:
        """Calculate term frequency score"""
        query_terms = self._tokenize(query.lower())
        content_terms = self._tokenize(content.lower())
        
        if not query_terms or not content_terms:
            return 0.0
        
        content_term_count = Counter(content_terms)
        total_content_terms = len(content_terms)
        
        tf_score = 0.0
        for term in query_terms:
            tf = content_term_count.get(term, 0) / total_content_terms
            tf_score += tf
        
        return tf_score / len(query_terms)
    
    def _calculate_length_score(self, content: str) -> float:
        """Calculate document length normalization score"""
        length = len(content)
        
        # Optimal length range (adjust based on domain)
        optimal_min = 100
        optimal_max = 1000
        
        if optimal_min <= length <= optimal_max:
            return 1.0
        elif length < optimal_min:
            return length / optimal_min
        else:
            # Penalize very long documents
            return max(0.3, optimal_max / length)
    
    def _calculate_authority_score(self, metadata: Dict[str, Any]) -> float:
        """Calculate document authority score"""
        authority_score = 0.5  # Default neutral score
        
        # Document type authority
        doc_type = metadata.get('document_type', '').lower()
        if 'law' in doc_type or 'regulation' in doc_type:
            authority_score += 0.3
        elif 'official' in doc_type or 'government' in doc_type:
            authority_score += 0.2
        
        # Source authority
        source = metadata.get('source', '').lower()
        if 'official' in source or 'government' in source:
            authority_score += 0.2
        
        # Citation count (if available)
        citations = metadata.get('citation_count', 0)
        if citations > 0:
            authority_score += min(0.3, math.log(citations + 1) / 10)
        
        return min(1.0, authority_score)
    
    def _calculate_temporal_relevance(self, metadata: Dict[str, Any]) -> float:
        """Calculate temporal relevance score"""
        created_date = metadata.get('created_date') or metadata.get('date')
        if not created_date:
            return 0.5  # Neutral score for unknown dates
        
        try:
            if isinstance(created_date, str):
                date_obj = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
            else:
                date_obj = created_date
            
            days_old = (datetime.now() - date_obj).days
            
            # Exponential decay with configurable rate
            decay_rate = self.config.temporal_decay
            relevance = math.exp(-decay_rate * days_old / 365)  # Decay over years
            
            return max(0.1, relevance)  # Minimum relevance
            
        except Exception:
            return 0.5
    
    def _calculate_user_engagement(self, metadata: Dict[str, Any], user_context: Dict[str, Any] = None) -> float:
        """Calculate user engagement score"""
        if not user_context:
            return 0.5
        
        engagement_score = 0.5
        
        # View count
        views = metadata.get('view_count', 0)
        if views > 0:
            engagement_score += min(0.3, math.log(views + 1) / 20)
        
        # User interaction history
        user_history = user_context.get('interaction_history', [])
        doc_id = metadata.get('document_id')
        if doc_id in user_history:
            engagement_score += 0.2
        
        return min(1.0, engagement_score)
    
    def _calculate_content_quality(self, content: str, metadata: Dict[str, Any]) -> float:
        """Calculate content quality score"""
        quality_score = 0.5
        
        # Length quality (not too short, not too long)
        length = len(content)
        if 50 <= length <= 2000:
            quality_score += 0.2
        
        # Language quality (basic checks)
        sentences = content.split('.')
        if len(sentences) > 1:  # Multiple sentences
            quality_score += 0.1
        
        # Formatting quality
        if any(char in content for char in ['\n', '\t']):  # Has structure
            quality_score += 0.1
        
        # Metadata completeness
        if metadata.get('title') and metadata.get('author'):
            quality_score += 0.1
        
        return min(1.0, quality_score)
    
    def _calculate_entity_matching(self, query: str, content: str) -> float:
        """Calculate entity matching score"""
        # Simple entity matching (can be enhanced with NER)
        query_entities = self._extract_simple_entities(query)
        content_entities = self._extract_simple_entities(content)
        
        if not query_entities:
            return 0.0
        
        matches = len(query_entities.intersection(content_entities))
        return matches / len(query_entities)
    
    def _extract_simple_entities(self, text: str) -> set:
        """Extract simple entities (capitalized words, numbers, dates)"""
        entities = set()
        
        # Capitalized words (potential proper nouns)
        capitalized = re.findall(r'\b[A-ZÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼẾỀỂỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪỬỮỰỲỴỶỸ][a-zàáâãèéêìíòóôõùúăđĩũơưăạảấầẩẫậắằẳẵặẹẻẽếềểễệỉịọỏốồổỗộớờởỡợụủứừửữựỳỵỷỹ]+\b', text)
        entities.update(capitalized)
        
        # Numbers
        numbers = re.findall(r'\b\d+\b', text)
        entities.update(numbers)
        
        # Dates (simple patterns)
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        entities.update(dates)
        
        return entities
    
    def _calculate_ranking_score(self, features: RankingFeatures, result: Any, query: str) -> float:
        """Calculate final ranking score based on algorithm"""
        if self.config.algorithm == RankingAlgorithm.HYBRID_SCORE:
            return self._calculate_hybrid_score(features)
        elif self.config.algorithm == RankingAlgorithm.BM25:
            return self._calculate_bm25_score(query, getattr(result, 'content', ''))
        elif self.config.algorithm == RankingAlgorithm.TF_IDF:
            return self._calculate_tfidf_score(query, getattr(result, 'content', ''))
        elif self.config.algorithm == RankingAlgorithm.COSINE_SIMILARITY:
            return features.semantic_similarity
        else:
            # Default to hybrid
            return self._calculate_hybrid_score(features)
    
    def _calculate_hybrid_score(self, features: RankingFeatures) -> float:
        """Calculate hybrid ranking score using weighted features"""
        score = 0.0
        
        for feature_type, weight in self.config.feature_weights.items():
            feature_value = getattr(features, feature_type.value, 0.0)
            score += feature_value * weight
        
        # Add custom features
        for custom_feature, value in features.custom_features.items():
            custom_weight = self.config.feature_weights.get(custom_feature, 0.1)
            score += value * custom_weight
        
        return min(1.0, score)
    
    def _calculate_bm25_score(self, query: str, content: str) -> float:
        """Calculate BM25 score"""
        k1, b = 1.5, 0.75  # BM25 parameters
        
        query_terms = self._tokenize(query.lower())
        content_terms = self._tokenize(content.lower())
        
        if not query_terms or not content_terms:
            return 0.0
        
        content_length = len(content_terms)
        avg_doc_length = self.corpus_stats.get('avg_document_length', content_length)
        
        score = 0.0
        content_term_count = Counter(content_terms)
        
        for term in query_terms:
            tf = content_term_count.get(term, 0)
            df = self.corpus_stats.get('term_doc_freq', {}).get(term, 1)
            total_docs = self.corpus_stats.get('total_documents', 1)
            
            idf = math.log((total_docs - df + 0.5) / (df + 0.5))
            
            numerator = tf * (k1 + 1)
            denominator = tf + k1 * (1 - b + b * (content_length / avg_doc_length))
            
            score += idf * (numerator / denominator)
        
        return score
    
    def _calculate_tfidf_score(self, query: str, content: str) -> float:
        """Calculate TF-IDF score"""
        query_terms = self._tokenize(query.lower())
        content_terms = self._tokenize(content.lower())
        
        if not query_terms or not content_terms:
            return 0.0
        
        content_term_count = Counter(content_terms)
        total_content_terms = len(content_terms)
        
        score = 0.0
        for term in query_terms:
            tf = content_term_count.get(term, 0) / total_content_terms
            df = self.corpus_stats.get('term_doc_freq', {}).get(term, 1)
            total_docs = self.corpus_stats.get('total_documents', 1)
            idf = math.log(total_docs / df)
            
            score += tf * idf
        
        return score
    
    def _apply_diversity_reranking(self, ranked_results: List[RankedResult]) -> List[RankedResult]:
        """Apply diversity-aware reranking (MMR-style)"""
        if len(ranked_results) <= 1:
            return ranked_results
        
        lambda_param = self.config.diversity_lambda
        reranked = [ranked_results[0]]  # Start with top result
        remaining = ranked_results[1:]
        
        while remaining and len(reranked) < len(ranked_results):
            best_score = -float('inf')
            best_idx = 0
            
            for i, candidate in enumerate(remaining):
                # Relevance score
                relevance = candidate.ranking_score
                
                # Diversity score (minimum similarity to already selected)
                min_similarity = float('inf')
                for selected in reranked:
                    similarity = self._calculate_result_similarity(candidate, selected)
                    min_similarity = min(min_similarity, similarity)
                
                diversity = 1.0 - min_similarity
                
                # MMR score
                mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity
                
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_idx = i
            
            reranked.append(remaining.pop(best_idx))
        
        return reranked
    
    def _calculate_result_similarity(self, result1: RankedResult, result2: RankedResult) -> float:
        """Calculate similarity between two results for diversity"""
        content1 = getattr(result1.original_result, 'content', '')
        content2 = getattr(result2.original_result, 'content', '')
        
        # Simple lexical similarity
        terms1 = set(self._tokenize(content1.lower()))
        terms2 = set(self._tokenize(content2.lower()))
        
        if not terms1 or not terms2:
            return 0.0
        
        intersection = len(terms1.intersection(terms2))
        union = len(terms1.union(terms2))
        
        return intersection / union if union > 0 else 0.0
    
    def _normalize_scores(self, ranked_results: List[RankedResult]):
        """Normalize ranking scores to [0, 1] range"""
        if not ranked_results:
            return
        
        scores = [result.ranking_score for result in ranked_results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score > min_score:
            for result in ranked_results:
                result.ranking_score = (result.ranking_score - min_score) / (max_score - min_score)
    
    def _generate_explanation(self, features: RankingFeatures, score: float) -> Dict[str, Any]:
        """Generate explanation for ranking score"""
        explanation = {
            'total_score': score,
            'feature_contributions': {},
            'top_features': []
        }
        
        # Calculate feature contributions
        for feature_type, weight in self.config.feature_weights.items():
            feature_value = getattr(features, feature_type.value, 0.0)
            contribution = feature_value * weight
            explanation['feature_contributions'][feature_type.value] = {
                'value': feature_value,
                'weight': weight,
                'contribution': contribution
            }
        
        # Find top contributing features
        contributions = [(k, v['contribution']) for k, v in explanation['feature_contributions'].items()]
        contributions.sort(key=lambda x: x[1], reverse=True)
        explanation['top_features'] = contributions[:3]
        
        return explanation
    
    def _calculate_document_stats(self):
        """Calculate document-level statistics"""
        self.document_stats = {}
        
        for i, doc in enumerate(self.documents):
            content = doc.get('content', '')
            terms = self._tokenize(content.lower())
            
            self.document_stats[i] = {
                'length': len(terms),
                'unique_terms': len(set(terms)),
                'term_counts': Counter(terms)
            }
    
    def _calculate_corpus_stats(self):
        """Calculate corpus-level statistics"""
        if not self.document_stats:
            return
        
        total_docs = len(self.documents)
        total_length = sum(stats['length'] for stats in self.document_stats.values())
        avg_length = total_length / total_docs if total_docs > 0 else 0
        
        # Calculate document frequency for each term
        term_doc_freq = defaultdict(int)
        for stats in self.document_stats.values():
            for term in stats['term_counts']:
                term_doc_freq[term] += 1
        
        self.corpus_stats.update({
            'total_documents': total_docs,
            'avg_document_length': avg_length,
            'term_doc_freq': dict(term_doc_freq),
            'vocabulary_size': len(term_doc_freq)
        })
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        # Remove punctuation and split
        text = re.sub(r'[^\w\s]', ' ', text)
        return [token for token in text.split() if len(token) > 1]
    
    def get_ranking_statistics(self) -> Dict[str, Any]:
        """Get ranking system statistics"""
        return {
            'algorithm': self.config.algorithm.value,
            'feature_weights': {k.value: v for k, v in self.config.feature_weights.items()},
            'corpus_stats': self.corpus_stats,
            'is_fitted': self.is_fitted,
            'diversity_enabled': self.config.apply_diversity
        }


class LearningToRankSystem:
    """
    Learning-to-Rank system for adaptive ranking
    """
    
    def __init__(self):
        self.model = None
        self.feature_names = []
        self.is_trained = False
    
    def train(self, training_data: List[Tuple[RankingFeatures, float]]):
        """
        Train the learning-to-rank model
        
        Args:
            training_data: List of (features, relevance_score) tuples
        """
        # This would implement a proper L2R algorithm
        # For now, we'll use a simple weighted combination
        
        if not training_data:
            return
        
        # Extract feature importance from training data
        feature_sums = defaultdict(float)
        feature_counts = defaultdict(int)
        
        for features, relevance in training_data:
            for feature_name in dir(features):
                if not feature_name.startswith('_') and feature_name != 'custom_features':
                    value = getattr(features, feature_name, 0.0)
                    if isinstance(value, (int, float)):
                        feature_sums[feature_name] += value * relevance
                        feature_counts[feature_name] += 1
        
        # Calculate average feature importance
        self.feature_weights = {}
        for feature_name in feature_sums:
            if feature_counts[feature_name] > 0:
                self.feature_weights[feature_name] = feature_sums[feature_name] / feature_counts[feature_name]
        
        self.is_trained = True
        logger.info(f"L2R model trained with {len(training_data)} examples")
    
    def predict(self, features: RankingFeatures) -> float:
        """Predict relevance score for given features"""
        if not self.is_trained:
            return 0.5
        
        score = 0.0
        total_weight = 0.0
        
        for feature_name, weight in self.feature_weights.items():
            value = getattr(features, feature_name, 0.0)
            if isinstance(value, (int, float)):
                score += value * weight
                total_weight += abs(weight)
        
        return score / total_weight if total_weight > 0 else 0.5


# Utility functions

def evaluate_ranking_quality(ranked_results: List[RankedResult], 
                           ground_truth: List[int],
                           k: int = 10) -> Dict[str, float]:
    """
    Evaluate ranking quality using standard metrics
    
    Args:
        ranked_results: Ranked search results
        ground_truth: List of relevant document IDs
        k: Evaluation cutoff
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not ranked_results or not ground_truth:
        return {}
    
    # Get top-k results
    top_k_results = ranked_results[:k]
    top_k_ids = [getattr(r.original_result, 'chunk_id', i) for i, r in enumerate(top_k_results)]
    
    # Calculate metrics
    relevant_retrieved = len(set(top_k_ids).intersection(set(ground_truth)))
    
    precision_at_k = relevant_retrieved / len(top_k_ids) if top_k_ids else 0.0
    recall_at_k = relevant_retrieved / len(ground_truth) if ground_truth else 0.0
    
    # Calculate NDCG@k
    ndcg_at_k = calculate_ndcg(top_k_ids, ground_truth, k)
    
    # Calculate MAP
    map_score = calculate_map(top_k_ids, ground_truth)
    
    return {
        'precision_at_k': precision_at_k,
        'recall_at_k': recall_at_k,
        'ndcg_at_k': ndcg_at_k,
        'map': map_score,
        'relevant_retrieved': relevant_retrieved,
        'total_retrieved': len(top_k_ids),
        'total_relevant': len(ground_truth)
    }


def calculate_ndcg(retrieved: List[int], relevant: List[int], k: int) -> float:
    """Calculate Normalized Discounted Cumulative Gain"""
    dcg = 0.0
    for i, doc_id in enumerate(retrieved[:k]):
        if doc_id in relevant:
            dcg += 1.0 / math.log2(i + 2)  # +2 because log2(1) = 0
    
    # Calculate ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(relevant))))
    
    return dcg / idcg if idcg > 0 else 0.0


def calculate_map(retrieved: List[int], relevant: List[int]) -> float:
    """Calculate Mean Average Precision"""
    if not relevant:
        return 0.0
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant) if relevant else 0.0