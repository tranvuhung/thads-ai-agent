"""
Query Expansion and Preprocessing System for Semantic Search

This module provides advanced query expansion and preprocessing capabilities
to improve search accuracy and recall in semantic search systems.
"""

import re
import string
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import nltk
from nltk.corpus import stopwords, wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tag import pos_tag
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from collections import defaultdict, Counter
import json
import logging

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
    nltk.download('omw-1.4', quiet=True)
except:
    pass

class ExpansionMethod(Enum):
    """Query expansion methods"""
    SYNONYM_EXPANSION = "synonym_expansion"
    SEMANTIC_EXPANSION = "semantic_expansion"
    STATISTICAL_EXPANSION = "statistical_expansion"
    CONTEXTUAL_EXPANSION = "contextual_expansion"
    DOMAIN_EXPANSION = "domain_expansion"
    PHONETIC_EXPANSION = "phonetic_expansion"
    ABBREVIATION_EXPANSION = "abbreviation_expansion"

class PreprocessingStep(Enum):
    """Text preprocessing steps"""
    TOKENIZATION = "tokenization"
    LOWERCASING = "lowercasing"
    PUNCTUATION_REMOVAL = "punctuation_removal"
    STOPWORD_REMOVAL = "stopword_removal"
    STEMMING = "stemming"
    LEMMATIZATION = "lemmatization"
    SPELL_CORRECTION = "spell_correction"
    NORMALIZATION = "normalization"

@dataclass
class QueryExpansionConfig:
    """Configuration for query expansion"""
    expansion_methods: List[ExpansionMethod]
    max_expansions_per_term: int = 3
    min_similarity_threshold: float = 0.7
    use_pos_tagging: bool = True
    preserve_original_query: bool = True
    expansion_weight: float = 0.5
    enable_phrase_expansion: bool = True
    domain_specific_terms: Optional[Dict[str, List[str]]] = None

@dataclass
class PreprocessingConfig:
    """Configuration for text preprocessing"""
    steps: List[PreprocessingStep]
    language: str = "english"
    custom_stopwords: Optional[Set[str]] = None
    preserve_entities: bool = True
    min_token_length: int = 2
    max_token_length: int = 50

@dataclass
class ExpandedQuery:
    """Expanded query result"""
    original_query: str
    expanded_terms: List[str]
    expansion_weights: Dict[str, float]
    preprocessing_applied: List[str]
    confidence_score: float
    expansion_sources: Dict[str, str]

class QueryPreprocessor:
    """Advanced text preprocessing for queries"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model for NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            self.nlp = None
            logging.warning("spaCy model not found. Entity preservation disabled.")
        
        # Initialize stopwords
        self.stopwords = set(stopwords.words(self.config.language))
        if self.config.custom_stopwords:
            self.stopwords.update(self.config.custom_stopwords)
    
    def preprocess(self, text: str) -> Tuple[str, List[str]]:
        """
        Preprocess text according to configuration
        
        Args:
            text: Input text to preprocess
            
        Returns:
            Tuple of (processed_text, applied_steps)
        """
        processed_text = text
        applied_steps = []
        
        # Extract entities if preservation is enabled
        entities = set()
        if self.config.preserve_entities and self.nlp:
            doc = self.nlp(text)
            entities = {ent.text.lower() for ent in doc.ents}
        
        for step in self.config.steps:
            if step == PreprocessingStep.LOWERCASING:
                processed_text = processed_text.lower()
                applied_steps.append("lowercasing")
            
            elif step == PreprocessingStep.PUNCTUATION_REMOVAL:
                processed_text = self._remove_punctuation(processed_text, entities)
                applied_steps.append("punctuation_removal")
            
            elif step == PreprocessingStep.TOKENIZATION:
                processed_text = self._tokenize(processed_text)
                applied_steps.append("tokenization")
            
            elif step == PreprocessingStep.STOPWORD_REMOVAL:
                processed_text = self._remove_stopwords(processed_text, entities)
                applied_steps.append("stopword_removal")
            
            elif step == PreprocessingStep.STEMMING:
                processed_text = self._apply_stemming(processed_text)
                applied_steps.append("stemming")
            
            elif step == PreprocessingStep.LEMMATIZATION:
                processed_text = self._apply_lemmatization(processed_text)
                applied_steps.append("lemmatization")
            
            elif step == PreprocessingStep.NORMALIZATION:
                processed_text = self._normalize_text(processed_text)
                applied_steps.append("normalization")
        
        return processed_text, applied_steps
    
    def _remove_punctuation(self, text: str, entities: Set[str]) -> str:
        """Remove punctuation while preserving entities"""
        # Simple punctuation removal with entity preservation
        result = ""
        for char in text:
            if char not in string.punctuation:
                result += char
            else:
                result += " "
        return re.sub(r'\s+', ' ', result).strip()
    
    def _tokenize(self, text: str) -> str:
        """Tokenize text"""
        if isinstance(text, str):
            tokens = word_tokenize(text)
            # Filter by length
            tokens = [
                token for token in tokens 
                if self.config.min_token_length <= len(token) <= self.config.max_token_length
            ]
            return " ".join(tokens)
        return text
    
    def _remove_stopwords(self, text: str, entities: Set[str]) -> str:
        """Remove stopwords while preserving entities"""
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        
        filtered_tokens = []
        for token in tokens:
            if token.lower() not in self.stopwords or token.lower() in entities:
                filtered_tokens.append(token)
        
        return " ".join(filtered_tokens)
    
    def _apply_stemming(self, text: str) -> str:
        """Apply stemming to tokens"""
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        return " ".join(stemmed_tokens)
    
    def _apply_lemmatization(self, text: str) -> str:
        """Apply lemmatization to tokens"""
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text
        
        # Get POS tags for better lemmatization
        pos_tags = pos_tag(tokens)
        lemmatized_tokens = []
        
        for token, pos in pos_tags:
            # Convert POS tag to WordNet format
            wordnet_pos = self._get_wordnet_pos(pos)
            lemmatized_token = self.lemmatizer.lemmatize(token, wordnet_pos)
            lemmatized_tokens.append(lemmatized_token)
        
        return " ".join(lemmatized_tokens)
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """Convert TreeBank POS tag to WordNet POS tag"""
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            return wordnet.NOUN
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text (handle contractions, special characters, etc.)"""
        # Handle contractions
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

class QueryExpander:
    """Advanced query expansion system"""
    
    def __init__(self, config: QueryExpansionConfig):
        self.config = config
        self.lemmatizer = WordNetLemmatizer()
        
        # Initialize domain-specific terms
        self.domain_terms = config.domain_specific_terms or {}
        
        # Initialize abbreviation dictionary
        self.abbreviations = self._load_abbreviations()
        
        # Initialize statistical models (would be trained on corpus)
        self.term_cooccurrence = defaultdict(Counter)
        self.term_frequencies = Counter()
    
    def expand_query(self, query: str, preprocessed_query: str = None) -> ExpandedQuery:
        """
        Expand query using configured methods
        
        Args:
            query: Original query
            preprocessed_query: Preprocessed version of query
            
        Returns:
            ExpandedQuery object with expansion results
        """
        if preprocessed_query is None:
            preprocessed_query = query
        
        expanded_terms = []
        expansion_weights = {}
        expansion_sources = {}
        
        # Tokenize query
        tokens = word_tokenize(preprocessed_query.lower())
        
        for method in self.config.expansion_methods:
            if method == ExpansionMethod.SYNONYM_EXPANSION:
                synonyms = self._expand_with_synonyms(tokens)
                expanded_terms.extend(synonyms)
                for term in synonyms:
                    expansion_weights[term] = 0.8
                    expansion_sources[term] = "synonym"
            
            elif method == ExpansionMethod.SEMANTIC_EXPANSION:
                semantic_terms = self._expand_semantically(tokens)
                expanded_terms.extend(semantic_terms)
                for term in semantic_terms:
                    expansion_weights[term] = 0.7
                    expansion_sources[term] = "semantic"
            
            elif method == ExpansionMethod.STATISTICAL_EXPANSION:
                statistical_terms = self._expand_statistically(tokens)
                expanded_terms.extend(statistical_terms)
                for term in statistical_terms:
                    expansion_weights[term] = 0.6
                    expansion_sources[term] = "statistical"
            
            elif method == ExpansionMethod.DOMAIN_EXPANSION:
                domain_terms = self._expand_with_domain_terms(tokens)
                expanded_terms.extend(domain_terms)
                for term in domain_terms:
                    expansion_weights[term] = 0.9
                    expansion_sources[term] = "domain"
            
            elif method == ExpansionMethod.ABBREVIATION_EXPANSION:
                abbreviation_terms = self._expand_abbreviations(tokens)
                expanded_terms.extend(abbreviation_terms)
                for term in abbreviation_terms:
                    expansion_weights[term] = 0.95
                    expansion_sources[term] = "abbreviation"
        
        # Remove duplicates and limit expansions
        unique_expansions = list(set(expanded_terms))
        if len(unique_expansions) > self.config.max_expansions_per_term * len(tokens):
            # Sort by weight and take top expansions
            sorted_expansions = sorted(
                unique_expansions,
                key=lambda x: expansion_weights.get(x, 0),
                reverse=True
            )
            unique_expansions = sorted_expansions[:self.config.max_expansions_per_term * len(tokens)]
        
        # Calculate confidence score
        confidence_score = self._calculate_confidence(unique_expansions, expansion_weights)
        
        return ExpandedQuery(
            original_query=query,
            expanded_terms=unique_expansions,
            expansion_weights=expansion_weights,
            preprocessing_applied=[],
            confidence_score=confidence_score,
            expansion_sources=expansion_sources
        )
    
    def _expand_with_synonyms(self, tokens: List[str]) -> List[str]:
        """Expand query using WordNet synonyms"""
        synonyms = []
        
        for token in tokens:
            synsets = wordnet.synsets(token)
            for synset in synsets[:2]:  # Limit to top 2 synsets
                for lemma in synset.lemmas()[:self.config.max_expansions_per_term]:
                    synonym = lemma.name().replace('_', ' ')
                    if synonym != token and len(synonym) > 2:
                        synonyms.append(synonym)
        
        return synonyms
    
    def _expand_semantically(self, tokens: List[str]) -> List[str]:
        """Expand query using semantic relationships"""
        semantic_terms = []
        
        for token in tokens:
            # Get hypernyms and hyponyms
            synsets = wordnet.synsets(token)
            for synset in synsets[:1]:  # Limit to top synset
                # Hypernyms (more general terms)
                for hypernym in synset.hypernyms()[:2]:
                    for lemma in hypernym.lemmas()[:1]:
                        term = lemma.name().replace('_', ' ')
                        if term != token:
                            semantic_terms.append(term)
                
                # Hyponyms (more specific terms)
                for hyponym in synset.hyponyms()[:2]:
                    for lemma in hyponym.lemmas()[:1]:
                        term = lemma.name().replace('_', ' ')
                        if term != token:
                            semantic_terms.append(term)
        
        return semantic_terms
    
    def _expand_statistically(self, tokens: List[str]) -> List[str]:
        """Expand query using statistical co-occurrence"""
        statistical_terms = []
        
        for token in tokens:
            if token in self.term_cooccurrence:
                # Get most frequent co-occurring terms
                cooccurring_terms = self.term_cooccurrence[token].most_common(
                    self.config.max_expansions_per_term
                )
                for term, freq in cooccurring_terms:
                    if term != token:
                        statistical_terms.append(term)
        
        return statistical_terms
    
    def _expand_with_domain_terms(self, tokens: List[str]) -> List[str]:
        """Expand query using domain-specific terms"""
        domain_terms = []
        
        for token in tokens:
            for domain, terms in self.domain_terms.items():
                if token in terms:
                    # Add related terms from the same domain
                    related_terms = [t for t in terms if t != token]
                    domain_terms.extend(related_terms[:self.config.max_expansions_per_term])
        
        return domain_terms
    
    def _expand_abbreviations(self, tokens: List[str]) -> List[str]:
        """Expand abbreviations to full forms"""
        expanded_terms = []
        
        for token in tokens:
            if token.upper() in self.abbreviations:
                expanded_terms.append(self.abbreviations[token.upper()])
        
        return expanded_terms
    
    def _load_abbreviations(self) -> Dict[str, str]:
        """Load common abbreviations dictionary"""
        # This would typically be loaded from a file
        return {
            "AI": "artificial intelligence",
            "ML": "machine learning",
            "NLP": "natural language processing",
            "API": "application programming interface",
            "UI": "user interface",
            "UX": "user experience",
            "DB": "database",
            "SQL": "structured query language",
            "HTTP": "hypertext transfer protocol",
            "URL": "uniform resource locator"
        }
    
    def _calculate_confidence(self, expansions: List[str], weights: Dict[str, float]) -> float:
        """Calculate confidence score for expansions"""
        if not expansions:
            return 0.0
        
        total_weight = sum(weights.get(term, 0) for term in expansions)
        avg_weight = total_weight / len(expansions)
        
        # Normalize to 0-1 range
        return min(avg_weight, 1.0)
    
    def update_statistical_model(self, documents: List[str]):
        """Update statistical models with new documents"""
        # This would update term co-occurrence and frequency statistics
        # Implementation would depend on the specific corpus and requirements
        pass

class IntegratedQueryProcessor:
    """Integrated query preprocessing and expansion system"""
    
    def __init__(self, 
                 preprocessing_config: PreprocessingConfig,
                 expansion_config: QueryExpansionConfig):
        self.preprocessor = QueryPreprocessor(preprocessing_config)
        self.expander = QueryExpander(expansion_config)
    
    def process_query(self, query: str) -> ExpandedQuery:
        """
        Process query with preprocessing and expansion
        
        Args:
            query: Input query string
            
        Returns:
            ExpandedQuery with all processing results
        """
        # Preprocess query
        preprocessed_query, preprocessing_steps = self.preprocessor.preprocess(query)
        
        # Expand query
        expanded_query = self.expander.expand_query(query, preprocessed_query)
        
        # Update preprocessing information
        expanded_query.preprocessing_applied = preprocessing_steps
        
        return expanded_query
    
    def get_final_query_terms(self, expanded_query: ExpandedQuery) -> List[str]:
        """
        Get final list of query terms for search
        
        Args:
            expanded_query: ExpandedQuery object
            
        Returns:
            List of final query terms with weights applied
        """
        terms = []
        
        # Add original query terms
        original_tokens = word_tokenize(expanded_query.original_query.lower())
        terms.extend(original_tokens)
        
        # Add expanded terms with weight consideration
        for term in expanded_query.expanded_terms:
            weight = expanded_query.expansion_weights.get(term, 0.5)
            if weight >= self.expander.config.min_similarity_threshold:
                terms.append(term)
        
        return list(set(terms))  # Remove duplicates

# Utility functions
def create_default_preprocessing_config() -> PreprocessingConfig:
    """Create default preprocessing configuration"""
    return PreprocessingConfig(
        steps=[
            PreprocessingStep.LOWERCASING,
            PreprocessingStep.PUNCTUATION_REMOVAL,
            PreprocessingStep.TOKENIZATION,
            PreprocessingStep.STOPWORD_REMOVAL,
            PreprocessingStep.LEMMATIZATION,
            PreprocessingStep.NORMALIZATION
        ],
        language="english",
        preserve_entities=True,
        min_token_length=2,
        max_token_length=50
    )

def create_default_expansion_config() -> QueryExpansionConfig:
    """Create default expansion configuration"""
    return QueryExpansionConfig(
        expansion_methods=[
            ExpansionMethod.SYNONYM_EXPANSION,
            ExpansionMethod.SEMANTIC_EXPANSION,
            ExpansionMethod.ABBREVIATION_EXPANSION
        ],
        max_expansions_per_term=3,
        min_similarity_threshold=0.7,
        use_pos_tagging=True,
        preserve_original_query=True,
        expansion_weight=0.5
    )

def create_integrated_query_processor() -> IntegratedQueryProcessor:
    """Create integrated query processor with default configurations"""
    preprocessing_config = create_default_preprocessing_config()
    expansion_config = create_default_expansion_config()
    
    return IntegratedQueryProcessor(preprocessing_config, expansion_config)