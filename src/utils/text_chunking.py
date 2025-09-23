"""
Text Chunking Module for Legal Documents

This module provides various strategies for chunking legal documents into
meaningful segments for embedding and vector search.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    LEGAL_SECTION = "legal_section"


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk"""
    chunk_index: int
    start_position: int
    end_position: int
    chunk_size: int
    word_count: int
    sentence_count: int
    content_type: str = "body"
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    coherence_score: float = 0.0
    completeness_score: float = 0.0


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    content: str
    metadata: ChunkMetadata


class BaseChunker:
    """Base class for text chunking strategies"""
    
    def __init__(self, overlap: int = 0, min_chunk_size: int = 50, max_chunk_size: int = 2000):
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk_text(self, text: str, document_id: Optional[int] = None) -> List[TextChunk]:
        """
        Chunk text using the specific strategy
        
        Args:
            text: Input text to chunk
            document_id: Optional document ID for context
            
        Returns:
            List of TextChunk objects
        """
        raise NotImplementedError("Subclasses must implement chunk_text method")
    
    def _calculate_word_count(self, text: str) -> int:
        """Calculate word count for a text"""
        return len(word_tokenize(text))
    
    def _calculate_sentence_count(self, text: str) -> int:
        """Calculate sentence count for a text"""
        return len(sent_tokenize(text))
    
    def _calculate_coherence_score(self, text: str) -> float:
        """
        Calculate coherence score based on sentence connectivity
        Simple implementation - can be enhanced with more sophisticated NLP
        """
        sentences = sent_tokenize(text)
        if len(sentences) <= 1:
            return 1.0
        
        # Simple coherence based on sentence length consistency
        lengths = [len(s.split()) for s in sentences]
        if not lengths:
            return 0.0
        
        avg_length = sum(lengths) / len(lengths)
        variance = sum((l - avg_length) ** 2 for l in lengths) / len(lengths)
        
        # Normalize coherence score (lower variance = higher coherence)
        coherence = max(0.0, 1.0 - (variance / (avg_length ** 2 + 1)))
        return min(1.0, coherence)
    
    def _calculate_completeness_score(self, text: str) -> float:
        """
        Calculate completeness score based on sentence/paragraph structure
        """
        # Check if text ends with proper punctuation
        text = text.strip()
        if not text:
            return 0.0
        
        # Check for complete sentences
        sentences = sent_tokenize(text)
        complete_sentences = sum(1 for s in sentences if s.strip().endswith(('.', '!', '?', ':', ';')))
        
        if not sentences:
            return 0.0
        
        completeness = complete_sentences / len(sentences)
        
        # Bonus for starting with capital letter
        if text[0].isupper():
            completeness += 0.1
        
        return min(1.0, completeness)


class SentenceChunker(BaseChunker):
    """Chunks text by sentences with optional grouping"""
    
    def __init__(self, sentences_per_chunk: int = 3, **kwargs):
        super().__init__(**kwargs)
        self.sentences_per_chunk = sentences_per_chunk
    
    def chunk_text(self, text: str, document_id: Optional[int] = None) -> List[TextChunk]:
        """Chunk text by grouping sentences"""
        sentences = sent_tokenize(text)
        chunks = []
        
        for i in range(0, len(sentences), self.sentences_per_chunk):
            chunk_sentences = sentences[i:i + self.sentences_per_chunk]
            
            # Add overlap if specified
            if self.overlap > 0 and i > 0:
                overlap_start = max(0, i - self.overlap)
                overlap_sentences = sentences[overlap_start:i]
                chunk_sentences = overlap_sentences + chunk_sentences
            
            chunk_content = ' '.join(chunk_sentences)
            
            # Skip chunks that are too small
            if len(chunk_content) < self.min_chunk_size:
                continue
            
            # Calculate positions
            start_pos = text.find(chunk_sentences[0]) if chunk_sentences else 0
            end_pos = start_pos + len(chunk_content)
            
            metadata = ChunkMetadata(
                chunk_index=len(chunks),
                start_position=start_pos,
                end_position=end_pos,
                chunk_size=len(chunk_content),
                word_count=self._calculate_word_count(chunk_content),
                sentence_count=len(chunk_sentences),
                coherence_score=self._calculate_coherence_score(chunk_content),
                completeness_score=self._calculate_completeness_score(chunk_content)
            )
            
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks


class ParagraphChunker(BaseChunker):
    """Chunks text by paragraphs"""
    
    def chunk_text(self, text: str, document_id: Optional[int] = None) -> List[TextChunk]:
        """Chunk text by paragraphs"""
        # Split by double newlines or more
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_position = 0
        
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph or len(paragraph) < self.min_chunk_size:
                current_position += len(paragraph) + 2  # Account for newlines
                continue
            
            # If paragraph is too long, split it further
            if len(paragraph) > self.max_chunk_size:
                sub_chunks = self._split_long_paragraph(paragraph, current_position, len(chunks))
                chunks.extend(sub_chunks)
            else:
                metadata = ChunkMetadata(
                    chunk_index=len(chunks),
                    start_position=current_position,
                    end_position=current_position + len(paragraph),
                    chunk_size=len(paragraph),
                    word_count=self._calculate_word_count(paragraph),
                    sentence_count=self._calculate_sentence_count(paragraph),
                    coherence_score=self._calculate_coherence_score(paragraph),
                    completeness_score=self._calculate_completeness_score(paragraph)
                )
                
                chunks.append(TextChunk(content=paragraph, metadata=metadata))
            
            current_position += len(paragraph) + 2
        
        return chunks
    
    def _split_long_paragraph(self, paragraph: str, start_pos: int, chunk_index_offset: int) -> List[TextChunk]:
        """Split a long paragraph into smaller chunks"""
        sentences = sent_tokenize(paragraph)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            if current_length + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                metadata = ChunkMetadata(
                    chunk_index=chunk_index_offset + len(chunks),
                    start_position=start_pos,
                    end_position=start_pos + len(chunk_content),
                    chunk_size=len(chunk_content),
                    word_count=self._calculate_word_count(chunk_content),
                    sentence_count=len(current_chunk),
                    coherence_score=self._calculate_coherence_score(chunk_content),
                    completeness_score=self._calculate_completeness_score(chunk_content)
                )
                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                
                start_pos += len(chunk_content) + 1
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += len(sentence) + 1
        
        # Add remaining sentences as final chunk
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            metadata = ChunkMetadata(
                chunk_index=chunk_index_offset + len(chunks),
                start_position=start_pos,
                end_position=start_pos + len(chunk_content),
                chunk_size=len(chunk_content),
                word_count=self._calculate_word_count(chunk_content),
                sentence_count=len(current_chunk),
                coherence_score=self._calculate_coherence_score(chunk_content),
                completeness_score=self._calculate_completeness_score(chunk_content)
            )
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks


class FixedSizeChunker(BaseChunker):
    """Chunks text into fixed-size segments"""
    
    def __init__(self, chunk_size: int = 1000, **kwargs):
        super().__init__(**kwargs)
        self.chunk_size = chunk_size
    
    def chunk_text(self, text: str, document_id: Optional[int] = None) -> List[TextChunk]:
        """Chunk text into fixed-size segments with sentence boundary respect"""
        chunks = []
        sentences = sent_tokenize(text)
        current_chunk = []
        current_length = 0
        current_position = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size and we have content
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_content = ' '.join(current_chunk)
                
                metadata = ChunkMetadata(
                    chunk_index=len(chunks),
                    start_position=current_position,
                    end_position=current_position + len(chunk_content),
                    chunk_size=len(chunk_content),
                    word_count=self._calculate_word_count(chunk_content),
                    sentence_count=len(current_chunk),
                    coherence_score=self._calculate_coherence_score(chunk_content),
                    completeness_score=self._calculate_completeness_score(chunk_content)
                )
                
                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                
                # Handle overlap
                if self.overlap > 0:
                    overlap_content = chunk_content[-self.overlap:]
                    current_chunk = [overlap_content, sentence]
                    current_length = len(overlap_content) + sentence_length + 1
                else:
                    current_chunk = [sentence]
                    current_length = sentence_length
                
                current_position += len(chunk_content) + 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + (1 if current_chunk else 0)
        
        # Add final chunk if there's remaining content
        if current_chunk:
            chunk_content = ' '.join(current_chunk)
            metadata = ChunkMetadata(
                chunk_index=len(chunks),
                start_position=current_position,
                end_position=current_position + len(chunk_content),
                chunk_size=len(chunk_content),
                word_count=self._calculate_word_count(chunk_content),
                sentence_count=len(current_chunk),
                coherence_score=self._calculate_coherence_score(chunk_content),
                completeness_score=self._calculate_completeness_score(chunk_content)
            )
            chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks


class LegalSectionChunker(BaseChunker):
    """Chunks text based on legal document structure"""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Vietnamese legal document patterns
        self.section_patterns = [
            r'(?:PHẦN|Phần)\s+([IVX]+|[0-9]+)[\.\:]?\s*([^\n]+)',  # PHẦN I, II, etc.
            r'(?:CHƯƠNG|Chương)\s+([IVX]+|[0-9]+)[\.\:]?\s*([^\n]+)',  # CHƯƠNG I, II, etc.
            r'(?:Điều|ĐIỀU)\s+([0-9]+)[\.\:]?\s*([^\n]+)',  # Điều 1, 2, etc.
            r'(?:Khoản|KHOẢN)\s+([0-9]+)[\.\:]?\s*([^\n]+)',  # Khoản 1, 2, etc.
            r'(?:Mục|MỤC)\s+([0-9]+)[\.\:]?\s*([^\n]+)',  # Mục 1, 2, etc.
            r'^([0-9]+)[\.\)]\s*([^\n]+)',  # Numbered items
            r'^([a-z][\.\)])\s*([^\n]+)',  # Lettered items
        ]
    
    def chunk_text(self, text: str, document_id: Optional[int] = None) -> List[TextChunk]:
        """Chunk text based on legal document structure"""
        chunks = []
        lines = text.split('\n')
        current_section = []
        current_section_title = None
        current_position = 0
        
        for line in lines:
            line = line.strip()
            if not line:
                current_position += 1
                continue
            
            # Check if line matches any section pattern
            section_match = self._find_section_match(line)
            
            if section_match:
                # Save previous section if it exists
                if current_section:
                    chunk_content = '\n'.join(current_section)
                    if len(chunk_content) >= self.min_chunk_size:
                        metadata = ChunkMetadata(
                            chunk_index=len(chunks),
                            start_position=current_position - len(chunk_content),
                            end_position=current_position,
                            chunk_size=len(chunk_content),
                            word_count=self._calculate_word_count(chunk_content),
                            sentence_count=self._calculate_sentence_count(chunk_content),
                            content_type="legal_section",
                            section_title=current_section_title,
                            coherence_score=self._calculate_coherence_score(chunk_content),
                            completeness_score=self._calculate_completeness_score(chunk_content)
                        )
                        chunks.append(TextChunk(content=chunk_content, metadata=metadata))
                
                # Start new section
                current_section = [line]
                current_section_title = section_match
            else:
                current_section.append(line)
            
            current_position += len(line) + 1
        
        # Add final section
        if current_section:
            chunk_content = '\n'.join(current_section)
            if len(chunk_content) >= self.min_chunk_size:
                metadata = ChunkMetadata(
                    chunk_index=len(chunks),
                    start_position=current_position - len(chunk_content),
                    end_position=current_position,
                    chunk_size=len(chunk_content),
                    word_count=self._calculate_word_count(chunk_content),
                    sentence_count=self._calculate_sentence_count(chunk_content),
                    content_type="legal_section",
                    section_title=current_section_title,
                    coherence_score=self._calculate_coherence_score(chunk_content),
                    completeness_score=self._calculate_completeness_score(chunk_content)
                )
                chunks.append(TextChunk(content=chunk_content, metadata=metadata))
        
        return chunks
    
    def _find_section_match(self, line: str) -> Optional[str]:
        """Find if line matches any legal section pattern"""
        for pattern in self.section_patterns:
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                return line
        return None


class TextChunkingService:
    """Service for managing text chunking operations"""
    
    def __init__(self):
        self.chunkers = {
            ChunkingStrategy.SENTENCE: SentenceChunker,
            ChunkingStrategy.PARAGRAPH: ParagraphChunker,
            ChunkingStrategy.FIXED_SIZE: FixedSizeChunker,
            ChunkingStrategy.LEGAL_SECTION: LegalSectionChunker,
        }
    
    def chunk_document(
        self,
        text: str,
        strategy: ChunkingStrategy = ChunkingStrategy.PARAGRAPH,
        **chunker_kwargs
    ) -> List[TextChunk]:
        """
        Chunk a document using the specified strategy
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy to use
            **chunker_kwargs: Additional arguments for the chunker
            
        Returns:
            List of TextChunk objects
        """
        if strategy not in self.chunkers:
            raise ValueError(f"Unsupported chunking strategy: {strategy}")
        
        chunker_class = self.chunkers[strategy]
        chunker = chunker_class(**chunker_kwargs)
        
        try:
            chunks = chunker.chunk_text(text)
            logger.info(f"Successfully chunked text into {len(chunks)} chunks using {strategy.value} strategy")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text with {strategy.value} strategy: {str(e)}")
            raise
    
    def get_optimal_strategy(self, text: str, document_type: str = None) -> ChunkingStrategy:
        """
        Suggest optimal chunking strategy based on text characteristics
        
        Args:
            text: Input text to analyze
            document_type: Type of document (if known)
            
        Returns:
            Recommended chunking strategy
        """
        # Simple heuristics for strategy selection
        if document_type and "legal" in document_type.lower():
            return ChunkingStrategy.LEGAL_SECTION
        
        # Check for paragraph structure
        paragraph_count = len(re.split(r'\n\s*\n', text))
        sentence_count = len(sent_tokenize(text))
        
        if paragraph_count > 5 and len(text) > 2000:
            return ChunkingStrategy.PARAGRAPH
        elif sentence_count > 20:
            return ChunkingStrategy.SENTENCE
        else:
            return ChunkingStrategy.FIXED_SIZE