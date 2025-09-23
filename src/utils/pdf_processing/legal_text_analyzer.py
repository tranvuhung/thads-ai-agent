"""
Module for analyzing legal text structure and content.
Provides specialized analysis for Vietnamese legal documents.
"""
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import Counter


@dataclass
class LegalTextAnalysis:
    """Results of legal text analysis"""
    document_structure: Dict[str, Any]
    legal_terminology: Dict[str, int]
    complexity_metrics: Dict[str, float]
    readability_score: float
    key_sections: List[Dict[str, Any]]
    legal_citations: List[str]
    entities: Dict[str, List[str]]


class LegalTextAnalyzer:
    """
    Analyzer for legal text structure, terminology, and complexity.
    """
    
    def __init__(self):
        """Initialize the LegalTextAnalyzer"""
        self.legal_terms = self._load_legal_terminology()
        self.section_patterns = self._compile_section_patterns()
        
    def _load_legal_terminology(self) -> Dict[str, List[str]]:
        """Load legal terminology categories"""
        return {
            'court_terms': [
                'tòa án', 'thẩm phán', 'hội thẩm', 'kiểm sát viên', 'luật sư',
                'bị cáo', 'nguyên đơn', 'bị đơn', 'người bào chữa', 'người bảo vệ quyền lợi'
            ],
            'procedure_terms': [
                'tố tụng', 'xét xử', 'điều tra', 'truy tố', 'kháng cáo', 'giám đốc thẩm',
                'tái thẩm', 'phiên tòa', 'biên bản', 'quyết định', 'bản án'
            ],
            'criminal_terms': [
                'tội phạm', 'hình phạt', 'tù giam', 'phạt tiền', 'cải tạo không giam giữ',
                'tịch thu', 'tái phạm', 'đồng phạm', 'tình tiết tăng nặng', 'tình tiết giảm nhẹ'
            ],
            'civil_terms': [
                'quyền sở hữu', 'nghĩa vụ', 'hợp đồng', 'bồi thường', 'thiệt hại',
                'tranh chấp', 'tài sản', 'thừa kế', 'hôn nhân gia đình'
            ],
            'legal_references': [
                'bộ luật', 'luật', 'nghị định', 'thông tư', 'quyết định', 'chỉ thị',
                'điều', 'khoản', 'điểm', 'chương', 'mục', 'phần'
            ]
        }
    
    def _compile_section_patterns(self) -> Dict[str, re.Pattern]:
        """Compile patterns for identifying document sections"""
        return {
            'header': re.compile(r'^[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ\s]+$', re.MULTILINE),
            'roman_section': re.compile(r'^[IVX]+\.\s*(.+)$', re.MULTILINE),
            'numbered_section': re.compile(r'^\d+\.\s*(.+)$', re.MULTILINE),
            'lettered_section': re.compile(r'^[a-z]\)\s*(.+)$', re.MULTILINE),
            'legal_article': re.compile(r'Điều\s+\d+', re.IGNORECASE),
            'legal_clause': re.compile(r'Khoản\s+\d+', re.IGNORECASE),
            'legal_point': re.compile(r'Điểm\s+[a-z]', re.IGNORECASE),
        }
    
    def analyze_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Analyze the structure of a legal document.
        
        Args:
            text (str): Legal document text
            
        Returns:
            Dict: Document structure analysis
        """
        lines = text.split('\n')
        structure = {
            'total_lines': len(lines),
            'sections': [],
            'subsections': [],
            'hierarchy_depth': 0,
            'has_legal_citations': False,
            'section_types': Counter()
        }
        
        current_section = None
        section_level = 0
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check for different section types
            if self.section_patterns['roman_section'].match(line):
                section_level = 1
                current_section = {
                    'type': 'roman_section',
                    'level': section_level,
                    'title': line,
                    'line_number': i + 1,
                    'content_lines': []
                }
                structure['sections'].append(current_section)
                structure['section_types']['roman'] += 1
                
            elif self.section_patterns['numbered_section'].match(line):
                section_level = 2
                current_section = {
                    'type': 'numbered_section',
                    'level': section_level,
                    'title': line,
                    'line_number': i + 1,
                    'content_lines': []
                }
                structure['subsections'].append(current_section)
                structure['section_types']['numbered'] += 1
                
            elif self.section_patterns['lettered_section'].match(line):
                section_level = 3
                structure['section_types']['lettered'] += 1
                
            elif current_section:
                current_section['content_lines'].append(line)
            
            # Check for legal citations
            if (self.section_patterns['legal_article'].search(line) or
                self.section_patterns['legal_clause'].search(line) or
                self.section_patterns['legal_point'].search(line)):
                structure['has_legal_citations'] = True
        
        structure['hierarchy_depth'] = max(section_level, structure['hierarchy_depth'])
        
        return structure
    
    def analyze_legal_terminology(self, text: str) -> Dict[str, int]:
        """
        Analyze legal terminology usage in the text.
        
        Args:
            text (str): Legal document text
            
        Returns:
            Dict: Terminology frequency analysis
        """
        text_lower = text.lower()
        terminology_count = {}
        
        for category, terms in self.legal_terms.items():
            category_count = {}
            for term in terms:
                count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if count > 0:
                    category_count[term] = count
            terminology_count[category] = category_count
        
        return terminology_count
    
    def calculate_complexity_metrics(self, text: str) -> Dict[str, float]:
        """
        Calculate complexity metrics for legal text.
        
        Args:
            text (str): Legal document text
            
        Returns:
            Dict: Complexity metrics
        """
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Basic metrics
        total_words = len(words)
        total_sentences = len(sentences)
        total_chars = len(text)
        
        # Calculate metrics
        avg_sentence_length = total_words / total_sentences if total_sentences > 0 else 0
        avg_word_length = sum(len(word) for word in words) / total_words if total_words > 0 else 0
        
        # Legal complexity indicators
        legal_term_density = self._calculate_legal_term_density(text)
        citation_density = self._calculate_citation_density(text)
        
        return {
            'average_sentence_length': avg_sentence_length,
            'average_word_length': avg_word_length,
            'legal_term_density': legal_term_density,
            'citation_density': citation_density,
            'total_words': total_words,
            'total_sentences': total_sentences,
            'total_characters': total_chars
        }
    
    def _calculate_legal_term_density(self, text: str) -> float:
        """Calculate density of legal terminology"""
        words = re.findall(r'\b\w+\b', text.lower())
        total_words = len(words)
        
        legal_word_count = 0
        for category, terms in self.legal_terms.items():
            for term in terms:
                legal_word_count += len(re.findall(r'\b' + re.escape(term) + r'\b', text.lower()))
        
        return legal_word_count / total_words if total_words > 0 else 0
    
    def _calculate_citation_density(self, text: str) -> float:
        """Calculate density of legal citations"""
        words = re.findall(r'\b\w+\b', text)
        total_words = len(words)
        
        citations = (len(self.section_patterns['legal_article'].findall(text)) +
                    len(self.section_patterns['legal_clause'].findall(text)) +
                    len(self.section_patterns['legal_point'].findall(text)))
        
        return citations / total_words * 100 if total_words > 0 else 0
    
    def calculate_readability_score(self, text: str) -> float:
        """
        Calculate readability score adapted for Vietnamese legal text.
        
        Args:
            text (str): Legal document text
            
        Returns:
            float: Readability score (0-100, higher is more readable)
        """
        complexity = self.calculate_complexity_metrics(text)
        
        # Adapted readability formula for Vietnamese legal text
        # Lower scores indicate higher complexity (less readable)
        sentence_factor = min(complexity['average_sentence_length'] / 20, 1.0)
        word_factor = min(complexity['average_word_length'] / 8, 1.0)
        legal_factor = min(complexity['legal_term_density'] * 10, 1.0)
        
        # Invert factors so higher complexity gives lower readability
        readability = 100 * (1 - (sentence_factor + word_factor + legal_factor) / 3)
        
        return max(0, min(100, readability))
    
    def extract_key_sections(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract key sections from legal document.
        
        Args:
            text (str): Legal document text
            
        Returns:
            List[Dict]: Key sections with metadata
        """
        key_sections = []
        
        # Common important sections in Vietnamese legal documents
        important_patterns = {
            'verdict': re.compile(r'(QUYẾT ĐỊNH|TUYÊN BỐ).*?(?=\n\n|\n[A-Z])', re.IGNORECASE | re.DOTALL),
            'charges': re.compile(r'(về tội|phạm tội).*?(?=\n\n|\n[A-Z])', re.IGNORECASE | re.DOTALL),
            'facts': re.compile(r'(NỘI DUNG VỤ ÁN|DIỄN BIẾN VỤ ÁN).*?(?=\n\n|\n[A-Z])', re.IGNORECASE | re.DOTALL),
            'reasoning': re.compile(r'(NHẬN ĐỊNH|XÉT THẤY).*?(?=\n\n|\n[A-Z])', re.IGNORECASE | re.DOTALL),
            'sentence': re.compile(r'(XỬ PHẠT|HÌNH PHẠT).*?(?=\n\n|\n[A-Z])', re.IGNORECASE | re.DOTALL),
        }
        
        for section_type, pattern in important_patterns.items():
            matches = pattern.finditer(text)
            for match in matches:
                key_sections.append({
                    'type': section_type,
                    'content': match.group(0).strip(),
                    'start_position': match.start(),
                    'end_position': match.end(),
                    'length': len(match.group(0))
                })
        
        return sorted(key_sections, key=lambda x: x['start_position'])
    
    def extract_legal_citations(self, text: str) -> List[str]:
        """
        Extract legal citations and references.
        
        Args:
            text (str): Legal document text
            
        Returns:
            List[str]: Legal citations found
        """
        citations = []
        
        # Patterns for different types of legal citations
        citation_patterns = [
            r'Điều\s+\d+(?:\s+khoản\s+\d+)?(?:\s+điểm\s+[a-z])?',
            r'Bộ luật.*?\d{4}',
            r'Luật.*?\d{4}',
            r'Nghị định.*?\d+/\d{4}',
            r'Thông tư.*?\d+/\d{4}',
            r'Quyết định.*?\d+/\d{4}',
        ]
        
        for pattern in citation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                citations.append(match.group(0))
        
        return list(set(citations))  # Remove duplicates
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from legal text.
        
        Args:
            text (str): Legal document text
            
        Returns:
            Dict[str, List[str]]: Extracted entities by category
        """
        entities = {
            'persons': [],
            'organizations': [],
            'locations': [],
            'dates': [],
            'amounts': [],
            'case_numbers': []
        }
        
        # Person names (Vietnamese pattern)
        person_pattern = r'\b(?:Ông|Bà|Anh|Chị)\s+([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]*(?:\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][a-zàáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]*)*)'
        entities['persons'] = [match.group(1) for match in re.finditer(person_pattern, text)]
        
        # Organizations
        org_pattern = r'\b(?:Công ty|Tòa án|Viện|Cơ quan|Ủy ban|Hội đồng)\s+[A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][^.]*?(?=\s*[.;,\n])'
        entities['organizations'] = [match.group(0) for match in re.finditer(org_pattern, text)]
        
        # Dates
        date_pattern = r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'
        entities['dates'] = [match.group(0) for match in re.finditer(date_pattern, text)]
        
        # Monetary amounts
        amount_pattern = r'\d+(?:\.\d+)*\s*(?:đồng|VNĐ|USD)'
        entities['amounts'] = [match.group(0) for match in re.finditer(amount_pattern, text, re.IGNORECASE)]
        
        # Case numbers
        case_pattern = r'\d+/\d{4}/[A-Z\-]+'
        entities['case_numbers'] = [match.group(0) for match in re.finditer(case_pattern, text)]
        
        # Remove duplicates and empty entries
        for key in entities:
            entities[key] = list(set([item.strip() for item in entities[key] if item.strip()]))
        
        return entities
    
    def analyze_text(self, text: str) -> LegalTextAnalysis:
        """
        Perform comprehensive analysis of legal text.
        
        Args:
            text (str): Legal document text
            
        Returns:
            LegalTextAnalysis: Comprehensive analysis results
        """
        return LegalTextAnalysis(
            document_structure=self.analyze_document_structure(text),
            legal_terminology=self.analyze_legal_terminology(text),
            complexity_metrics=self.calculate_complexity_metrics(text),
            readability_score=self.calculate_readability_score(text),
            key_sections=self.extract_key_sections(text),
            legal_citations=self.extract_legal_citations(text),
            entities=self.extract_entities(text)
        )