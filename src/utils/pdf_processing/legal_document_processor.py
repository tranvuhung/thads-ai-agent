"""
Module for processing legal documents with specialized features.
Handles court decisions, legal judgments, and other legal documents.
"""
import os
import re
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass

from .document_processor import DocumentProcessor
from .text_normalizer import VietnameseTextNormalizer


@dataclass
class LegalDocumentInfo:
    """Structured information extracted from legal documents"""
    document_type: str
    case_number: str
    court_name: str
    date_issued: Optional[str]
    parties: Dict[str, List[str]]
    charges: List[str]
    verdict: Optional[str]
    sentence: Optional[str]
    legal_references: List[str]
    key_facts: List[str]


class LegalDocumentProcessor(DocumentProcessor):
    """
    Specialized processor for legal documents with enhanced features
    for Vietnamese court decisions and legal texts.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = "vie"):
        """
        Initialize the LegalDocumentProcessor.
        
        Args:
            tesseract_cmd (str, optional): Path to tesseract executable
            lang (str): Language for OCR (default: 'vie' for Vietnamese)
        """
        super().__init__(tesseract_cmd, lang)
        self.legal_patterns = self._compile_legal_patterns()
        
    def _compile_legal_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for legal document parsing"""
        return {
            # Document identification
            'case_number': re.compile(r'(?:Bản án số|Quyết định số|Số)\s*:?\s*(\d+/\d{4}/[A-Z\-]+)', re.IGNORECASE),
            'court_name': re.compile(r'(TÒA ÁN.*?)(?=\n|$)', re.IGNORECASE),
            'date_issued': re.compile(r'Ngày\s*:?\s*(\d{1,2}/\d{1,2}/\d{4})', re.IGNORECASE),
            
            # Parties involved
            'defendant': re.compile(r'(?:Bị cáo|Bị đơn)\s*:?\s*(.*?)(?=\n|;)', re.IGNORECASE),
            'plaintiff': re.compile(r'(?:Nguyên đơn|Đơn vị khởi tố)\s*:?\s*(.*?)(?=\n|;)', re.IGNORECASE),
            'prosecutor': re.compile(r'(?:Kiểm sát viên|Đại diện VKS)\s*:?\s*(.*?)(?=\n|;)', re.IGNORECASE),
            'judge': re.compile(r'(?:Thẩm phán|Chủ tọa)\s*:?\s*(.*?)(?=\n|;)', re.IGNORECASE),
            
            # Legal charges and crimes
            'charges': re.compile(r'(?:về tội|phạm tội)\s*["\']?(.*?)["\']?(?=\s*theo|$)', re.IGNORECASE),
            'legal_article': re.compile(r'(?:Điều|điều)\s*(\d+)(?:\s*khoản\s*(\d+))?(?:\s*điểm\s*([a-z]))?', re.IGNORECASE),
            'law_reference': re.compile(r'(Bộ luật.*?\d{4})', re.IGNORECASE),
            
            # Verdict and sentence
            'verdict': re.compile(r'(?:Tuyên bố|Quyết định)\s*:?\s*(.*?)(?=\n|$)', re.IGNORECASE),
            'sentence': re.compile(r'(?:Xử phạt|Phạt)\s*.*?(\d+\s*(?:năm|tháng|ngày)\s*tù)', re.IGNORECASE),
            'fine': re.compile(r'(?:Phạt tiền|phạt)\s*(\d+(?:\.\d+)*)\s*(?:đồng|VNĐ)', re.IGNORECASE),
            
            # Document structure
            'section_header': re.compile(r'^[IVX]+\.\s*([A-ZÀÁẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬĐÈÉẺẼẸÊẾỀỂỄỆÌÍỈĨỊÒÓỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÙÚỦŨỤƯỨỪỬỮỰỲÝỶỸỴ][^:]*?)(?=\n|$)', re.MULTILINE),
            'numbered_item': re.compile(r'^\d+\.\s*(.*?)(?=\n|$)', re.MULTILINE),
            
            # Personal information
            'birth_year': re.compile(r'sinh năm\s*(\d{4})', re.IGNORECASE),
            'address': re.compile(r'(?:địa chỉ|nơi thường trú)\s*:?\s*(.*?)(?=\n|;)', re.IGNORECASE),
            'id_number': re.compile(r'(?:CMND|CCCD)\s*:?\s*(\d+)', re.IGNORECASE),
        }
    
    def extract_legal_entities(self, text: str) -> LegalDocumentInfo:
        """
        Extract structured information from legal document text.
        
        Args:
            text (str): Legal document text
            
        Returns:
            LegalDocumentInfo: Structured legal information
        """
        # Determine document type
        doc_type = self._determine_document_type(text)
        
        # Extract basic information
        case_number = self._extract_case_number(text)
        court_name = self._extract_court_name(text)
        date_issued = self._extract_date_issued(text)
        
        # Extract parties
        parties = self._extract_parties(text)
        
        # Extract charges and legal references
        charges = self._extract_charges(text)
        legal_references = self._extract_legal_references(text)
        
        # Extract verdict and sentence
        verdict = self._extract_verdict(text)
        sentence = self._extract_sentence(text)
        
        # Extract key facts
        key_facts = self._extract_key_facts(text)
        
        return LegalDocumentInfo(
            document_type=doc_type,
            case_number=case_number,
            court_name=court_name,
            date_issued=date_issued,
            parties=parties,
            charges=charges,
            verdict=verdict,
            sentence=sentence,
            legal_references=legal_references,
            key_facts=key_facts
        )
    
    def _determine_document_type(self, text: str) -> str:
        """Determine the type of legal document"""
        text_lower = text.lower()
        
        if 'bản án' in text_lower:
            if 'hình sự' in text_lower:
                return 'Bản án hình sự'
            elif 'dân sự' in text_lower:
                return 'Bản án dân sự'
            elif 'hành chính' in text_lower:
                return 'Bản án hành chính'
            else:
                return 'Bản án'
        elif 'quyết định' in text_lower:
            if 'thi hành án' in text_lower:
                return 'Quyết định thi hành án'
            else:
                return 'Quyết định'
        elif 'cáo trạng' in text_lower:
            return 'Cáo trạng'
        elif 'biên bản' in text_lower:
            return 'Biên bản'
        else:
            return 'Văn bản pháp lý'
    
    def _extract_case_number(self, text: str) -> str:
        """Extract case number from text"""
        match = self.legal_patterns['case_number'].search(text)
        return match.group(1) if match else ""
    
    def _extract_court_name(self, text: str) -> str:
        """Extract court name from text"""
        match = self.legal_patterns['court_name'].search(text)
        if match:
            return match.group(1).strip()
        return ""
    
    def _extract_date_issued(self, text: str) -> Optional[str]:
        """Extract date issued from text"""
        match = self.legal_patterns['date_issued'].search(text)
        return match.group(1) if match else None
    
    def _extract_parties(self, text: str) -> Dict[str, List[str]]:
        """Extract parties involved in the case"""
        parties = {
            'defendants': [],
            'plaintiffs': [],
            'prosecutors': [],
            'judges': []
        }
        
        # Extract defendants
        for match in self.legal_patterns['defendant'].finditer(text):
            parties['defendants'].append(match.group(1).strip())
        
        # Extract plaintiffs
        for match in self.legal_patterns['plaintiff'].finditer(text):
            parties['plaintiffs'].append(match.group(1).strip())
        
        # Extract prosecutors
        for match in self.legal_patterns['prosecutor'].finditer(text):
            parties['prosecutors'].append(match.group(1).strip())
        
        # Extract judges
        for match in self.legal_patterns['judge'].finditer(text):
            parties['judges'].append(match.group(1).strip())
        
        return parties
    
    def _extract_charges(self, text: str) -> List[str]:
        """Extract criminal charges from text"""
        charges = []
        for match in self.legal_patterns['charges'].finditer(text):
            charge = match.group(1).strip()
            if charge and len(charge) > 5:  # Filter out very short matches
                charges.append(charge)
        return list(set(charges))  # Remove duplicates
    
    def _extract_legal_references(self, text: str) -> List[str]:
        """Extract legal article and law references"""
        references = []
        
        # Extract law references
        for match in self.legal_patterns['law_reference'].finditer(text):
            references.append(match.group(1))
        
        # Extract article references
        for match in self.legal_patterns['legal_article'].finditer(text):
            article = f"Điều {match.group(1)}"
            if match.group(2):
                article += f" khoản {match.group(2)}"
            if match.group(3):
                article += f" điểm {match.group(3)}"
            references.append(article)
        
        return list(set(references))
    
    def _extract_verdict(self, text: str) -> Optional[str]:
        """Extract verdict from text"""
        match = self.legal_patterns['verdict'].search(text)
        return match.group(1).strip() if match else None
    
    def _extract_sentence(self, text: str) -> Optional[str]:
        """Extract sentence information from text"""
        sentence_match = self.legal_patterns['sentence'].search(text)
        fine_match = self.legal_patterns['fine'].search(text)
        
        sentence_parts = []
        if sentence_match:
            sentence_parts.append(sentence_match.group(1))
        if fine_match:
            sentence_parts.append(f"Phạt tiền {fine_match.group(1)} đồng")
        
        return "; ".join(sentence_parts) if sentence_parts else None
    
    def _extract_key_facts(self, text: str) -> List[str]:
        """Extract key facts from the document"""
        facts = []
        
        # Look for numbered items that might be facts
        for match in self.legal_patterns['numbered_item'].finditer(text):
            fact = match.group(1).strip()
            if len(fact) > 20:  # Only include substantial facts
                facts.append(fact)
        
        return facts[:10]  # Limit to top 10 facts
    
    def process_legal_pdf(self, pdf_path: str, output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Process a legal PDF with enhanced extraction and analysis.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str, optional): Directory to save results
            
        Returns:
            Dict: Comprehensive processing results
        """
        # Process with base functionality
        base_result = self.process_pdf(
            pdf_path=pdf_path,
            output_dir=output_dir,
            normalize_text=True,
            expand_abbreviations=True
        )
        
        # Extract legal entities
        normalized_pages = base_result['content']['normalized_pages']
        normalized_text = '\n'.join(normalized_pages) if normalized_pages else ''
        legal_info = self.extract_legal_entities(normalized_text)
        
        # Add legal analysis to results
        base_result['legal_analysis'] = {
            'document_info': legal_info.__dict__,
            'processing_timestamp': datetime.now().isoformat(),
            'confidence_score': self._calculate_confidence_score(legal_info)
        }
        
        # Save enhanced results if output directory provided
        if output_dir:
            legal_output_path = os.path.join(output_dir, "legal_analysis.json")
            with open(legal_output_path, 'w', encoding='utf-8') as f:
                json.dump(base_result['legal_analysis'], f, indent=2, ensure_ascii=False)
        
        return base_result
    
    def _calculate_confidence_score(self, legal_info: LegalDocumentInfo) -> float:
        """Calculate confidence score based on extracted information"""
        score = 0.0
        max_score = 10.0
        
        # Basic information
        if legal_info.case_number:
            score += 2.0
        if legal_info.court_name:
            score += 1.5
        if legal_info.date_issued:
            score += 1.0
        
        # Parties
        if legal_info.parties['defendants']:
            score += 1.5
        if legal_info.parties['judges']:
            score += 1.0
        
        # Legal content
        if legal_info.charges:
            score += 1.5
        if legal_info.legal_references:
            score += 1.0
        if legal_info.verdict:
            score += 0.5
        
        return min(score / max_score, 1.0)
    
    def batch_process_legal_pdfs(self, pdf_paths: List[str], output_dir: str) -> List[Dict[str, Any]]:
        """
        Process multiple legal PDFs in batch.
        
        Args:
            pdf_paths (List[str]): List of PDF file paths
            output_dir (str): Directory to save results
            
        Returns:
            List[Dict]: Results for each processed PDF
        """
        results = []
        
        for i, pdf_path in enumerate(pdf_paths):
            print(f"Processing {i+1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
            
            try:
                # Create subdirectory for each PDF
                pdf_output_dir = os.path.join(output_dir, f"pdf_{i+1}_{os.path.splitext(os.path.basename(pdf_path))[0]}")
                os.makedirs(pdf_output_dir, exist_ok=True)
                
                result = self.process_legal_pdf(pdf_path, pdf_output_dir)
                result['source_file'] = pdf_path
                result['processing_index'] = i + 1
                results.append(result)
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {str(e)}")
                results.append({
                    'source_file': pdf_path,
                    'processing_index': i + 1,
                    'error': str(e),
                    'success': False
                })
        
        # Save batch summary
        summary_path = os.path.join(output_dir, "batch_processing_summary.json")
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_files': len(pdf_paths),
                'successful': len([r for r in results if r.get('success', True)]),
                'failed': len([r for r in results if not r.get('success', True)]),
                'processing_timestamp': datetime.now().isoformat(),
                'results': results
            }, f, indent=2, ensure_ascii=False)
        
        return results