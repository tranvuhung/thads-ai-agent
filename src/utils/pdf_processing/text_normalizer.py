"""
Module for cleaning and normalizing Vietnamese text.
"""
import re
import unicodedata
from typing import Dict, List, Optional, Tuple, Union


class VietnameseTextNormalizer:
    """
    Class for cleaning and normalizing Vietnamese text.
    """
    
    def __init__(self):
        """
        Initialize the VietnameseTextNormalizer.
        """
        # Common Vietnamese abbreviations and their expansions
        self.abbreviations = {
            # Địa phương
            "TP.": "Thành phố",
            "Tp.": "Thành phố",
            "TPHCM": "Thành phố Hồ Chí Minh",
            "TP.HCM": "Thành phố Hồ Chí Minh",
            "HN": "Hà Nội",
            "ĐN": "Đà Nẵng",
            "CT": "Cần Thơ",
            "HP": "Hải Phòng",
            "Q.": "Quận",
            "P.": "Phường",
            "H.": "Huyện",
            "T.": "Tỉnh",
            "TT.": "Thị trấn",
            "TX.": "Thị xã",
            
            # Cơ quan nhà nước
            "UBND": "Ủy ban nhân dân",
            "HĐND": "Hội đồng nhân dân",
            "ĐBQH": "Đại biểu Quốc hội",
            "CQNN": "Cơ quan nhà nước",
            "BTP": "Bộ Tư pháp",
            "BTC": "Bộ Tài chính",
            "BCA": "Bộ Công an",
            "BQP": "Bộ Quốc phòng",
            "BTNMT": "Bộ Tài nguyên và Môi trường",
            "BGTVT": "Bộ Giao thông vận tải",
            "BXD": "Bộ Xây dựng",
            "BYT": "Bộ Y tế",
            "BGDĐT": "Bộ Giáo dục và Đào tạo",
            "BLĐTBXH": "Bộ Lao động - Thương binh và Xã hội",
            "BKHĐT": "Bộ Kế hoạch và Đầu tư",
            "BKHCN": "Bộ Khoa học và Công nghệ",
            "BVHTTDL": "Bộ Văn hóa, Thể thao và Du lịch",
            "BTTTT": "Bộ Thông tin và Truyền thông",
            "BNG": "Bộ Ngoại giao",
            "BNN&PTNT": "Bộ Nông nghiệp và Phát triển nông thôn",
            "BCT": "Bộ Công Thương",
            
            # Văn bản pháp luật
            "NĐ-CP": "Nghị định Chính phủ",
            "QĐ-TTg": "Quyết định Thủ tướng",
            "TT-BTC": "Thông tư Bộ Tài chính",
            "TTLT": "Thông tư liên tịch",
            "NQ-CP": "Nghị quyết Chính phủ",
            "QH": "Quốc hội",
            "HĐTP": "Hội đồng Thẩm phán",
            "CT-TTg": "Chỉ thị Thủ tướng",
            "VB": "Văn bản",
            "VBQPPL": "Văn bản quy phạm pháp luật",
            "VBPL": "Văn bản pháp luật",
            "QCVN": "Quy chuẩn Việt Nam",
            "TCVN": "Tiêu chuẩn Việt Nam",
            
            # Giấy tờ cá nhân
            "CMND": "Chứng minh nhân dân",
            "CCCD": "Căn cước công dân",
            "HC": "Hộ chiếu",
            "ĐKKD": "Đăng ký kinh doanh",
            "ĐKDN": "Đăng ký doanh nghiệp",
            "MST": "Mã số thuế",
            "MSDN": "Mã số doanh nghiệp",
            "GPLĐ": "Giấy phép lao động",
            "GPKD": "Giấy phép kinh doanh",
            "GCNQSDĐ": "Giấy chứng nhận quyền sử dụng đất",
            "GCNQSHNƠ": "Giấy chứng nhận quyền sở hữu nhà ở",
            
            # Bảo hiểm và an sinh xã hội
            "BHXH": "Bảo hiểm xã hội",
            "BHYT": "Bảo hiểm y tế",
            "BHTN": "Bảo hiểm thất nghiệp",
            "BHTNLĐ-BNN": "Bảo hiểm tai nạn lao động - bệnh nghề nghiệp",
            "KBCB": "Khám bệnh, chữa bệnh",
            "CSSK": "Chăm sóc sức khỏe",
            
            # Doanh nghiệp
            "CTCP": "Công ty cổ phần",
            "TNHH": "Trách nhiệm hữu hạn",
            "MTV": "Một thành viên",
            "DN": "Doanh nghiệp",
            "DNNN": "Doanh nghiệp nhà nước",
            "DNNVV": "Doanh nghiệp nhỏ và vừa",
            "TCT": "Tổng công ty",
            "CT": "Công ty",
            
            # Hợp đồng
            "HĐLĐ": "Hợp đồng lao động",
            "HĐKT": "Hợp đồng kinh tế",
            "HĐMB": "Hợp đồng mua bán",
            "HĐDV": "Hợp đồng dịch vụ",
            "HĐTD": "Hợp đồng tín dụng",
            "HĐCNQSDĐ": "Hợp đồng chuyển nhượng quyền sử dụng đất",
            "HĐMBN": "Hợp đồng mua bán nhà",
            "HĐGC": "Hợp đồng góp vốn",
            "HĐHTDT": "Hợp đồng hợp tác đầu tư",
            "HĐTC": "Hợp đồng thuê",
            
            # Tài chính, ngân hàng
            "TCTD": "Tổ chức tín dụng",
            "NHTM": "Ngân hàng thương mại",
            "NHNN": "Ngân hàng Nhà nước",
            "TPDN": "Trái phiếu doanh nghiệp",
            "TPCP": "Trái phiếu chính phủ",
            "TTCK": "Thị trường chứng khoán",
            "CTCK": "Công ty chứng khoán",
            "UBCKNN": "Ủy ban Chứng khoán Nhà nước",
            "TTGDCK": "Trung tâm Giao dịch Chứng khoán",
            "SGDCK": "Sở Giao dịch Chứng khoán",
            "KTNN": "Kiểm toán Nhà nước",
            
            # Tòa án, tư pháp
            "TAND": "Tòa án nhân dân",
            "TANDTC": "Tòa án nhân dân tối cao",
            "VKSND": "Viện kiểm sát nhân dân",
            "VKSNDTC": "Viện kiểm sát nhân dân tối cao",
            "CQĐT": "Cơ quan điều tra",
            "CQTHADS": "Cơ quan thi hành án dân sự",
            "THADS": "Thi hành án dân sự",
            "CQCSĐT": "Cơ quan Cảnh sát điều tra",
            "CSĐT": "Cảnh sát điều tra",
            "CA": "Công an",
            "TA": "Tòa án",
            "VKS": "Viện kiểm sát",
            "TP": "Tư pháp",
            "THA": "Thi hành án",
            
            # Bộ luật, luật
            "BLDS": "Bộ luật dân sự",
            "BLHS": "Bộ luật hình sự",
            "BLTTHS": "Bộ luật tố tụng hình sự",
            "BLTTDS": "Bộ luật tố tụng dân sự",
            "BLLĐ": "Bộ luật lao động",
            "LĐĐ": "Luật đất đai",
            "LXD": "Luật xây dựng",
            "LĐT": "Luật đầu tư",
            "LDN": "Luật doanh nghiệp",
            "LTM": "Luật thương mại",
            "LCTD": "Luật các tổ chức tín dụng",
            "LBVMT": "Luật bảo vệ môi trường",
            "LBHXH": "Luật bảo hiểm xã hội",
            "LBHYT": "Luật bảo hiểm y tế",
            "LGTĐB": "Luật giao thông đường bộ",
            "LNVCC": "Luật người viết công chứng",
            "LĐGTS": "Luật đấu giá tài sản",
            "LPCTN": "Luật phòng, chống tham nhũng",
            "LPCRT": "Luật phòng, chống rửa tiền",
            "LNTK": "Luật ngân sách nhà nước",
            "LTT": "Luật thuế thu nhập",
            "LGTGT": "Luật thuế giá trị gia tăng",
            "LTCB": "Luật thuế tiêu thụ đặc biệt",
            "LTTN": "Luật thuế tài nguyên",
            "LTSDĐ": "Luật thuế sử dụng đất"
        }
        
        # Common OCR errors in Vietnamese, with focus on legal documents
        self.ocr_errors = {
            # Basic character confusions
            'l': 'i',
            '0': 'o',
            # '1': 'l',
            '5': 's',
            '8': 'B',
            'rn': 'm',
            'cl': 'd',
            'ii': 'u',
            'nn': 'm',
            'iii': 'm',
            'li': 'h',
            'I-l': 'H',
            'vv': 'w',
            '1': 'I',

            # Vietnamese specific
            'đ': 'đ',   # Normalize đ character
            'Đ': 'Đ',   # Normalize Đ character
            
            # Legal document specific OCR errors
            'Diéu': 'Điều',
            'Diêu': 'Điều',
            'Dieu': 'Điều',
            'Khoán': 'Khoản',
            'Khoàn': 'Khoản',
            'Khoan': 'Khoản',
            'Diêm': 'Điểm',
            'Diém': 'Điểm',
            'Diem': 'Điểm',
            'Chuong': 'Chương',
            'Chưong': 'Chương',
            'Chưcmg': 'Chương',
            'Luàt': 'Luật',
            'Luât': 'Luật',
            'Luat': 'Luật',
            'Nghj': 'Nghị',
            'Nghi': 'Nghị',
            'Quyét': 'Quyết',
            'Quyêt': 'Quyết',
            'Quyet': 'Quyết',
            'Thông': 'Thông',
            'Thóng': 'Thông',
            'Thong': 'Thông',
            'Só': 'Số',
            'Sô': 'Số',
            'So': 'Số',
            'Chinh': 'Chính',
            'Chinh phú': 'Chính phủ',
            'Chinh phù': 'Chính phủ',
            'Chinh phu': 'Chính phủ',
            'Thú tuóng': 'Thủ tướng',
            'Thù tưóng': 'Thủ tướng',
            'Thu tuong': 'Thủ tướng',
            'Bô': 'Bộ',
            'Bo': 'Bộ',
            'Hôi đông': 'Hội đồng',
            'Hôi dông': 'Hội đồng',
            'Hoi dong': 'Hội đồng',
            'Uy ban': 'Ủy ban',
            'Ùy ban': 'Ủy ban',
            'Viên': 'Viện',
            'Vien': 'Viện',
            'Tòa án': 'Tòa án',
            'Toa an': 'Tòa án',
            'Kiêm sát': 'Kiểm sát',
            'Kiém sát': 'Kiểm sát',
            'Kiem sat': 'Kiểm sát',
            'Công an': 'Công an',
            'Cong an': 'Công an',
            'Dân sư': 'Dân sự',
            'Dân su': 'Dân sự',
            'Dan su': 'Dân sự',
            'Hinh sư': 'Hình sự',
            'Hinh su': 'Hình sự',
            'Hinh sự': 'Hình sự',
            'Tô tung': 'Tố tụng',
            'Tó tung': 'Tố tụng',
            'To tung': 'Tố tụng',
            'Thi hành': 'Thi hành',
            'Thi hanh': 'Thi hành',
            'Thuê': 'Thuế',
            'Thue': 'Thuế',
            'toàn': 'toàn',
            'quyết': 'quyết',
            'định': 'định',
            'thi': 'thi',
            'hành': 'hành',
        }

        # Regex patterns
        self.patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns để tái sử dụng"""
        return {
            'extra_whitespace': re.compile(r'\s+'),
            'page_number': re.compile(r'Trang\s*\d+', re.IGNORECASE),
            'header_footer': re.compile(r'^-{3,}.*?-{3,}$', re.MULTILINE),
            'date_pattern': re.compile(r'\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}'),
            'decision_number': re.compile(r'\d+\/\d{4}\/[A-Z\-]+'),
            'court_name': re.compile(r'TOÀN?\s+ÁN.*?(?=\n)', re.IGNORECASE),
            'special_chars': re.compile(r'[^\w\s\.\,\;\:\!\?\-\(\)\/\%\&\'\"\n]'),
            'multiple_newlines': re.compile(r'\n{3,}'),
            'bullet_points': re.compile(r'^[\-\*\+]\s*', re.MULTILINE),
        }
    
    def normalize_unicode(self, text: str) -> str:
        """
        Normalize Unicode characters in Vietnamese text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        # Normalize Unicode to form NFC (Normalization Form C)
        # This is important for Vietnamese text with diacritics
        return unicodedata.normalize('NFC', text)
    
    def remove_extra_spaces(self, text: str) -> str:
        """
        Remove extra spaces, tabs, and newlines.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Cleaned text
        """
        # Replace multiple spaces with a single space
        text = re.sub(r'\s+', ' ', text)
        # Remove spaces at the beginning and end of the text
        text = text.strip()
        return text
    
    def fix_common_ocr_errors(self, text: str) -> str:
        """
        Fix common OCR errors in Vietnamese text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Corrected text
        """
        # Fix common OCR errors
        for error, correction in self.ocr_errors.items():
            # Only replace if the error is a standalone character or part of a word
            text = re.sub(r'\b' + error + r'\b', correction, text)
        
        return text
    
    def expand_abbreviations(self, text: str, expand_all: bool = False) -> str:
        """
        Expand common Vietnamese abbreviations.
        
        Args:
            text (str): Input text
            expand_all (bool): Whether to expand all abbreviations or only those followed by a period
            
        Returns:
            str: Text with expanded abbreviations
        """
        # Expand abbreviations
        for abbr, expansion in self.abbreviations.items():
            if expand_all:
                # Replace all occurrences of the abbreviation
                text = re.sub(r'\b' + re.escape(abbr) + r'\b', expansion, text)
            else:
                # Replace only abbreviations that are followed by a period
                if abbr.endswith('.'):
                    text = re.sub(r'\b' + re.escape(abbr) + r'\s', expansion + ' ', text)
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """
        Normalize punctuation in Vietnamese text, with special handling for legal documents.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized punctuation
        """
        # Ensure space after punctuation marks
        text = re.sub(r'([.,;:!?)])(\S)', r'\1 \2', text)
        
        # Ensure no space before punctuation marks
        text = re.sub(r'(\S)(\s+)([.,;:!?)])', r'\1\3', text)
        
        # Ensure space after opening parenthesis
        text = re.sub(r'([(])(\S)', r'\1 \2', text)
        
        # Ensure no space before opening parenthesis
        text = re.sub(r'(\S)(\s+)([(])', r'\1 \3', text)
        
        # Legal document specific normalization
        # Standardize dashes in legal references
        text = re.sub(r'(?<=\d)\s*[-–—]\s*(?=\d)', '-', text)
        
        # Standardize slashes in legal references
        text = re.sub(r'(?<=\d)\s*[/\\]\s*(?=\d)', '/', text)
        
        # Standardize article symbols
        text = re.sub(r'(?i)điều\s+', 'Điều ', text)
        text = re.sub(r'(?i)khoản\s+', 'Khoản ', text)
        text = re.sub(r'(?i)điểm\s+', 'Điểm ', text)
        text = re.sub(r'(?i)chương\s+', 'Chương ', text)
        text = re.sub(r'(?i)mục\s+', 'Mục ', text)
        text = re.sub(r'(?i)phần\s+', 'Phần ', text)
        text = re.sub(r'(?i)phụ lục\s+', 'Phụ lục ', text)
        
        # Standardize legal document references
        text = re.sub(r'(?i)số\s+', 'số ', text)
        text = re.sub(r'(?i)ngày\s+', 'ngày ', text)
        text = re.sub(r'(?i)tháng\s+', 'tháng ', text)
        text = re.sub(r'(?i)năm\s+', 'năm ', text)
        
        return text
    
    def normalize_numbers(self, text: str) -> str:
        """
        Normalize number formats in Vietnamese text.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with normalized numbers
        """
        # Normalize decimal separators (replace comma with period)
        text = re.sub(r'(\d+),(\d+)', r'\1.\2', text)
        
        # Normalize thousand separators (replace period with comma)
        # This is a bit tricky because we need to distinguish between decimal points and thousand separators
        # For simplicity, we'll assume that a period followed by exactly 3 digits is a thousand separator
        text = re.sub(r'(\d+)\.(\d{3})(?!\d)', r'\1,\2', text)
        
        # Fix spacing around numbers
        text = re.sub(r'(\d+)\s+%', r'\1%', text)
        text = re.sub(r'(\d+)\s+°C', r'\1°C', text)
        
        return text
        
    def normalize_legal_document_numbers(self, text: str) -> str:
        """
        Normalize legal document numbers and references in Vietnamese legal texts.
        
        Args:
            text (str): The input text containing legal document references
            
        Returns:
            str: Text with normalized legal document references
        """
        # Normalize document numbers (e.g., "Số 12/2020/NĐ-CP")
        text = re.sub(r'(?i)số\s+(\d+)\s*/\s*(\d+)\s*/\s*([A-Z0-9Đ-]+)\s*-\s*([A-Z0-9]+)', 
                      r'Số \1/\2/\3-\4', text)
        
        # Normalize law references (e.g., "Luật số 45/2019/QH14")
        text = re.sub(r'(?i)(luật|nghị định|thông tư|quyết định|chỉ thị)\s+số\s+(\d+)\s*/\s*(\d+)\s*/\s*([A-Z0-9Đ-]+)', 
                      lambda m: f"{m.group(1).capitalize()} số {m.group(2)}/{m.group(3)}/{m.group(4)}", text)
        
        # Normalize article references (e.g., "Điều 5 khoản 2 điểm a")
        text = re.sub(r'(?i)(điều)\s+(\d+)\s+(khoản)\s+(\d+)\s+(điểm)\s+([a-z])', 
                      r'Điều \2 Khoản \4 Điểm \6', text)
        
        # Normalize date references in legal documents
        text = re.sub(r'(?i)ngày\s+(\d{1,2})\s+tháng\s+(\d{1,2})\s+năm\s+(\d{4})', 
                      r'ngày \1 tháng \2 năm \3', text)
        
        return text
        
    def normalize_legal_references(self, text: str) -> str:
        """
        Normalize references to other legal documents within Vietnamese legal texts.
        
        Args:
            text (str): The input text containing legal references
            
        Returns:
            str: Text with normalized legal references
        """
        # Normalize references to laws
        patterns = [
            # Pattern for "theo quy định tại Điều X Luật Y"
            (r'(?i)(theo|quy định tại|căn cứ|chiếu|áp dụng)\s+(điều|khoản|điểm)\s+(\d+|[a-z])\s+(của|tại)?\s*(luật|bộ luật|nghị định|thông tư)\s+([^,\.;]+)', 
             lambda m: f"{m.group(1)} {m.group(2).capitalize()} {m.group(3)} {m.group(5).capitalize()} {m.group(6)}"),
            
            # Pattern for "theo Luật X số Y/Z/T"
            (r'(?i)(theo|căn cứ|chiếu)\s+(luật|bộ luật|nghị định|thông tư)\s+([^,\.;]+)\s+số\s+(\d+/\d+/[A-Z0-9-]+)', 
             lambda m: f"{m.group(1)} {m.group(2).capitalize()} {m.group(3)} số {m.group(4)}"),
            
            # Pattern for references to specific articles
            (r'(?i)(điều)\s+(\d+)\s+(luật|bộ luật|nghị định|thông tư)\s+([^,\.;]+)', 
             lambda m: f"{m.group(1).capitalize()} {m.group(2)} {m.group(3).capitalize()} {m.group(4)}")
        ]
        
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
            
        return text
    
    def clean_and_normalize(self, text: str, expand_abbreviations: bool = False, is_legal_document: bool = False) -> str:
        """
        Clean and normalize Vietnamese text.
        
        Args:
            text (str): Input text
            expand_abbreviations (bool): Whether to expand abbreviations
            is_legal_document (bool): Flag to indicate if the text is a legal document
                                     to apply specialized legal document normalization
            
        Returns:
            str: Cleaned and normalized text
        """
        # Apply all normalization steps
        text = self.normalize_unicode(text)
        text = self.remove_extra_spaces(text)
        text = self.fix_common_ocr_errors(text)
        
        if expand_abbreviations:
            text = self.expand_abbreviations(text, expand_all=True)
            
        text = self.normalize_punctuation(text)
        text = self.normalize_numbers(text)
        
        # Apply legal document specific normalization if requested
        if is_legal_document:
            text = self.normalize_legal_document_numbers(text)
            text = self.normalize_legal_references(text)
        
        return text
        
    def clean_and_normalize_legal_document(self, text: str, expand_abbreviations: bool = False) -> str:
        """
        Specialized method for cleaning and normalizing Vietnamese legal documents.
        This is a convenience method that calls clean_and_normalize with is_legal_document=True.
        
        Args:
            text (str): The legal document text to normalize
            expand_abbreviations (bool): Whether to expand abbreviations
            
        Returns:
            str: Cleaned and normalized legal document text
        """
        return self.clean_and_normalize(text, expand_abbreviations=expand_abbreviations, is_legal_document=True)
    
    def batch_normalize(self, texts: List[str], expand_abbreviations: bool = False, is_legal_document: bool = False) -> List[str]:
        """
        Clean and normalize a list of Vietnamese texts.
        
        Args:
            texts (List[str]): List of input texts
            expand_abbreviations (bool): Whether to expand abbreviations
            is_legal_document (bool): Flag to indicate if the texts are legal documents
            
        Returns:
            List[str]: List of cleaned and normalized texts
        """
        return [self.clean_and_normalize(text, expand_abbreviations, is_legal_document) for text in texts]
        
    def batch_normalize_legal_documents(self, texts: List[str], expand_abbreviations: bool = False) -> List[str]:
        """
        Apply legal document normalization to a list of texts.
        
        Args:
            texts (List[str]): List of legal document text strings to normalize
            expand_abbreviations (bool): Whether to expand abbreviations
            
        Returns:
            List[str]: List of normalized legal document text strings
        """
        return self.batch_normalize(texts, expand_abbreviations, is_legal_document=True)