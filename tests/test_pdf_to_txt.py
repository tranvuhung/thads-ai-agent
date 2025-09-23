#!/usr/bin/env python3
"""
Script test và xuất nội dung PDF ra file txt
Hỗ trợ cả PDF thông thường và PDF scan (OCR)
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Thêm src vào Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.pdf_processing.pdf_extractor import PDFExtractor
from utils.pdf_processing.text_normalizer import VietnameseTextNormalizer
from utils.pdf_processing.ocr_processor import OCRProcessor
from utils.pdf_processing.document_processor import DocumentProcessor

def create_output_directory():
    """Tạo thư mục output nếu chưa có"""
    output_dir = Path("docs/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def extract_pdf_content_standard(pdf_path):
    """Trích xuất nội dung PDF bằng phương pháp thông thường"""
    print(f"📄 Đang trích xuất PDF thông thường: {pdf_path}")
    
    try:
        with PDFExtractor(pdf_path) as extractor:
            # Trích xuất text
            text = extractor.extract_text()
            
            # Trích xuất metadata
            metadata = extractor.extract_text_with_metadata()
            
            # Trích xuất text theo trang
            pages_text = []
            for page_num in range(len(extractor.doc)):
                page_text = extractor.extract_text_by_page(page_num)
                pages_text.append({
                    'page': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text)
                })
            
            return {
                'method': 'standard',
                'success': True,
                'total_text': text,
                'total_chars': len(text),
                'metadata': metadata,
                'pages': pages_text,
                'page_count': len(pages_text)
            }
            
    except Exception as e:
        return {
            'method': 'standard',
            'success': False,
            'error': str(e),
            'total_chars': 0
        }

def extract_pdf_content_ocr(pdf_path):
    """Trích xuất nội dung PDF bằng OCR"""
    print(f"🔍 Đang trích xuất PDF bằng OCR: {pdf_path}")
    
    try:
        # Khởi tạo OCR processor
        ocr = OCRProcessor(
            tesseract_cmd='/opt/homebrew/bin/tesseract',
            lang='vie+eng'
        )
        
        # Convert PDF pages to images và OCR
        import fitz  # PyMuPDF
        import tempfile
        import cv2
        import numpy as np
        
        doc = fitz.open(pdf_path)
        
        all_text = ""
        pages_text = []
        
        with tempfile.TemporaryDirectory() as temp_dir:
            for page_num in range(len(doc)):
                print(f"   Đang xử lý trang {page_num + 1}/{len(doc)}...")
                
                # Convert page to image
                page = doc[page_num]
                mat = fitz.Matrix(2.0, 2.0)  # Zoom 2x for better OCR
                pix = page.get_pixmap(matrix=mat)
                
                # Save to temporary file
                temp_image_path = os.path.join(temp_dir, f"page_{page_num + 1}.png")
                pix.save(temp_image_path)
                
                # Read image with OpenCV
                img = cv2.imread(temp_image_path)
                if img is None:
                    print(f"      ⚠️ Không thể đọc ảnh trang {page_num + 1}")
                    continue
                
                # OCR
                start_time = time.time()
                page_text = ocr.extract_text_from_image(temp_image_path)
                ocr_time = time.time() - start_time
                
                all_text += page_text + "\n\n"
                pages_text.append({
                    'page': page_num + 1,
                    'text': page_text,
                    'char_count': len(page_text),
                    'ocr_time': round(ocr_time, 2)
                })
                
                print(f"      ✓ Trang {page_num + 1}: {len(page_text)} ký tự ({ocr_time:.2f}s)")
        
        doc.close()
        
        return {
            'method': 'ocr',
            'success': True,
            'total_text': all_text.strip(),
            'total_chars': len(all_text.strip()),
            'pages': pages_text,
            'page_count': len(pages_text)
        }
        
    except Exception as e:
        return {
            'method': 'ocr',
            'success': False,
            'error': str(e),
            'total_chars': 0
        }

def normalize_text(text):
    """Chuẩn hóa text tiếng Việt"""
    try:
        normalizer = VietnameseTextNormalizer()
        normalized = normalizer.clean_and_normalize(text, expand_abbreviations=True, is_legal_document=True)
        return normalized
    except Exception as e:
        print(f"⚠️ Lỗi chuẩn hóa text: {e}")
        return text

def save_to_txt_file(content, filename, output_dir):
    """Lưu nội dung ra file txt"""
    filepath = output_dir / filename
    
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        file_size = filepath.stat().st_size
        print(f"✅ Đã lưu: {filepath} ({file_size:,} bytes)")
        return str(filepath)
        
    except Exception as e:
        print(f"❌ Lỗi lưu file {filepath}: {e}")
        return None

def format_extraction_report(result, normalized_text=None):
    """Tạo báo cáo chi tiết về quá trình trích xuất"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
================================================================================
BÁO CÁO TRÍCH XUẤT PDF
================================================================================
Thời gian: {timestamp}
Phương pháp: {result['method'].upper()}
Trạng thái: {'THÀNH CÔNG' if result['success'] else 'THẤT BẠI'}

"""
    
    if result['success']:
        report += f"""THỐNG KÊ TỔNG QUAN:
- Tổng số ký tự: {result['total_chars']:,}
- Số trang: {result['page_count']}
- Trung bình ký tự/trang: {result['total_chars'] // result['page_count']:,}

CHI TIẾT THEO TRANG:
"""
        
        for page_info in result['pages']:
            report += f"- Trang {page_info['page']}: {page_info['char_count']:,} ký tự"
            if 'ocr_time' in page_info:
                report += f" (OCR: {page_info['ocr_time']}s)"
            report += "\n"
        
        if normalized_text:
            report += f"\nSAU CHUẨN HÓA:\n- Số ký tự: {len(normalized_text):,}\n"
        
        report += f"""
================================================================================
NỘI DUNG TRÍCH XUẤT:
================================================================================

{result['total_text']}

"""
        
        if normalized_text and normalized_text != result['total_text']:
            report += f"""
================================================================================
NỘI DUNG SAU CHUẨN HÓA:
================================================================================

{normalized_text}

"""
    else:
        report += f"LỖI: {result['error']}\n"
    
    report += "================================================================================\n"
    return report

def main():
    """Hàm chính"""
    print("🚀 BẮT ĐẦU TEST VÀ XUẤT NỘI DUNG PDF RA FILE TXT")
    print("=" * 80)
    
    # Tạo thư mục output
    output_dir = create_output_directory()
    print(f"📁 Thư mục output: {output_dir}")
    
    # File PDF để test
    pdf_file = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    if not os.path.exists(pdf_file):
        print(f"❌ Không tìm thấy file PDF: {pdf_file}")
        return
    
    file_size = os.path.getsize(pdf_file) / 1024  # KB
    print(f"📄 File PDF: {pdf_file} ({file_size:.1f} KB)")
    print()
    
    # Test 1: PDF extraction thông thường
    print("🔄 TEST 1: PDF EXTRACTION THÔNG THƯỜNG")
    print("-" * 50)
    
    standard_result = extract_pdf_content_standard(pdf_file)
    
    if standard_result['success'] and standard_result['total_chars'] > 100:
        print(f"✅ PDF extraction thành công: {standard_result['total_chars']:,} ký tự")
        
        # Chuẩn hóa text
        normalized_text = normalize_text(standard_result['total_text'])
        
        # Tạo báo cáo và lưu file
        report = format_extraction_report(standard_result, normalized_text)
        
        # Lưu file txt gốc
        save_to_txt_file(
            standard_result['total_text'],
            "pdf_content_standard.txt",
            output_dir
        )
        
        # Lưu file txt đã chuẩn hóa
        save_to_txt_file(
            normalized_text,
            "pdf_content_standard_normalized.txt",
            output_dir
        )
        
        # Lưu báo cáo
        save_to_txt_file(
            report,
            "pdf_extraction_report_standard.txt",
            output_dir
        )
        
    else:
        print(f"⚠️ PDF extraction ít nội dung: {standard_result['total_chars']} ký tự")
        print("   → Chuyển sang OCR...")
        
        # Test 2: OCR extraction
        print("\n🔄 TEST 2: OCR EXTRACTION")
        print("-" * 50)
        
        ocr_result = extract_pdf_content_ocr(pdf_file)
        
        if ocr_result['success']:
            print(f"✅ OCR extraction thành công: {ocr_result['total_chars']:,} ký tự")
            
            # Chuẩn hóa text
            normalized_text = normalize_text(ocr_result['total_text'])
            
            # Tạo báo cáo và lưu file
            report = format_extraction_report(ocr_result, normalized_text)
            
            # Lưu file txt gốc
            save_to_txt_file(
                ocr_result['total_text'],
                "pdf_content_ocr.txt",
                output_dir
            )
            
            # Lưu file txt đã chuẩn hóa
            save_to_txt_file(
                normalized_text,
                "pdf_content_ocr_normalized.txt",
                output_dir
            )
            
            # Lưu báo cáo
            save_to_txt_file(
                report,
                "pdf_extraction_report_ocr.txt",
                output_dir
            )
            
        else:
            print(f"❌ OCR extraction thất bại: {ocr_result.get('error', 'Unknown error')}")
    
    print("\n🎉 HOÀN THÀNH!")
    print(f"📁 Kiểm tra các file output trong: {output_dir}")

if __name__ == "__main__":
    main()