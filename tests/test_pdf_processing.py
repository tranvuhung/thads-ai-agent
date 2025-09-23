#!/usr/bin/env python3
"""
Test script for PDF processing modules
Tests the functionality with the sample PDF file in docs/pdf_input/
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.pdf_processing.document_processor import DocumentProcessor
from utils.pdf_processing.pdf_extractor import PDFExtractor
from utils.pdf_processing.text_normalizer import VietnameseTextNormalizer
from utils.pdf_processing.ocr_processor import OCRProcessor
from utils.pdf_processing.layout_detector import LayoutDetector

def test_pdf_extraction():
    """Test PDF text extraction functionality"""
    print("=" * 60)
    print("TESTING PDF TEXT EXTRACTION")
    print("=" * 60)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    try:
        extractor = PDFExtractor(pdf_path)
        
        # Test basic text extraction
        print("1. Testing basic text extraction...")
        text = extractor.extract_text()
        print(f"   ✓ Extracted {len(text)} characters")
        print(f"   First 200 characters: {text[:200]}...")
        
        # Test page-by-page extraction
        print("\n2. Testing page-by-page extraction...")
        pages_text = extractor.extract_text_by_page()
        print(f"   ✓ Document has {len(pages_text)} pages")
        for i, page_text in enumerate(pages_text[:3]):  # Show first 3 pages
            print(f"   Page {i+1}: {len(page_text)} characters")
        
        # Test metadata extraction
        print("\n3. Testing metadata extraction...")
        metadata = extractor.extract_text_with_metadata()
        print(f"   ✓ Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        
        # Test text with positions
        print("\n4. Testing text extraction with positions...")
        text_with_pos = extractor.extract_text_with_positions()
        print(f"   ✓ Extracted {len(text_with_pos)} text blocks with positions")
        if text_with_pos:
            print(f"   Sample block: {text_with_pos[0]}")
        
        # Close the extractor
        extractor.close()
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False

def test_text_normalization():
    """Test Vietnamese text normalization for legal documents"""
    print("\n" + "=" * 60)
    print("TESTING VIETNAMESE TEXT NORMALIZATION")
    print("=" * 60)
    
    try:
        normalizer = VietnameseTextNormalizer()
        
        # Sample legal text with common issues
        sample_texts = [
            "Theo quy định tại Điều 5 khoản 2 điểm a của Luật Doanh nghiệp số 59/2020/QH14",
            "UBND TP.HCM ban hành QĐ-TTg số 123/2024/QĐ-TTg ngày 15 tháng 3 năm 2024",
            "Căn cứ BLDS, BLHS và các VBQPPL có liên quan",
            "Công ty TNHH MTV ABC ký HĐLĐ với người lao động",
            "Diéu 10 Khoán 1 của Luàt Doanh nghiêp"  # Text with OCR errors
        ]
        
        print("1. Testing basic text normalization...")
        for i, text in enumerate(sample_texts):
            normalized = normalizer.clean_and_normalize(text)
            print(f"   Original {i+1}: {text}")
            print(f"   Normalized: {normalized}")
            print()
        
        print("2. Testing legal document specific normalization...")
        for i, text in enumerate(sample_texts):
            normalized = normalizer.clean_and_normalize_legal_document(text)
            print(f"   Legal {i+1}: {text}")
            print(f"   Normalized: {normalized}")
            print()
        
        print("3. Testing batch normalization...")
        batch_normalized = normalizer.batch_normalize_legal_documents(sample_texts)
        print(f"   ✓ Batch processed {len(batch_normalized)} texts")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False

def test_document_processor():
    """Test integrated document processor"""
    print("\n" + "=" * 60)
    print("TESTING INTEGRATED DOCUMENT PROCESSOR")
    print("=" * 60)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    try:
        processor = DocumentProcessor()
        
        print("1. Testing PDF processing...")
        result = processor.process_pdf(pdf_path)
        
        print(f"   ✓ Processing completed")
        print(f"   Raw text length: {len(result.get('raw_text', ''))}")
        print(f"   Normalized text length: {len(result.get('normalized_text', ''))}")
        print(f"   Number of pages: {result.get('num_pages', 0)}")
        print(f"   Processing time: {result.get('processing_time', 0):.2f} seconds")
        
        # Show sample of normalized text
        normalized_text = result.get('normalized_text', '')
        if normalized_text:
            print(f"\n   Sample normalized text (first 300 chars):")
            print(f"   {normalized_text[:300]}...")
        
        # Show metadata if available
        metadata = result.get('metadata', {})
        if metadata:
            print(f"\n   Document metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        return False

def test_ocr_processor():
    """Test OCR processing functionality"""
    print("\n" + "=" * 60)
    print("TESTING OCR PROCESSOR")
    print("=" * 60)
    
    try:
        ocr = OCRProcessor()
        
        # Test if OCR is available
        print("1. Testing OCR availability...")
        if hasattr(ocr, 'extract_text_from_image'):
            print("   ✓ OCR processor initialized successfully")
            print("   Note: OCR testing requires image files or PDF pages as images")
        else:
            print("   ✗ OCR processor not properly initialized")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        print("   Note: OCR functionality requires pytesseract installation")
        return False

def test_layout_detector():
    """Test layout detection functionality"""
    print("\n" + "=" * 60)
    print("TESTING LAYOUT DETECTOR")
    print("=" * 60)
    
    try:
        detector = LayoutDetector()
        
        print("1. Testing layout detector initialization...")
        if hasattr(detector, 'detect_layout_from_image'):
            print("   ✓ Layout detector initialized successfully")
            print("   Note: Layout detection requires image files for full testing")
        else:
            print("   ✗ Layout detector not properly initialized")
            
        return True
        
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
        print("   Note: Layout detection requires OpenCV installation")
        return False

def main():
    """Run all tests"""
    print("PDF PROCESSING MODULE TEST SUITE")
    print("Testing with file: docs/pdf_input/BA 311.2025.HS.ST.pdf")
    print("=" * 80)
    
    # Check if PDF file exists
    pdf_path = Path("docs/pdf_input/BA 311.2025.HS.ST.pdf")
    if not pdf_path.exists():
        print(f"❌ PDF file not found: {pdf_path}")
        return
    
    print(f"✓ PDF file found: {pdf_path}")
    print(f"✓ File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    
    # Run tests
    tests = [
        ("PDF Extraction", test_pdf_extraction),
        ("Text Normalization", test_text_normalization),
        ("Document Processor", test_document_processor),
        ("OCR Processor", test_ocr_processor),
        ("Layout Detector", test_layout_detector)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ {test_name} failed with error: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:.<50} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()