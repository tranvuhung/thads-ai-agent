#!/usr/bin/env python3
"""
Test script for integrated PDF OCR processing.
This script tests the ability to process scanned PDFs using OCR.
"""

import os
import sys
import json
import time
import tempfile
import fitz  # PyMuPDF

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.pdf_processing.pdf_extractor import PDFExtractor
from utils.pdf_processing.ocr_processor import OCRProcessor
from utils.pdf_processing.text_normalizer import VietnameseTextNormalizer


def convert_pdf_to_images(pdf_path: str, output_dir: str) -> list:
    """
    Convert PDF pages to images for OCR processing.
    
    Args:
        pdf_path (str): Path to PDF file
        output_dir (str): Directory to save images
        
    Returns:
        list: List of image file paths
    """
    print(f"Converting PDF to images: {pdf_path}")
    
    doc = fitz.open(pdf_path)
    image_paths = []
    
    for page_num in range(len(doc)):
        page = doc[page_num]
        
        # Convert page to image
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better quality
        pix = page.get_pixmap(matrix=mat)
        
        # Save as PNG
        image_path = os.path.join(output_dir, f"page_{page_num + 1}.png")
        pix.save(image_path)
        image_paths.append(image_path)
        
        print(f"   ✓ Page {page_num + 1} saved as {os.path.basename(image_path)}")
    
    doc.close()
    return image_paths


def test_pdf_ocr_integration():
    """Test integrated PDF OCR processing"""
    print("=" * 80)
    print("TESTING INTEGRATED PDF OCR PROCESSING")
    print("=" * 80)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"   ✗ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Create temporary directory for images
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Step 1: Try direct text extraction first
            print("\n1. Testing direct text extraction...")
            extractor = PDFExtractor(pdf_path)
            direct_text = extractor.extract_text()
            print(f"   Direct text length: {len(direct_text)} characters")
            
            if len(direct_text.strip()) > 50:
                print("   ✓ PDF contains extractable text, no OCR needed")
                extractor.close()
                return True
            else:
                print("   ⚠ PDF appears to be scanned, proceeding with OCR...")
            
            # Step 2: Convert PDF to images
            print("\n2. Converting PDF pages to images...")
            image_paths = convert_pdf_to_images(pdf_path, temp_dir)
            print(f"   ✓ Converted {len(image_paths)} pages to images")
            
            # Step 3: Initialize OCR processor
            print("\n3. Initializing OCR processor...")
            ocr_processor = OCRProcessor(lang="vie")
            print("   ✓ OCR processor initialized")
            
            # Step 4: Process each page with OCR
            print("\n4. Processing pages with OCR...")
            all_ocr_text = []
            total_chars = 0
            
            for i, image_path in enumerate(image_paths):
                print(f"   Processing page {i + 1}...")
                
                try:
                    # Extract text with OCR
                    page_text = ocr_processor.extract_text_from_image(image_path, preprocess=True)
                    page_chars = len(page_text.strip())
                    total_chars += page_chars
                    
                    all_ocr_text.append(page_text)
                    print(f"      ✓ Extracted {page_chars} characters")
                    
                    # Show sample text from first page
                    if i == 0 and page_chars > 0:
                        sample_text = page_text.strip()[:200]
                        print(f"      Sample: {sample_text}...")
                        
                except Exception as e:
                    print(f"      ✗ OCR failed for page {i + 1}: {str(e)}")
                    all_ocr_text.append("")
            
            # Step 5: Combine and normalize text
            print(f"\n5. Text processing results...")
            combined_text = "\n\n".join(all_ocr_text)
            print(f"   ✓ Total OCR text: {total_chars} characters")
            
            if total_chars > 0:
                # Test text normalization
                print("\n6. Testing text normalization...")
                normalizer = VietnameseTextNormalizer()
                normalized_text = normalizer.clean_and_normalize_legal_document(combined_text)
                print(f"   ✓ Normalized text: {len(normalized_text)} characters")
                
                # Show comparison
                if len(combined_text.strip()) > 100:
                    print("\n   Sample comparison:")
                    original_sample = combined_text.strip()[:200]
                    normalized_sample = normalized_text.strip()[:200]
                    print(f"   Original: {original_sample}...")
                    print(f"   Normalized: {normalized_sample}...")
            
            extractor.close()
            
            return total_chars > 0
            
    except Exception as e:
        print(f"   ✗ Integration test failed: {str(e)}")
        return False


def test_ocr_performance():
    """Test OCR performance and accuracy"""
    print("\n" + "=" * 80)
    print("TESTING OCR PERFORMANCE")
    print("=" * 80)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Convert first page only for performance test
            doc = fitz.open(pdf_path)
            page = doc[0]
            
            # Test different zoom levels
            zoom_levels = [1.0, 1.5, 2.0, 3.0]
            ocr_processor = OCRProcessor(lang="vie")
            
            print("Testing different image qualities:")
            
            for zoom in zoom_levels:
                print(f"\n   Testing zoom level {zoom}x...")
                
                # Convert page to image with specific zoom
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                image_path = os.path.join(temp_dir, f"test_page_zoom_{zoom}.png")
                pix.save(image_path)
                
                # Measure OCR performance
                start_time = time.time()
                text = ocr_processor.extract_text_from_image(image_path, preprocess=True)
                processing_time = time.time() - start_time
                
                print(f"      Image size: {pix.width}x{pix.height}")
                print(f"      Processing time: {processing_time:.2f} seconds")
                print(f"      Extracted characters: {len(text.strip())}")
                
                if len(text.strip()) > 50:
                    sample = text.strip()[:100].replace('\n', ' ')
                    print(f"      Sample: {sample}...")
            
            doc.close()
            return True
            
    except Exception as e:
        print(f"   ✗ Performance test failed: {str(e)}")
        return False


def main():
    """Main test function"""
    print("PDF OCR INTEGRATION TEST SUITE")
    print("Testing with file: docs/pdf_input/BA 311.2025.HS.ST.pdf")
    print("=" * 80)
    
    # Check if PDF exists
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    if not os.path.exists(pdf_path):
        print(f"✗ PDF file not found: {pdf_path}")
        return
    
    file_size = os.path.getsize(pdf_path) / 1024
    print(f"✓ PDF file found: {pdf_path}")
    print(f"✓ File size: {file_size:.1f} KB")
    
    # Run tests
    tests = [
        ("PDF OCR Integration", test_pdf_ocr_integration),
        ("OCR Performance", test_ocr_performance),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"RUNNING: {test_name}")
        print(f"{'='*80}")
        
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✓ PASSED" if result else "✗ FAILED"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results.append((test_name, False))
            print(f"\n{test_name}: ✗ FAILED - {str(e)}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name:<40} {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed!")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()