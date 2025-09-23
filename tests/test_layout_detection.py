#!/usr/bin/env python3
"""
Test script for layout detection functionality.
This script tests the ability to detect document layout and structure.
"""

import os
import sys
import json
import tempfile
import fitz  # PyMuPDF

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.pdf_processing.layout_detector import LayoutDetector


def convert_pdf_page_to_image(pdf_path: str, page_num: int, output_path: str) -> str:
    """
    Convert a specific PDF page to image for layout detection.
    
    Args:
        pdf_path (str): Path to PDF file
        page_num (int): Page number (0-indexed)
        output_path (str): Output image path
        
    Returns:
        str: Path to the created image
    """
    doc = fitz.open(pdf_path)
    page = doc[page_num]
    
    # Convert page to image with high resolution for better layout detection
    mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
    pix = page.get_pixmap(matrix=mat)
    
    # Save as PNG
    pix.save(output_path)
    doc.close()
    
    return output_path


def test_layout_detection():
    """Test layout detection functionality"""
    print("=" * 80)
    print("TESTING LAYOUT DETECTION")
    print("=" * 80)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"   ✗ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Create temporary directory for test images
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Using temporary directory: {temp_dir}")
            
            # Initialize layout detector
            print("\n1. Initializing layout detector...")
            layout_detector = LayoutDetector()
            print("   ✓ Layout detector initialized")
            
            # Test with first page
            print("\n2. Converting first page to image...")
            image_path = os.path.join(temp_dir, "test_page_1.png")
            convert_pdf_page_to_image(pdf_path, 0, image_path)
            print(f"   ✓ Page 1 converted to: {os.path.basename(image_path)}")
            
            # Detect layout
            print("\n3. Detecting layout elements...")
            layout_data = layout_detector.detect_layout_from_image(image_path)
            
            # Print layout analysis results
            print("   ✓ Layout detection completed")
            print(f"   Tables detected: {len(layout_data.get('tables', []))}")
            print(f"   Text blocks detected: {len(layout_data.get('text_blocks', []))}")
            print(f"   Headers detected: {len(layout_data.get('headers', []))}")
            print(f"   Footers detected: {len(layout_data.get('footers', []))}")
            print(f"   Horizontal lines: {len(layout_data.get('horizontal_lines', []))}")
            print(f"   Vertical lines: {len(layout_data.get('vertical_lines', []))}")
            
            # Analyze document structure
            print("\n4. Analyzing document structure...")
            structure = layout_detector.analyze_document_structure(layout_data)
            print("   ✓ Document structure analysis completed")
            print(f"   Document type: {structure.get('document_type', 'Unknown')}")
            print(f"   Sections identified: {len(structure.get('sections', []))}")
            
            # Show section details
            if structure.get('sections'):
                print("\n   Section details:")
                for i, section in enumerate(structure['sections'][:3]):  # Show first 3 sections
                    print(f"      Section {i+1}: {section.get('type', 'Unknown')} "
                          f"({section.get('block_count', 0)} blocks)")
            
            # Test with multiple pages
            print("\n5. Testing with multiple pages...")
            page_results = []
            
            # Test first 3 pages
            doc = fitz.open(pdf_path)
            total_pages = min(len(doc), 3)
            doc.close()
            
            for page_num in range(total_pages):
                print(f"   Processing page {page_num + 1}...")
                
                page_image_path = os.path.join(temp_dir, f"test_page_{page_num + 1}.png")
                convert_pdf_page_to_image(pdf_path, page_num, page_image_path)
                
                page_layout = layout_detector.detect_layout_from_image(page_image_path)
                page_structure = layout_detector.analyze_document_structure(page_layout)
                
                page_results.append({
                    'page': page_num + 1,
                    'tables': len(page_layout.get('tables', [])),
                    'text_blocks': len(page_layout.get('text_blocks', [])),
                    'document_type': page_structure.get('document_type', 'Unknown')
                })
                
                print(f"      ✓ Page {page_num + 1}: {page_results[-1]['text_blocks']} text blocks, "
                      f"{page_results[-1]['tables']} tables")
            
            # Summary
            print("\n6. Multi-page analysis summary:")
            total_text_blocks = sum(r['text_blocks'] for r in page_results)
            total_tables = sum(r['tables'] for r in page_results)
            print(f"   ✓ Total text blocks across {len(page_results)} pages: {total_text_blocks}")
            print(f"   ✓ Total tables across {len(page_results)} pages: {total_tables}")
            
            # Document type consistency
            doc_types = [r['document_type'] for r in page_results]
            unique_types = set(doc_types)
            print(f"   ✓ Document types detected: {', '.join(unique_types)}")
            
            return True
            
    except Exception as e:
        print(f"   ✗ Layout detection test failed: {str(e)}")
        return False


def test_layout_analysis_accuracy():
    """Test layout analysis accuracy with detailed inspection"""
    print("\n" + "=" * 80)
    print("TESTING LAYOUT ANALYSIS ACCURACY")
    print("=" * 80)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Initialize layout detector
            layout_detector = LayoutDetector()
            
            # Convert and analyze first page in detail
            print("1. Detailed analysis of first page...")
            image_path = os.path.join(temp_dir, "detailed_test_page.png")
            convert_pdf_page_to_image(pdf_path, 0, image_path)
            
            # Get detailed layout data
            layout_data = layout_detector.detect_layout_from_image(image_path)
            
            # Analyze text blocks in detail
            text_blocks = layout_data.get('text_blocks', [])
            if text_blocks:
                print(f"\n   Text blocks analysis ({len(text_blocks)} blocks):")
                
                # Sort blocks by vertical position (top to bottom)
                sorted_blocks = sorted(text_blocks, key=lambda x: x.get('y', 0))
                
                for i, block in enumerate(sorted_blocks[:5]):  # Show first 5 blocks
                    x, y, w, h = block.get('x', 0), block.get('y', 0), block.get('width', 0), block.get('height', 0)
                    area = w * h
                    print(f"      Block {i+1}: Position({x}, {y}), Size({w}x{h}), Area: {area}")
            
            # Analyze tables in detail
            tables = layout_data.get('tables', [])
            if tables:
                print(f"\n   Tables analysis ({len(tables)} tables):")
                for i, table in enumerate(tables):
                    x, y, w, h = table.get('x', 0), table.get('y', 0), table.get('width', 0), table.get('height', 0)
                    rows = table.get('rows', 0)
                    cols = table.get('cols', 0)
                    print(f"      Table {i+1}: Position({x}, {y}), Size({w}x{h}), Grid({rows}x{cols})")
            
            # Analyze headers and footers
            headers = layout_data.get('headers', [])
            footers = layout_data.get('footers', [])
            
            if headers:
                print(f"\n   Headers analysis ({len(headers)} headers):")
                for i, header in enumerate(headers):
                    x, y, w, h = header.get('x', 0), header.get('y', 0), header.get('width', 0), header.get('height', 0)
                    print(f"      Header {i+1}: Position({x}, {y}), Size({w}x{h})")
            
            if footers:
                print(f"\n   Footers analysis ({len(footers)} footers):")
                for i, footer in enumerate(footers):
                    x, y, w, h = footer.get('x', 0), footer.get('y', 0), footer.get('width', 0), footer.get('height', 0)
                    print(f"      Footer {i+1}: Position({x}, {y}), Size({w}x{h})")
            
            # Save detailed analysis
            output_file = os.path.join(temp_dir, "layout_analysis.json")
            layout_detector.save_layout_analysis(layout_data, output_file)
            print(f"\n   ✓ Detailed analysis saved to: {os.path.basename(output_file)}")
            
            return True
            
    except Exception as e:
        print(f"   ✗ Layout analysis accuracy test failed: {str(e)}")
        return False


def test_layout_performance():
    """Test layout detection performance"""
    print("\n" + "=" * 80)
    print("TESTING LAYOUT DETECTION PERFORMANCE")
    print("=" * 80)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    try:
        import time
        
        with tempfile.TemporaryDirectory() as temp_dir:
            layout_detector = LayoutDetector()
            
            # Test performance with different image sizes
            zoom_levels = [1.0, 1.5, 2.0]
            
            print("Testing layout detection performance at different resolutions:")
            
            for zoom in zoom_levels:
                print(f"\n   Testing zoom level {zoom}x...")
                
                # Convert page with specific zoom
                doc = fitz.open(pdf_path)
                page = doc[0]
                mat = fitz.Matrix(zoom, zoom)
                pix = page.get_pixmap(matrix=mat)
                
                image_path = os.path.join(temp_dir, f"perf_test_zoom_{zoom}.png")
                pix.save(image_path)
                doc.close()
                
                # Measure layout detection performance
                start_time = time.time()
                layout_data = layout_detector.detect_layout_from_image(image_path)
                processing_time = time.time() - start_time
                
                # Count detected elements
                text_blocks = len(layout_data.get('text_blocks', []))
                tables = len(layout_data.get('tables', []))
                total_elements = text_blocks + tables
                
                print(f"      Image size: {pix.width}x{pix.height}")
                print(f"      Processing time: {processing_time:.2f} seconds")
                print(f"      Elements detected: {total_elements} (blocks: {text_blocks}, tables: {tables})")
                print(f"      Performance: {total_elements/processing_time:.1f} elements/second")
            
            return True
            
    except Exception as e:
        print(f"   ✗ Layout performance test failed: {str(e)}")
        return False


def main():
    """Main test function"""
    print("LAYOUT DETECTION TEST SUITE")
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
        ("Layout Detection", test_layout_detection),
        ("Layout Analysis Accuracy", test_layout_analysis_accuracy),
        ("Layout Performance", test_layout_performance),
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