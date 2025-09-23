#!/usr/bin/env python3
"""
Test script for legal document processing modules.
Tests the enhanced functionality for Vietnamese legal documents.
"""

import os
import sys
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.pdf_processing.legal_document_processor import LegalDocumentProcessor
from src.utils.pdf_processing.legal_text_analyzer import LegalTextAnalyzer


def test_legal_document_processor():
    """Test legal document processor functionality"""
    print("=" * 80)
    print("TESTING LEGAL DOCUMENT PROCESSOR")
    print("=" * 80)
    
    pdf_path = "docs/pdf_input/BA 311.2025.HS.ST.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"❌ PDF file not found: {pdf_path}")
        return False
    
    try:
        # Initialize processor
        processor = LegalDocumentProcessor()
        
        # Create output directory
        output_dir = "docs/output/legal_processing_test"
        os.makedirs(output_dir, exist_ok=True)
        
        print("1. Processing legal PDF...")
        result = processor.process_legal_pdf(pdf_path, output_dir)
        
        print(f"   ✓ Processing completed")
        print(f"   Document type: {result['legal_analysis']['document_info']['document_type']}")
        print(f"   Case number: {result['legal_analysis']['document_info']['case_number']}")
        print(f"   Court: {result['legal_analysis']['document_info']['court_name']}")
        print(f"   Confidence score: {result['legal_analysis']['confidence_score']:.2f}")
        
        # Display extracted information
        legal_info = result['legal_analysis']['document_info']
        
        print("\n2. Extracted Legal Information:")
        print(f"   Charges: {legal_info['charges']}")
        print(f"   Defendants: {legal_info['parties']['defendants']}")
        print(f"   Judges: {legal_info['parties']['judges']}")
        print(f"   Verdict: {legal_info['verdict']}")
        print(f"   Sentence: {legal_info['sentence']}")
        print(f"   Legal references: {len(legal_info['legal_references'])} found")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_legal_text_analyzer():
    """Test legal text analyzer functionality"""
    print("\n" + "=" * 80)
    print("TESTING LEGAL TEXT ANALYZER")
    print("=" * 80)
    
    # Read sample text from processed output
    sample_text_path = "docs/output/pdf_content_ocr_normalized.txt"
    
    if not os.path.exists(sample_text_path):
        print(f"❌ Sample text file not found: {sample_text_path}")
        return False
    
    try:
        with open(sample_text_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Initialize analyzer
        analyzer = LegalTextAnalyzer()
        
        print("1. Analyzing document structure...")
        structure = analyzer.analyze_document_structure(text)
        print(f"   ✓ Total lines: {structure['total_lines']}")
        print(f"   ✓ Sections: {len(structure['sections'])}")
        print(f"   ✓ Hierarchy depth: {structure['hierarchy_depth']}")
        print(f"   ✓ Has legal citations: {structure['has_legal_citations']}")
        
        print("\n2. Analyzing legal terminology...")
        terminology = analyzer.analyze_legal_terminology(text)
        for category, terms in terminology.items():
            if terms:
                print(f"   {category}: {len(terms)} terms found")
                # Show top 3 terms
                sorted_terms = sorted(terms.items(), key=lambda x: x[1], reverse=True)[:3]
                for term, count in sorted_terms:
                    print(f"     - {term}: {count}")
        
        print("\n3. Calculating complexity metrics...")
        complexity = analyzer.calculate_complexity_metrics(text)
        print(f"   ✓ Average sentence length: {complexity['average_sentence_length']:.1f} words")
        print(f"   ✓ Average word length: {complexity['average_word_length']:.1f} characters")
        print(f"   ✓ Legal term density: {complexity['legal_term_density']:.3f}")
        print(f"   ✓ Citation density: {complexity['citation_density']:.2f}%")
        
        print("\n4. Calculating readability score...")
        readability = analyzer.calculate_readability_score(text)
        print(f"   ✓ Readability score: {readability:.1f}/100")
        
        print("\n5. Extracting key sections...")
        key_sections = analyzer.extract_key_sections(text)
        print(f"   ✓ Key sections found: {len(key_sections)}")
        for section in key_sections[:3]:  # Show first 3
            print(f"     - {section['type']}: {section['length']} characters")
        
        print("\n6. Extracting legal citations...")
        citations = analyzer.extract_legal_citations(text)
        print(f"   ✓ Legal citations found: {len(citations)}")
        for citation in citations[:5]:  # Show first 5
            print(f"     - {citation}")
        
        print("\n7. Extracting entities...")
        entities = analyzer.extract_entities(text)
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"   {entity_type}: {len(entity_list)} found")
                # Show first 3
                for entity in entity_list[:3]:
                    print(f"     - {entity}")
        
        print("\n8. Performing comprehensive analysis...")
        analysis = analyzer.analyze_text(text)
        
        # Save analysis results
        output_dir = "docs/output/legal_analysis_test"
        os.makedirs(output_dir, exist_ok=True)
        
        analysis_path = os.path.join(output_dir, "text_analysis.json")
        with open(analysis_path, 'w', encoding='utf-8') as f:
            # Convert dataclass to dict for JSON serialization
            analysis_dict = {
                'document_structure': analysis.document_structure,
                'legal_terminology': analysis.legal_terminology,
                'complexity_metrics': analysis.complexity_metrics,
                'readability_score': analysis.readability_score,
                'key_sections': analysis.key_sections,
                'legal_citations': analysis.legal_citations,
                'entities': analysis.entities
            }
            json.dump(analysis_dict, f, indent=2, ensure_ascii=False)
        
        print(f"   ✓ Analysis saved to: {analysis_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def test_batch_processing():
    """Test batch processing functionality"""
    print("\n" + "=" * 80)
    print("TESTING BATCH PROCESSING")
    print("=" * 80)
    
    # For now, test with single file (can be extended for multiple files)
    pdf_files = ["docs/pdf_input/BA 311.2025.HS.ST.pdf"]
    
    try:
        processor = LegalDocumentProcessor()
        
        output_dir = "docs/output/batch_processing_test"
        os.makedirs(output_dir, exist_ok=True)
        
        print("1. Processing batch of legal PDFs...")
        results = processor.batch_process_legal_pdfs(pdf_files, output_dir)
        
        print(f"   ✓ Processed {len(results)} files")
        
        successful = len([r for r in results if r.get('success', True)])
        failed = len(results) - successful
        
        print(f"   ✓ Successful: {successful}")
        print(f"   ✓ Failed: {failed}")
        
        # Check if summary file was created
        summary_path = os.path.join(output_dir, "batch_processing_summary.json")
        if os.path.exists(summary_path):
            print(f"   ✓ Batch summary saved to: {summary_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {str(e)}")
        return False


def main():
    """Run all legal processing tests"""
    print("LEGAL DOCUMENT PROCESSING TEST SUITE")
    print("Testing enhanced functionality for Vietnamese legal documents")
    print("=" * 80)
    
    # Check if sample PDF exists
    pdf_path = Path("docs/pdf_input/BA 311.2025.HS.ST.pdf")
    if not pdf_path.exists():
        print(f"❌ Sample PDF not found: {pdf_path}")
        print("Please ensure the sample legal document is available for testing.")
        return
    
    print(f"✓ Sample PDF found: {pdf_path}")
    print(f"✓ File size: {pdf_path.stat().st_size / 1024:.1f} KB")
    
    # Run tests
    tests = [
        ("Legal Document Processor", test_legal_document_processor),
        ("Legal Text Analyzer", test_legal_text_analyzer),
        ("Batch Processing", test_batch_processing),
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
        print("🎉 All legal processing tests passed!")
        print("\nNext steps:")
        print("1. Review the generated analysis files in docs/output/")
        print("2. Fine-tune the legal entity extraction patterns")
        print("3. Add more legal terminology and patterns")
        print("4. Integrate with AI Agent for Q&A functionality")
    else:
        print("⚠️ Some tests failed. Check the output above for details.")


if __name__ == "__main__":
    main()