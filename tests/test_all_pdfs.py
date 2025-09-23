#!/usr/bin/env python3
"""
Comprehensive test script for all PDF files in pdf_input directory.
Tests the legal document processing system with various document types.
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.pdf_processing.legal_document_processor import LegalDocumentProcessor
from src.utils.pdf_processing.legal_text_analyzer import LegalTextAnalyzer


class ComprehensiveTestRunner:
    """Comprehensive test runner for all PDF files"""
    
    def __init__(self):
        self.legal_processor = LegalDocumentProcessor()
        self.text_analyzer = LegalTextAnalyzer()
        self.pdf_input_dir = "docs/pdf_input"
        self.output_dir = "docs/output/comprehensive_test"
        self.results = []
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_all_pdf_files(self) -> List[str]:
        """Get all PDF files from input directory"""
        pdf_files = []
        for file in os.listdir(self.pdf_input_dir):
            if file.endswith('.pdf'):
                pdf_files.append(os.path.join(self.pdf_input_dir, file))
        return sorted(pdf_files)
    
    def categorize_document(self, filename: str) -> str:
        """Categorize document type based on filename"""
        if filename.startswith('BA'):
            return 'Bản án (Criminal Judgment)'
        elif filename.startswith('QD'):
            return 'Quyết định (Civil Decision)'
        else:
            return 'Unknown'
    
    def process_single_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Process a single PDF file and return results"""
        filename = os.path.basename(pdf_path)
        print(f"\n{'='*80}")
        print(f"PROCESSING: {filename}")
        print(f"{'='*80}")
        
        start_time = time.time()
        result = {
            'filename': filename,
            'file_path': pdf_path,
            'document_category': self.categorize_document(filename),
            'processing_start': datetime.now().isoformat(),
            'success': False,
            'error': None,
            'processing_time': 0,
            'legal_analysis': None,
            'text_analysis': None,
            'statistics': {}
        }
        
        try:
            # Create individual output directory for this file
            file_output_dir = os.path.join(self.output_dir, filename.replace('.pdf', ''))
            os.makedirs(file_output_dir, exist_ok=True)
            
            print(f"1. Processing legal document...")
            # Process with legal document processor
            legal_result = self.legal_processor.process_legal_pdf(
                pdf_path=pdf_path,
                output_dir=file_output_dir
            )
            
            print(f"2. Analyzing text content...")
            # Get normalized text for analysis
            normalized_pages = legal_result['content']['normalized_pages']
            full_text = '\n'.join(normalized_pages) if normalized_pages else ''
            
            # Perform text analysis
            text_analysis_result = self.text_analyzer.analyze_text(full_text)
            
            # Convert to dict for JSON serialization
            text_analysis = {
                'document_structure': text_analysis_result.document_structure,
                'legal_terminology': text_analysis_result.legal_terminology,
                'complexity_metrics': text_analysis_result.complexity_metrics,
                'readability_score': text_analysis_result.readability_score,
                'key_sections': text_analysis_result.key_sections,
                'legal_citations': text_analysis_result.legal_citations,
                'entities': text_analysis_result.entities,
                'statistics': {
                    'character_count': len(full_text),
                    'word_count': len(full_text.split()),
                    'sentence_count': len([s for s in full_text.split('.') if s.strip()]),
                    'legal_term_density': text_analysis_result.complexity_metrics.get('legal_term_density', 0)
                }
            }
            
            # Calculate statistics
            stats = self.calculate_statistics(legal_result, text_analysis)
            
            # Update result
            result.update({
                'success': True,
                'legal_analysis': legal_result.get('legal_analysis', {}),
                'text_analysis': text_analysis,
                'statistics': stats,
                'processing_time': time.time() - start_time
            })
            
            print(f"✓ Successfully processed {filename}")
            print(f"  - Document type: {result['document_category']}")
            print(f"  - Processing time: {result['processing_time']:.2f}s")
            print(f"  - Text length: {stats.get('text_length', 0):,} characters")
            print(f"  - Legal entities found: {stats.get('total_entities', 0)}")
            
        except Exception as e:
            result['error'] = str(e)
            result['processing_time'] = time.time() - start_time
            print(f"❌ Error processing {filename}: {e}")
        
        return result
    
    def calculate_statistics(self, legal_result: Dict, text_analysis: Dict) -> Dict[str, Any]:
        """Calculate comprehensive statistics"""
        legal_info = legal_result.get('legal_analysis', {}).get('document_info', {})
        
        # Count entities
        entity_counts = {
            'persons': len(legal_info.get('parties', {}).get('defendants', [])) + 
                      len(legal_info.get('parties', {}).get('plaintiffs', [])),
            'organizations': len([ref for ref in legal_info.get('legal_references', []) 
                                if any(org in ref for org in ['Tòa', 'Cơ quan', 'Ủy ban'])]),
            'legal_references': len(legal_info.get('legal_references', [])),
            'charges': len(legal_info.get('charges', []))
        }
        
        # Text statistics
        text_stats = text_analysis.get('statistics', {})
        
        return {
            'text_length': text_stats.get('character_count', 0),
            'word_count': text_stats.get('word_count', 0),
            'sentence_count': text_stats.get('sentence_count', 0),
            'legal_term_density': text_stats.get('legal_term_density', 0),
            'readability_score': text_analysis.get('readability_score', 0),
            'total_entities': sum(entity_counts.values()),
            'entity_breakdown': entity_counts,
            'confidence_score': legal_result.get('legal_analysis', {}).get('confidence_score', 0),
            'document_type': legal_info.get('document_type', 'Unknown'),
            'case_number': legal_info.get('case_number', 'Not found'),
            'court_name': legal_info.get('court_name', 'Not found')
        }
    
    def run_comprehensive_test(self):
        """Run comprehensive test on all PDF files"""
        print("🚀 STARTING COMPREHENSIVE PDF PROCESSING TEST")
        print("="*80)
        
        pdf_files = self.get_all_pdf_files()
        print(f"Found {len(pdf_files)} PDF files to process:")
        for i, pdf_path in enumerate(pdf_files, 1):
            filename = os.path.basename(pdf_path)
            category = self.categorize_document(filename)
            print(f"  {i:2d}. {filename} ({category})")
        
        print(f"\nOutput directory: {self.output_dir}")
        print("="*80)
        
        # Process each file
        total_start_time = time.time()
        for pdf_path in pdf_files:
            result = self.process_single_pdf(pdf_path)
            self.results.append(result)
        
        total_time = time.time() - total_start_time
        
        # Generate summary report
        self.generate_summary_report(total_time)
        
        print(f"\n🎉 COMPREHENSIVE TEST COMPLETED!")
        print(f"Total processing time: {total_time:.2f}s")
        print(f"Results saved to: {self.output_dir}")
    
    def generate_summary_report(self, total_time: float):
        """Generate comprehensive summary report"""
        print(f"\n{'='*80}")
        print("GENERATING SUMMARY REPORT")
        print(f"{'='*80}")
        
        # Calculate overall statistics
        successful = [r for r in self.results if r['success']]
        failed = [r for r in self.results if not r['success']]
        
        # Document type breakdown
        doc_types = {}
        for result in successful:
            doc_type = result['document_category']
            if doc_type not in doc_types:
                doc_types[doc_type] = []
            doc_types[doc_type].append(result)
        
        # Performance statistics
        processing_times = [r['processing_time'] for r in successful]
        avg_time = sum(processing_times) / len(processing_times) if processing_times else 0
        
        # Text statistics
        text_lengths = [r['statistics'].get('text_length', 0) for r in successful]
        avg_text_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        
        # Entity statistics
        total_entities = sum(r['statistics'].get('total_entities', 0) for r in successful)
        
        summary = {
            'test_summary': {
                'total_files': len(self.results),
                'successful': len(successful),
                'failed': len(failed),
                'success_rate': len(successful) / len(self.results) * 100 if self.results else 0,
                'total_processing_time': total_time,
                'average_processing_time': avg_time
            },
            'document_types': {
                doc_type: {
                    'count': len(results),
                    'avg_processing_time': sum(r['processing_time'] for r in results) / len(results),
                    'avg_confidence': sum(r['statistics'].get('confidence_score', 0) for r in results) / len(results),
                    'avg_entities': sum(r['statistics'].get('total_entities', 0) for r in results) / len(results)
                }
                for doc_type, results in doc_types.items()
            },
            'overall_statistics': {
                'total_entities_extracted': total_entities,
                'average_text_length': avg_text_length,
                'average_confidence_score': sum(r['statistics'].get('confidence_score', 0) for r in successful) / len(successful) if successful else 0
            },
            'failed_files': [
                {
                    'filename': r['filename'],
                    'error': r['error'],
                    'category': r['document_category']
                }
                for r in failed
            ],
            'detailed_results': self.results
        }
        
        # Save summary report
        summary_path = os.path.join(self.output_dir, 'comprehensive_test_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Print summary to console
        print(f"📊 TEST RESULTS SUMMARY:")
        print(f"  Total files processed: {summary['test_summary']['total_files']}")
        print(f"  Successful: {summary['test_summary']['successful']}")
        print(f"  Failed: {summary['test_summary']['failed']}")
        print(f"  Success rate: {summary['test_summary']['success_rate']:.1f}%")
        print(f"  Total processing time: {total_time:.2f}s")
        print(f"  Average processing time: {avg_time:.2f}s")
        
        print(f"\n📈 DOCUMENT TYPE BREAKDOWN:")
        for doc_type, stats in summary['document_types'].items():
            print(f"  {doc_type}:")
            print(f"    - Count: {stats['count']}")
            print(f"    - Avg processing time: {stats['avg_processing_time']:.2f}s")
            print(f"    - Avg confidence: {stats['avg_confidence']:.2f}")
            print(f"    - Avg entities: {stats['avg_entities']:.1f}")
        
        print(f"\n📋 OVERALL STATISTICS:")
        print(f"  Total entities extracted: {total_entities}")
        print(f"  Average text length: {avg_text_length:,.0f} characters")
        print(f"  Average confidence score: {summary['overall_statistics']['average_confidence_score']:.2f}")
        
        if failed:
            print(f"\n❌ FAILED FILES:")
            for fail in summary['failed_files']:
                print(f"  - {fail['filename']} ({fail['category']}): {fail['error']}")
        
        print(f"\n💾 Summary report saved to: {summary_path}")


def main():
    """Main function"""
    test_runner = ComprehensiveTestRunner()
    test_runner.run_comprehensive_test()


if __name__ == "__main__":
    main()