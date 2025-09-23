"""
Module for integrating all PDF and image processing functionalities.
"""
import os
import json
import fitz  # PyMuPDF for PDF to image conversion
from typing import Dict, List, Optional, Tuple, Union

from .pdf_extractor import PDFExtractor
from .ocr_processor import OCRProcessor
from .layout_detector import LayoutDetector
from .text_normalizer import VietnameseTextNormalizer


class DocumentProcessor:
    """
    Class for processing documents with integrated PDF extraction, OCR, layout detection,
    and text normalization.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = "vie"):
        """
        Initialize the DocumentProcessor.
        
        Args:
            tesseract_cmd (str, optional): Path to tesseract executable
            lang (str): Language for OCR (default: 'vie' for Vietnamese)
        """
        self.ocr_processor = OCRProcessor(tesseract_cmd=tesseract_cmd, lang=lang)
        self.layout_detector = LayoutDetector()
        self.text_normalizer = VietnameseTextNormalizer()
    
    def process_pdf(self, pdf_path: str, output_dir: Optional[str] = None, 
                   extract_images: bool = False, normalize_text: bool = True,
                   expand_abbreviations: bool = False) -> Dict:
        """
        Process a PDF document with integrated extraction, layout detection, and normalization.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_dir (str, optional): Directory to save extracted images and analysis
            extract_images (bool): Whether to extract images from the PDF
            normalize_text (bool): Whether to normalize extracted text
            expand_abbreviations (bool): Whether to expand abbreviations in text
            
        Returns:
            Dict: Processed document data
        """
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract text from PDF
        with PDFExtractor(pdf_path) as pdf_extractor:
            # Get text with metadata
            pdf_data = pdf_extractor.extract_text_with_metadata()
            
            # Check if PDF is scanned (image-based) and needs OCR
            is_scanned = self._is_scanned_pdf(pdf_extractor, pdf_data)
            
            if is_scanned:
                # Use OCR for scanned PDFs
                pdf_data = self._process_scanned_pdf(pdf_extractor, output_dir)
            
            # Get text with positions for layout analysis
            text_positions = pdf_extractor.extract_text_with_positions()
            
            # Extract images if requested
            image_paths = []
            if extract_images and output_dir:
                images_dir = os.path.join(output_dir, "images")
                image_paths = pdf_extractor.extract_images(images_dir)
        
        # Normalize text if requested
        if normalize_text:
            normalized_pages = []
            for page_text in pdf_data["pages"]:
                normalized_text = self.text_normalizer.clean_and_normalize(
                    page_text, expand_abbreviations=expand_abbreviations
                )
                normalized_pages.append(normalized_text)
            pdf_data["normalized_pages"] = normalized_pages
        
        # Combine all data
        result = {
            "document_info": {
                "path": pdf_path,
                "metadata": pdf_data["metadata"],
                "total_pages": pdf_data["total_pages"]
            },
            "content": {
                "raw_pages": pdf_data["pages"],
                "normalized_pages": pdf_data.get("normalized_pages", []),
                "text_positions": text_positions,
                "extracted_images": image_paths
            }
        }
        
        # Save results if output directory is provided
        if output_dir:
            result_path = os.path.join(output_dir, "document_data.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def process_image(self, image_path: str, output_dir: Optional[str] = None,
                     preprocess: bool = True, detect_layout: bool = True,
                     normalize_text: bool = True, expand_abbreviations: bool = False) -> Dict:
        """
        Process an image document with OCR, layout detection, and text normalization.
        
        Args:
            image_path (str): Path to the image file
            output_dir (str, optional): Directory to save analysis results
            preprocess (bool): Whether to preprocess the image before OCR
            detect_layout (bool): Whether to detect layout in the image
            normalize_text (bool): Whether to normalize extracted text
            expand_abbreviations (bool): Whether to expand abbreviations in text
            
        Returns:
            Dict: Processed document data
        """
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Extract text from image using OCR
        ocr_text = self.ocr_processor.extract_text_from_image(image_path, preprocess=preprocess)
        
        # Extract text with layout information
        ocr_data = self.ocr_processor.extract_text_with_layout(image_path, preprocess=preprocess)
        
        # Detect layout if requested
        layout_data = None
        if detect_layout:
            layout_data = self.layout_detector.detect_layout_from_image(image_path)
            
            # Analyze document structure
            structure = self.layout_detector.analyze_document_structure(layout_data)
            layout_data["structure"] = structure
        
        # Normalize text if requested
        normalized_text = None
        if normalize_text:
            normalized_text = self.text_normalizer.clean_and_normalize(
                ocr_text, expand_abbreviations=expand_abbreviations
            )
        
        # Combine all data
        result = {
            "document_info": {
                "path": image_path,
                "type": "image"
            },
            "content": {
                "raw_text": ocr_text,
                "normalized_text": normalized_text,
                "ocr_data": ocr_data,
                "layout_data": layout_data
            }
        }
        
        # Save results if output directory is provided
        if output_dir:
            result_path = os.path.join(output_dir, "image_data.json")
            with open(result_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        
        return result
    
    def batch_process_images(self, image_dir: str, output_dir: Optional[str] = None,
                           preprocess: bool = True, detect_layout: bool = True,
                           normalize_text: bool = True, expand_abbreviations: bool = False) -> List[Dict]:
        """
        Process multiple images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            output_dir (str, optional): Directory to save analysis results
            preprocess (bool): Whether to preprocess images before OCR
            detect_layout (bool): Whether to detect layout in images
            normalize_text (bool): Whether to normalize extracted text
            expand_abbreviations (bool): Whether to expand abbreviations in text
            
        Returns:
            List[Dict]: List of processed document data
        """
        results = []
        
        # Create output directory if needed
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # Process each image in the directory
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(image_dir, filename)
                
                # Create individual output directory for this image
                if output_dir:
                    image_output_dir = os.path.join(output_dir, os.path.splitext(filename)[0])
                    if not os.path.exists(image_output_dir):
                        os.makedirs(image_output_dir)
                else:
                    image_output_dir = None
                
                # Process the image
                result = self.process_image(
                    image_path=image_path,
                    output_dir=image_output_dir,
                    preprocess=preprocess,
                    detect_layout=detect_layout,
                    normalize_text=normalize_text,
                    expand_abbreviations=expand_abbreviations
                )
                
                results.append(result)
        
        # Save combined results if output directory is provided
        if output_dir:
            combined_path = os.path.join(output_dir, "batch_results.json")
            with open(combined_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        
        return results
    
    def _is_scanned_pdf(self, pdf_extractor, pdf_data) -> bool:
        """
        Determine if a PDF is scanned (image-based) and needs OCR.
        
        Args:
            pdf_extractor: PDFExtractor instance
            pdf_data: Extracted PDF data
            
        Returns:
            bool: True if PDF is scanned and needs OCR
        """
        # Check if text extraction returned empty or very little text
        total_text_length = sum(len(page_text.strip()) for page_text in pdf_data["pages"])
        
        # If very little text was extracted, check for images
        if total_text_length < 50:  # Threshold for considering PDF as scanned
            # Check if pages contain images
            for page_num in range(len(pdf_extractor.doc)):
                page = pdf_extractor.doc[page_num]
                images = page.get_images()
                if images:
                    return True
        
        return False
    
    def _process_scanned_pdf(self, pdf_extractor, output_dir: Optional[str] = None) -> Dict:
        """
        Process a scanned PDF using OCR.
        
        Args:
            pdf_extractor: PDFExtractor instance
            output_dir: Optional output directory for temporary images
            
        Returns:
            Dict: Processed PDF data with OCR text
        """
        import tempfile
        import shutil
        
        # Create temporary directory for images if no output_dir provided
        temp_dir = None
        if output_dir:
            images_dir = os.path.join(output_dir, "temp_images")
        else:
            temp_dir = tempfile.mkdtemp()
            images_dir = temp_dir
        
        try:
            # Extract images from PDF for OCR
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            # Convert PDF pages to images and process with OCR
            pages_text = []
            for page_num in range(len(pdf_extractor.doc)):
                page = pdf_extractor.doc[page_num]
                
                # Convert page to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better OCR
                img_path = os.path.join(images_dir, f"page_{page_num + 1}.png")
                pix.save(img_path)
                
                # Process with OCR
                ocr_text = self.ocr_processor.extract_text_from_image(img_path)
                pages_text.append(ocr_text)
                
                # Clean up page image if using temp directory
                if temp_dir:
                    os.remove(img_path)
            
            # Create result similar to regular PDF extraction
            result = {
                "metadata": pdf_extractor.doc.metadata,
                "pages": pages_text,
                "total_pages": len(pdf_extractor.doc)
            }
            
            return result
            
        finally:
            # Clean up temporary directory
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)