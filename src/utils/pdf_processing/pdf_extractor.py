"""
Module for extracting text from PDF files.
"""
import os
import fitz  # PyMuPDF
from typing import Dict, List, Optional, Tuple, Union


class PDFExtractor:
    """
    Class for extracting text from PDF files.
    """
    
    def __init__(self, pdf_path: str):
        """
        Initialize the PDFExtractor with a PDF file path.
        
        Args:
            pdf_path (str): Path to the PDF file
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
        
    def extract_text(self) -> str:
        """
        Extract all text from the PDF as a single string.
        
        Returns:
            str: All text content from the PDF
        """
        all_text = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            all_text.append(text)
            
        return "\n".join(all_text)
        
    def extract_text_by_page(self) -> List[str]:
        """
        Extract text from each page of the PDF.
        
        Returns:
            List[str]: List of text content for each page
        """
        text_by_page = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            text = page.get_text()
            text_by_page.append(text)
            
        return text_by_page
    
    def extract_text_with_metadata(self) -> Dict[str, Union[str, List[str]]]:
        """
        Extract text and metadata from the PDF.
        
        Returns:
            Dict: Dictionary containing text content and metadata
        """
        metadata = self.doc.metadata
        text_by_page = self.extract_text_by_page()
        
        result = {
            "metadata": metadata,
            "pages": text_by_page,
            "total_pages": len(self.doc)
        }
        
        return result
    
    def extract_text_with_positions(self) -> List[Dict]:
        """
        Extract text with position information for layout analysis.
        
        Returns:
            List[Dict]: List of dictionaries containing text blocks with position information
        """
        pages_blocks = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            page_blocks = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            page_blocks.append({
                                "text": span["text"],
                                "bbox": span["bbox"],  # (x0, y0, x1, y1)
                                "font": span["font"],
                                "size": span["size"]
                            })
            
            pages_blocks.append({
                "page_num": page_num + 1,
                "blocks": page_blocks
            })
            
        return pages_blocks
    
    def extract_images(self, output_dir: str) -> List[str]:
        """
        Extract images from the PDF.
        
        Args:
            output_dir (str): Directory to save extracted images
            
        Returns:
            List[str]: List of paths to extracted images
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        image_paths = []
        
        for page_num in range(len(self.doc)):
            page = self.doc[page_num]
            image_list = page.get_images(full=True)
            
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = self.doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                image_ext = base_image["ext"]
                image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
                image_path = os.path.join(output_dir, image_filename)
                
                with open(image_path, "wb") as img_file:
                    img_file.write(image_bytes)
                    
                image_paths.append(image_path)
                
        return image_paths
    
    def close(self):
        """
        Close the PDF document.
        """
        self.doc.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()