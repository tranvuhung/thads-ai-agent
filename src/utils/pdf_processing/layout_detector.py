"""
Module for document layout detection and structure analysis.
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import json
import os


class LayoutDetector:
    """
    Class for document layout detection and structure analysis.
    """
    
    def __init__(self):
        """
        Initialize the LayoutDetector.
        """
        pass
    
    def detect_layout_from_image(self, image_path: str) -> Dict:
        """
        Detect layout elements from an image.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            Dict: Dictionary containing detected layout elements
        """
        # Read image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get image dimensions
        height, width = gray.shape
        
        # Detect lines
        horizontal_lines, vertical_lines = self._detect_lines(gray)
        
        # Detect tables
        tables = self._detect_tables(horizontal_lines, vertical_lines, height, width)
        
        # Detect paragraphs and text blocks
        text_blocks = self._detect_text_blocks(gray, tables)
        
        # Detect headers and footers
        headers, footers = self._detect_headers_footers(gray, height, width)
        
        # Combine results
        layout = {
            "tables": tables,
            "text_blocks": text_blocks,
            "headers": headers,
            "footers": footers,
            "image_size": (width, height)
        }
        
        return layout
    
    def _detect_lines(self, gray_img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect horizontal and vertical lines in the document.
        
        Args:
            gray_img (np.ndarray): Grayscale image
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Horizontal and vertical lines
        """
        # Threshold the image
        thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Create kernels for line detection
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        # Detect horizontal lines
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        
        # Detect vertical lines
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        return horizontal_lines, vertical_lines
    
    def _detect_tables(self, horizontal_lines: np.ndarray, vertical_lines: np.ndarray, 
                      height: int, width: int) -> List[Dict]:
        """
        Detect tables in the document based on line intersections.
        
        Args:
            horizontal_lines (np.ndarray): Detected horizontal lines
            vertical_lines (np.ndarray): Detected vertical lines
            height (int): Image height
            width (int): Image width
            
        Returns:
            List[Dict]: List of detected tables with their coordinates
        """
        # Combine horizontal and vertical lines
        table_mask = cv2.bitwise_or(horizontal_lines, vertical_lines)
        
        # Find contours in the combined image
        contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out small regions
            if w > width * 0.1 and h > height * 0.05:
                tables.append({
                    "type": "table",
                    "bbox": (x, y, x + w, y + h)
                })
        
        return tables
    
    def _detect_text_blocks(self, gray_img: np.ndarray, tables: List[Dict]) -> List[Dict]:
        """
        Detect text blocks in the document.
        
        Args:
            gray_img (np.ndarray): Grayscale image
            tables (List[Dict]): Detected tables to exclude from text block detection
            
        Returns:
            List[Dict]: List of detected text blocks with their coordinates
        """
        # Create a mask for tables
        height, width = gray_img.shape
        table_mask = np.zeros((height, width), dtype=np.uint8)
        
        for table in tables:
            x1, y1, x2, y2 = table["bbox"]
            table_mask[y1:y2, x1:x2] = 255
        
        # Invert table mask
        table_mask_inv = cv2.bitwise_not(table_mask)
        
        # Apply threshold to get text regions
        thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Remove tables from text regions
        text_mask = cv2.bitwise_and(thresh, thresh, mask=table_mask_inv)
        
        # Apply morphological operations to connect text into blocks
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        text_mask = cv2.dilate(text_mask, kernel, iterations=3)
        text_mask = cv2.erode(text_mask, kernel, iterations=2)
        
        # Find contours for text blocks
        contours, _ = cv2.findContours(text_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        text_blocks = []
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small regions
            if w > 50 and h > 15:
                text_blocks.append({
                    "type": "text_block",
                    "bbox": (x, y, x + w, y + h)
                })
        
        return text_blocks
    
    def _detect_headers_footers(self, gray_img: np.ndarray, height: int, width: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Detect headers and footers in the document.
        
        Args:
            gray_img (np.ndarray): Grayscale image
            height (int): Image height
            width (int): Image width
            
        Returns:
            Tuple[List[Dict], List[Dict]]: Lists of detected headers and footers
        """
        # Define header and footer regions (top 10% and bottom 10% of the page)
        header_region = gray_img[:int(height * 0.1), :]
        footer_region = gray_img[int(height * 0.9):, :]
        
        # Apply threshold to get text in header and footer regions
        header_thresh = cv2.threshold(header_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        footer_thresh = cv2.threshold(footer_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        
        # Find contours in header region
        header_contours, _ = cv2.findContours(header_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        headers = []
        for contour in header_contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter out very small regions
            if w > 50 and h > 10:
                headers.append({
                    "type": "header",
                    "bbox": (x, y, x + w, y + h)
                })
        
        # Find contours in footer region
        footer_contours, _ = cv2.findContours(footer_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        footers = []
        for contour in footer_contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Adjust y-coordinate to account for footer region offset
            y += int(height * 0.9)
            
            # Filter out very small regions
            if w > 50 and h > 10:
                footers.append({
                    "type": "footer",
                    "bbox": (x, y, x + w, y + h)
                })
        
        return headers, footers
    
    def analyze_document_structure(self, layout: Dict) -> Dict:
        """
        Analyze document structure based on detected layout elements.
        
        Args:
            layout (Dict): Detected layout elements
            
        Returns:
            Dict: Document structure analysis
        """
        # Extract layout elements
        tables = layout.get("tables", [])
        text_blocks = layout.get("text_blocks", [])
        headers = layout.get("headers", [])
        footers = layout.get("footers", [])
        image_width, image_height = layout.get("image_size", (0, 0))
        
        # Analyze document structure
        structure = {
            "document_type": self._determine_document_type(tables, text_blocks, headers, footers),
            "sections": self._identify_sections(text_blocks, image_height),
            "has_tables": len(tables) > 0,
            "table_count": len(tables),
            "has_header": len(headers) > 0,
            "has_footer": len(footers) > 0
        }
        
        return structure
    
    def _determine_document_type(self, tables: List[Dict], text_blocks: List[Dict], 
                               headers: List[Dict], footers: List[Dict]) -> str:
        """
        Determine document type based on layout elements.
        
        Args:
            tables (List[Dict]): Detected tables
            text_blocks (List[Dict]): Detected text blocks
            headers (List[Dict]): Detected headers
            footers (List[Dict]): Detected footers
            
        Returns:
            str: Document type
        """
        # Simple heuristic for document type determination
        if len(tables) > 3:
            return "table_heavy_document"
        elif len(tables) > 0 and len(text_blocks) > 10:
            return "mixed_document"
        elif len(headers) > 0 and len(footers) > 0 and len(text_blocks) > 5:
            return "formal_document"
        else:
            return "text_document"
    
    def _identify_sections(self, text_blocks: List[Dict], image_height: int) -> List[Dict]:
        """
        Identify document sections based on text block positions.
        
        Args:
            text_blocks (List[Dict]): Detected text blocks
            image_height (int): Image height
            
        Returns:
            List[Dict]: Identified document sections
        """
        # Sort text blocks by y-coordinate
        sorted_blocks = sorted(text_blocks, key=lambda block: block["bbox"][1])
        
        # Group blocks into sections based on vertical proximity
        sections = []
        current_section = []
        
        for i, block in enumerate(sorted_blocks):
            if not current_section:
                current_section.append(block)
            else:
                prev_block = current_section[-1]
                prev_bottom = prev_block["bbox"][3]
                curr_top = block["bbox"][1]
                
                # If blocks are close, add to current section
                if curr_top - prev_bottom < image_height * 0.05:
                    current_section.append(block)
                else:
                    # Start a new section
                    section_bbox = self._get_section_bbox(current_section)
                    sections.append({
                        "type": "section",
                        "bbox": section_bbox,
                        "block_count": len(current_section)
                    })
                    current_section = [block]
        
        # Add the last section
        if current_section:
            section_bbox = self._get_section_bbox(current_section)
            sections.append({
                "type": "section",
                "bbox": section_bbox,
                "block_count": len(current_section)
            })
        
        return sections
    
    def _get_section_bbox(self, blocks: List[Dict]) -> Tuple[int, int, int, int]:
        """
        Get bounding box for a section.
        
        Args:
            blocks (List[Dict]): Blocks in the section
            
        Returns:
            Tuple[int, int, int, int]: Section bounding box
        """
        min_x = min(block["bbox"][0] for block in blocks)
        min_y = min(block["bbox"][1] for block in blocks)
        max_x = max(block["bbox"][2] for block in blocks)
        max_y = max(block["bbox"][3] for block in blocks)
        
        return (min_x, min_y, max_x, max_y)
    
    def save_layout_analysis(self, layout: Dict, output_path: str) -> None:
        """
        Save layout analysis to a JSON file.
        
        Args:
            layout (Dict): Layout analysis
            output_path (str): Path to save the analysis
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(layout, f, indent=2)