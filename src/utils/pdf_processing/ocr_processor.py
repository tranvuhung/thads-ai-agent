"""
Module for OCR processing of scanned documents and images.
"""
import os
import cv2
import numpy as np
import pytesseract
from PIL import Image
from typing import Dict, List, Optional, Tuple, Union


class OCRProcessor:
    """
    Class for OCR processing of scanned documents and images.
    """
    
    def __init__(self, tesseract_cmd: Optional[str] = None, lang: str = "vie"):
        """
        Initialize the OCRProcessor.
        
        Args:
            tesseract_cmd (str, optional): Path to tesseract executable
            lang (str): Language for OCR (default: 'vie' for Vietnamese)
        """
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        self.lang = lang
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            np.ndarray: Preprocessed image
        """
        # Read image
        img = cv2.imread(image_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply thresholding to handle shadows and normalize background
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        
        # Apply dilation and erosion to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opening
    
    def extract_text_from_image(self, image_path: str, preprocess: bool = True) -> str:
        """
        Extract text from an image using OCR.
        
        Args:
            image_path (str): Path to the image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            str: Extracted text
        """
        if preprocess:
            img = self.preprocess_image(image_path)
            # Convert numpy array to PIL Image
            pil_img = Image.fromarray(img)
        else:
            pil_img = Image.open(image_path)
        
        # Extract text using pytesseract
        text = pytesseract.image_to_string(pil_img, lang=self.lang)
        
        return text
    
    def extract_text_with_layout(self, image_path: str, preprocess: bool = True) -> Dict:
        """
        Extract text with layout information.
        
        Args:
            image_path (str): Path to the image
            preprocess (bool): Whether to preprocess the image
            
        Returns:
            Dict: Dictionary containing text and layout information
        """
        if preprocess:
            img = self.preprocess_image(image_path)
        else:
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Get detailed information including bounding boxes
        data = pytesseract.image_to_data(img, lang=self.lang, output_type=pytesseract.Output.DICT)
        
        # Process the data to create a structured output
        result = {
            "blocks": []
        }
        
        n_boxes = len(data['text'])
        current_block = {"text": "", "lines": []}
        current_line = {"text": "", "words": []}
        
        for i in range(n_boxes):
            if data['text'][i].strip() != '':
                word = {
                    "text": data['text'][i],
                    "confidence": data['conf'][i],
                    "bbox": (data['left'][i], data['top'][i], 
                             data['left'][i] + data['width'][i], 
                             data['top'][i] + data['height'][i])
                }
                
                current_line["words"].append(word)
                current_line["text"] += data['text'][i] + " "
                
            # New line
            if (i + 1 < n_boxes and data['line_num'][i] != data['line_num'][i+1]) or i == n_boxes - 1:
                if current_line["text"].strip():
                    current_block["lines"].append(current_line)
                    current_block["text"] += current_line["text"] + "\n"
                    current_line = {"text": "", "words": []}
            
            # New block
            if (i + 1 < n_boxes and data['block_num'][i] != data['block_num'][i+1]) or i == n_boxes - 1:
                if current_block["text"].strip():
                    result["blocks"].append(current_block)
                    current_block = {"text": "", "lines": []}
        
        return result
    
    def batch_process_images(self, image_dir: str, output_file: Optional[str] = None) -> Dict[str, str]:
        """
        Process multiple images in a directory.
        
        Args:
            image_dir (str): Directory containing images
            output_file (str, optional): Path to save the combined text
            
        Returns:
            Dict[str, str]: Dictionary mapping image paths to extracted text
        """
        results = {}
        
        for filename in os.listdir(image_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
                image_path = os.path.join(image_dir, filename)
                text = self.extract_text_from_image(image_path)
                results[image_path] = text
        
        # Optionally save combined text to a file
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                for image_path, text in results.items():
                    f.write(f"=== {os.path.basename(image_path)} ===\n")
                    f.write(text)
                    f.write("\n\n")
        
        return results