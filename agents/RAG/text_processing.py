"""
Enhanced Text Processing Module - Multimodal Support Added
Current Date and Time (UTC): 2025-06-13 09:21:09
Current User's Login: Sagar4276
"""

import os
import re
import numpy as np
import hashlib
from typing import List, Optional, Tuple
from PIL import Image, ImageEnhance, ImageFilter
from io import BytesIO
import fitz

# Import your existing structures + new multimodal ones
from .data_structures import MultimodalImageData, MultimodalConfig

class TextProcessor:
    """Enhanced text processor with multimodal capabilities"""
    
    def __init__(self, current_user: str = "Sagar4276", enable_multimodal: bool = True):
        self.current_user = current_user
        self.enable_multimodal = enable_multimodal
        self.config = MultimodalConfig()
        
        # Initialize OCR engines if multimodal is enabled
        self.ocr_reader = None
        self.pytesseract = None
        
        if self.enable_multimodal:
            self._initialize_ocr_engines()
    
    def _initialize_ocr_engines(self):
        """Initialize OCR engines (Safwan's approach)"""
        try:
            # Try EasyOCR first (better for some content)
            import easyocr
            self.ocr_reader = easyocr.Reader(['en'])
            print(f"[TextProcessor] ✅ EasyOCR initialized for {self.current_user}")
        except ImportError:
            print(f"[TextProcessor] ⚠️ EasyOCR not available. Install: pip install easyocr")
        except Exception as e:
            print(f"[TextProcessor] ⚠️ EasyOCR init failed: {str(e)}")
        
        try:
            # Fallback to Tesseract
            import pytesseract
            self.pytesseract = pytesseract
            print(f"[TextProcessor] ✅ Tesseract available as fallback")
        except ImportError:
            print(f"[TextProcessor] ⚠️ Tesseract not available. Install: pip install pytesseract")
    
    # ===== YOUR EXISTING TEXT EXTRACTION (UNCHANGED) =====
    
    def extract_text(self, filepath: str) -> Optional[str]:
        """Extract text from various file formats"""
        try:
            file_ext = filepath.lower().split('.')[-1]
            
            if file_ext in ['txt', 'md']:
                return self._extract_text_file(filepath)
            elif file_ext == 'pdf':
                return self._extract_pdf_file(filepath)
            elif file_ext == 'docx':
                return self._extract_docx_file(filepath)
            else:
                print(f"[TextProcessor] ❌ Unsupported file type: {file_ext}")
                return None
                
        except Exception as e:
            print(f"[TextProcessor] ❌ Error reading {os.path.basename(filepath)}: {str(e)}")
            return None
    
    def _extract_text_file(self, filepath: str) -> Optional[str]:
        """Extract from text/markdown files"""
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                return content if len(content.strip()) > 50 else None
        except Exception as e:
            print(f"[TextProcessor] ❌ Text file error: {str(e)}")
            return None
    
    def _extract_pdf_file(self, filepath: str) -> Optional[str]:
        """Extract text from PDF with smart content filtering"""
        try:
            import PyPDF2
            with open(filepath, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text_parts = []
            
                for page_num, page in enumerate(reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text and len(page_text.strip()) > 50:
                            cleaned_page = self._clean_pdf_text(page_text)
                            
                            if self._is_content_page(cleaned_page):
                                text_parts.append(cleaned_page)
                                print(f"[TextProcessor] ✅ Page {page_num + 1}: Content accepted")
                            else:
                                print(f"[TextProcessor] ⚠️ Page {page_num + 1}: Skipping (TOC/header)")
                            
                    except Exception as e:
                        print(f"[TextProcessor] ⚠️ Error on page {page_num + 1}: {str(e)}")
                        continue
            
                if text_parts:
                    full_text = '\n\n'.join(text_parts)
                    print(f"[TextProcessor] ✅ PDF processed: {len(text_parts)} pages, {len(full_text)} characters")
                    return full_text if len(full_text.strip()) > 500 else None
                else:
                    print(f"[TextProcessor] ❌ No meaningful content found in PDF")
                    return None
                
        except ImportError:
            print(f"[TextProcessor] ❌ PyPDF2 not available. Install with: pip install PyPDF2")
            return None
        except Exception as e:
            print(f"[TextProcessor] ❌ PDF extraction error: {str(e)}")
            return None

    def _clean_pdf_text(self, text: str) -> str:
        """Clean PDF text efficiently"""
        if not text:
            return ""
        
        # Basic cleaning
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\.{3,}', ' ', text)  # Remove multiple dots (TOC patterns)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)  # Remove page numbers
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)  # Fix hyphenated words
        
        return text.strip()

    def _is_content_page(self, text: str) -> bool:
        """Smart detection of actual content vs TOC/headers"""
        if not text or len(text.strip()) < 100:
            return False
        return True
    
    def _extract_docx_file(self, filepath: str) -> Optional[str]:
        """Extract from DOCX files"""
        try:
            import docx
            doc = docx.Document(filepath)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = '\n'.join(paragraphs)
            return full_text if len(full_text.strip()) > 50 else None
            
        except ImportError:
            print(f"[TextProcessor] ❌ python-docx not available. Install with: pip install python-docx")
            return None
        except Exception as e:
            print(f"[TextProcessor] ❌ DOCX error: {str(e)}")
            return None
    
    # ===== NEW: MULTIMODAL PDF PROCESSING (SAFWAN'S APPROACH) =====
    
    def extract_multimodal_content(self, filepath: str) -> Tuple[Optional[str], List[MultimodalImageData]]:
        """Extract both text and images from PDF (Safwan's pipeline)"""
        if not self.enable_multimodal:
            # Fallback to text-only
            text_content = self.extract_text(filepath)
            return text_content, []
        
        print(f"[TextProcessor] 🖼️ Extracting multimodal content from: {os.path.basename(filepath)}")
        
        # Only process PDFs for multimodal (for now)
        if not filepath.lower().endswith('.pdf'):
            text_content = self.extract_text(filepath)
            return text_content, []
        
        try:
            # Use PyMuPDF for layout-aware extraction (Safwan's choice)
            import fitz
            
            pdf_document = fitz.open(filepath)
            text_parts = []
            all_images = []
            
            for page_num in range(pdf_document.page_count):
                print(f"[TextProcessor] 📄 Processing page {page_num + 1}/{pdf_document.page_count}")
                
                page = pdf_document[page_num]
                
                # Extract text
                page_text = page.get_text()
                if page_text.strip():
                    cleaned_text = self._clean_pdf_text(page_text)
                    if self._is_content_page(cleaned_text):
                        text_parts.append(cleaned_text)
                
                # Extract images (Safwan's method)
                page_images = self._extract_images_from_page_safwan(
                    page, os.path.basename(filepath), page_num, page_text
                )
                all_images.extend(page_images)
            
            pdf_document.close()
            
            # Combine text
            full_text = '\n\n'.join(text_parts) if text_parts else None
            
            print(f"[TextProcessor] ✅ Multimodal extraction complete:")
            print(f"[TextProcessor] 📝 Text length: {len(full_text) if full_text else 0} chars")
            print(f"[TextProcessor] 🖼️ Images extracted: {len(all_images)}")
            print(f"[TextProcessor] 🎯 Quality images: {len([img for img in all_images if img.quality_score >= self.config.image_quality_threshold])}")
            
            return full_text, all_images
            
        except ImportError:
            print(f"[TextProcessor] ❌ PyMuPDF not available. Install: pip install PyMuPDF")
            # Fallback to text-only
            text_content = self.extract_text(filepath)
            return text_content, []
        except Exception as e:
            print(f"[TextProcessor] ❌ Multimodal extraction error: {str(e)}")
            # Fallback to text-only
            text_content = self.extract_text(filepath)
            return text_content, []
    
    def _extract_images_from_page_safwan(self, page, document_name: str, 
                                       page_number: int, page_text: str) -> List[MultimodalImageData]:
        """Extract images from page using Safwan's approach"""
        images = []
        
        try:
            # Get images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image using PyMuPDF
                    xref = img[0]
                    pix = page.get_pixmap(matrix=fitz.Matrix(1, 1), clip=img[:4])
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.pil_tobytes(format="PNG")
                        pil_image = Image.open(BytesIO(img_data))
                        
                        # Process image using Safwan's criteria
                        image_obj = self._process_image_safwan_style(
                            pil_image, img_data, document_name, page_number, 
                            img_index, page_text
                        )
                        
                        if image_obj and image_obj.quality_score >= self.config.image_quality_threshold:
                            images.append(image_obj)
                            print(f"[TextProcessor] ✅ Image {img_index + 1}: Quality {image_obj.quality_score:.2f}")
                        else:
                            print(f"[TextProcessor] ⚠️ Image {img_index + 1}: Low quality, skipped")
                    
                    pix = None  # Free memory
                    
                except Exception as e:
                    print(f"[TextProcessor] ⚠️ Image {img_index + 1} processing error: {str(e)}")
                    continue
            
        except Exception as e:
            print(f"[TextProcessor] ⚠️ Page image extraction error: {str(e)}")
        
        return images
    
    def _process_image_safwan_style(self, pil_image: Image.Image, img_data: bytes,
                                   document_name: str, page_number: int,
                                   img_index: int, page_text: str) -> Optional[MultimodalImageData]:
        """Process image using Safwan's quality criteria"""
        try:
            # Create image hash for deduplication
            img_hash = hashlib.md5(img_data).hexdigest()
            
            # Basic image info
            width, height = pil_image.size
            file_size = len(img_data)
            
            # Safwan's quality assessment
            quality_score = self._assess_image_quality_safwan(pil_image)
            
            # Skip images below minimum size (Safwan's criteria)
            if width < self.config.min_image_size or height < self.config.min_image_size:
                return None
            
            # OCR processing (Safwan's approach)
            ocr_text, ocr_confidence = self._extract_ocr_text_safwan(pil_image)
            
            # Extract context (Safwan's ±500 chars around image)
            context_text = self._extract_image_context_safwan(page_text, img_index)
            
            # Create ImageData object
            image_obj = MultimodalImageData(
                id=f"{document_name}_page_{page_number}_img_{img_index}",
                document_name=document_name,
                page_number=page_number,
                bbox=(0.0, 0.0, float(width), float(height)),  # Simplified bbox
                image_hash=img_hash,
                quality_score=quality_score,
                ocr_text=ocr_text,
                ocr_confidence=ocr_confidence,
                width=width,
                height=height,
                file_size=file_size,
                context_text=context_text
            )
            
            return image_obj
            
        except Exception as e:
            print(f"[TextProcessor] ⚠️ Image processing error: {str(e)}")
            return None
    
    def _assess_image_quality_safwan(self, pil_image: Image.Image) -> float:
        """Assess image quality using Safwan's criteria"""
        try:
            width, height = pil_image.size
            
            # Size criteria (≥ 50k pixels)
            pixel_count = width * height
            size_score = min(pixel_count / self.config.min_pixel_count, 1.0)
            
            # Aspect ratio criteria
            aspect_ratio = max(width, height) / min(width, height)
            aspect_score = max(0, 1.0 - (aspect_ratio - 1.0) / 4.0)
            
            # Color diversity (simplified)
            img_array = np.array(pil_image)
            if len(img_array.shape) == 3:  # Color image
                color_var = np.var(img_array, axis=(0, 1)).mean()
                color_score = min(color_var / 1000, 1.0)
            else:
                color_score = 0.5  # Grayscale gets medium score
            
            # Safwan's combined scoring
            quality = (size_score * 0.4) + (aspect_score * 0.3) + (color_score * 0.3)
            return round(quality, 3)
            
        except Exception:
            return 0.0
    
    def _extract_ocr_text_safwan(self, pil_image: Image.Image) -> Tuple[str, float]:
        """Extract OCR text using Safwan's approach"""
        try:
            # Enhance image for OCR (Safwan's preprocessing)
            enhanced_image = self._enhance_image_for_ocr_safwan(pil_image)
            
            # Try EasyOCR first (Safwan's preference)
            if self.ocr_reader:
                try:
                    results = self.ocr_reader.readtext(np.array(enhanced_image))
                    text_parts = []
                    confidences = []
                    
                    for (bbox, text, confidence) in results:
                        if confidence > (self.config.ocr_confidence_threshold / 100):
                            text_parts.append(text)
                            confidences.append(confidence)
                    
                    if text_parts:
                        combined_text = " ".join(text_parts)
                        avg_confidence = np.mean(confidences) * 100
                        return combined_text, avg_confidence
                        
                except Exception as e:
                    print(f"[TextProcessor] ⚠️ EasyOCR failed: {str(e)}")
            
            # Fallback to Tesseract
            if self.pytesseract:
                text = self.pytesseract.image_to_string(enhanced_image)
                data = self.pytesseract.image_to_data(enhanced_image, output_type=self.pytesseract.Output.DICT)
                
                # Calculate confidence
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                avg_confidence = np.mean(confidences) if confidences else 0
                
                return text.strip(), avg_confidence
            
            return "", 0.0
            
        except Exception as e:
            print(f"[TextProcessor] ⚠️ OCR error: {str(e)}")
            return "", 0.0
    
    def _enhance_image_for_ocr_safwan(self, pil_image: Image.Image) -> Image.Image:
        """Enhance image for OCR using Safwan's preprocessing"""
        try:
            # Convert to grayscale
            if pil_image.mode != 'L':
                enhanced = pil_image.convert('L')
            else:
                enhanced = pil_image.copy()
            
            # Safwan's enhancements: contrast + sharpness
            # Enhance contrast
            enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = enhancer.enhance(1.5)
            
            # Enhance sharpness
            enhancer = ImageEnhance.Sharpness(enhanced)
            enhanced = enhancer.enhance(2.0)
            
            # Light blur to reduce noise
            enhanced = enhanced.filter(ImageFilter.GaussianBlur(radius=0.5))
            
            return enhanced
            
        except Exception:
            return pil_image
    
    def _extract_image_context_safwan(self, page_text: str, img_index: int) -> str:
        """Extract image context using Safwan's ±500 chars method"""
        try:
            # Safwan's approach: 500 chars before/after image position
            words = page_text.split()
            
            # Estimate image position (simple heuristic)
            estimated_position = len(words) // (img_index + 2)  # Rough estimate
            
            # Extract ±25 words (roughly 500 chars)
            start_idx = max(0, estimated_position - 25)
            end_idx = min(len(words), estimated_position + 25)
            
            context_words = words[start_idx:end_idx]
            return " ".join(context_words)
            
        except Exception:
            return page_text[:500]  # Fallback
    
    # ===== YOUR EXISTING CHUNKING (ENHANCED) =====
    
    def create_smart_chunks(self, text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
        """Create intelligent chunks optimized for modern embeddings"""
        
        # Use config values if not specified
        chunk_size = chunk_size or self.config.chunk_size
        overlap = overlap or self.config.chunk_overlap
        
        if not text or len(text.strip()) < 100:
            return []
        
        print(f"[TextProcessor] 🔄 Creating smart chunks from {len(text)} characters...")
        print(f"[TextProcessor] ⚙️ Chunk size: {chunk_size}, Overlap: {overlap}")
        
        # Clean the text first
        text = self._clean_for_chunking(text)
        
        if len(text) <= chunk_size:
            return [text] if len(text.strip()) > 200 else []
        
        # Split into meaningful segments - USING BETTER SEMANTIC SEGMENTATION
        segments = self._split_into_semantic_segments(text)
        print(f"[TextProcessor] 📄 Found {len(segments)} text segments")
        
        # Build chunks
        chunks = []
        current_chunk = ""
        
        for segment in segments:
            # Check if adding this segment exceeds chunk size
            if len(current_chunk) + len(segment) > chunk_size and current_chunk:
                if len(current_chunk.strip()) > 200:  # Lower minimum chunk size
                    chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                overlap_text = self._get_overlap(current_chunk, overlap)
                current_chunk = overlap_text + " " + segment if overlap_text else segment
            else:
                current_chunk = current_chunk + " " + segment if current_chunk else segment
        
        # Add final chunk
        if current_chunk.strip() and len(current_chunk.strip()) > 200:
            chunks.append(current_chunk.strip())
        
        # Filter chunks for quality - LESS AGGRESSIVE FILTERING
        quality_chunks = self._filter_for_content_quality(chunks)
        
        print(f"[TextProcessor] ✅ Created {len(quality_chunks)} quality chunks")
        return quality_chunks
    
    def _clean_for_chunking(self, text: str) -> str:
        """Clean text specifically for chunking"""
        # Remove obvious TOC patterns
        text = re.sub(r'table of contents.*?(?=\n\n|\n[A-Z])', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Clean up spacing
        text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize paragraph breaks
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        return text.strip()
    
    def _split_into_semantic_segments(self, text: str) -> List[str]:
        """Split text into meaningful semantic segments"""
        # First try to split by headings (## or ###)
        if '##' in text:
            segments = []
            heading_pattern = r'(#{2,3}\s+[^\n]+)'
            parts = re.split(heading_pattern, text)
            
            current_segment = ""
            for i, part in enumerate(parts):
                if re.match(heading_pattern, part):  # This is a heading
                    if current_segment:
                        segments.append(current_segment.strip())
                    current_segment = part
                else:
                    current_segment += part
            
            if current_segment:  # Add final segment
                segments.append(current_segment.strip())
                
            if segments:
                return [s for s in segments if len(s.strip()) > 50]
        
        # Try paragraph-based splitting
        if '\n\n' in text:
            segments = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
            if segments:
                return segments
        
        # Fallback to sentence-based splitting
        sentence_segments = []
        current_segment = ""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            if len(current_segment) + len(sentence) > 800:
                if current_segment:
                    sentence_segments.append(current_segment)
                current_segment = sentence
            else:
                current_segment = current_segment + " " + sentence if current_segment else sentence
        
        if current_segment:
            sentence_segments.append(current_segment)
            
        if sentence_segments:
            return sentence_segments
        
        # Last resort: word-based chunks
        words = text.split()
        segments = []
        for i in range(0, len(words), 200):
            segment = ' '.join(words[i:i + 200])
            if len(segment.strip()) > 50:
                segments.append(segment)
        
        return segments
    
    def _get_overlap(self, text: str, overlap_size: int) -> str:
        """Get overlap text for chunk continuity"""
        if len(text) <= overlap_size:
            return text
        
        overlap_text = text[-overlap_size:]
        
        # Find sentence boundary
        sentence_start = overlap_text.find('. ')
        if sentence_start != -1:
            return overlap_text[sentence_start + 2:]
        
        # Find word boundary
        word_start = overlap_text.find(' ')
        if word_start != -1:
            return overlap_text[word_start + 1:]
        
        return overlap_text
    
    def _filter_for_content_quality(self, chunks: List[str]) -> List[str]:
        """Filter chunks but be more lenient for medical content"""
        quality_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Basic quality checks
            words = chunk.split()
            if len(words) < 40:  # Lower minimum word count
                print(f"[TextProcessor] ⚠️ Chunk {i+1} too short: {len(words)} words")
                continue
                
            # For medical content, we need to be very lenient
            chunk_lower = chunk.lower()
            
            # Check for content words, but be more lenient
            content_words = [
                'parkinson', 'disease', 'patient', 'treatment', 'symptoms', 'diagnosis',
                'medical', 'research', 'study', 'brain', 'condition', 'therapy', 
                'medication', 'motor', 'tremor', 'movement', 'dopamine', 'neurological',
                'algorithm', 'neural', 'network', 'learning', 'data', 'analysis'
            ]
            
            # Count content terms
            content_matches = sum(1 for word in content_words if word in chunk_lower)
            
            # Accept most chunks, but log their quality
            if content_matches >= 2:
                quality_chunks.append(chunk)
                print(f"[TextProcessor] ✅ Chunk {i+1} approved: {len(words)} words, {content_matches} content terms")
            elif content_matches > 0 or len(words) >= 60:
                # Accept with at least one match or if long enough
                quality_chunks.append(chunk)
                print(f"[TextProcessor] ✅ Chunk {i+1} approved: {len(words)} words, {content_matches} content terms")
            else:
                print(f"[TextProcessor] ⚠️ Chunk {i+1} lacks content: 0 terms")
        
        return quality_chunks
    
    # ===== YOUR EXISTING UTILITY METHODS (UNCHANGED) =====
    
    def clean_text_for_embeddings(self, text: str) -> str:
        """Clean text specifically for modern embeddings (simplified)"""
        if not text:
            return ""
        
        # Basic normalization for embeddings
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'[^\x20-\x7E\n]', ' ', text)  # Keep only printable ASCII + newlines
        text = re.sub(r'\.{2,}', '.', text)  # Fix multiple periods
        
        return text.strip()
    
    def extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms for search (simplified for embeddings)"""
        if not text:
            return []
        
        # Simple word extraction for modern embeddings (they handle context better)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
        
        # Filter out common stop words
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
            'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'this',
            'that', 'with', 'have', 'will', 'they', 'from', 'been', 'said', 'each',
            'which', 'their', 'time', 'would', 'there', 'what', 'were', 'when'
        }
        
        # Keep medical terms and meaningful words
        meaningful_words = []
        for word in words:
            if len(word) > 3 and word not in stop_words:
                meaningful_words.append(word)
        
        return meaningful_words[:50]  # Limit for efficiency
    
    def create_excerpt(self, content: str, max_length: int = 400) -> str:
        """Create a clean excerpt from content"""
        if not content:
            return ""
        
        if len(content) <= max_length:
            return content
        
        # Try to cut at sentence boundary
        if '. ' in content[:max_length]:
            last_period = content[:max_length].rfind('. ')
            return content[:last_period + 1]
        
        # Fallback to word boundary
        if ' ' in content[:max_length]:
            last_space = content[:max_length].rfind(' ')
            return content[:last_space] + "..."
        
        return content[:max_length] + "..."
    
    def detect_scientific_terms(self, text: str) -> List[str]:
        """Detect scientific terms for bonus scoring (Safwan's feature)"""
        scientific_terms = [
            'algorithm', 'neural', 'network', 'machine learning', 'deep learning',
            'artificial intelligence', 'data science', 'statistics', 'regression',
            'classification', 'clustering', 'optimization', 'gradient', 'matrix',
            'vector', 'tensor', 'probability', 'distribution', 'hypothesis',
            'parkinson', 'neurological', 'dopamine', 'brain', 'medical', 'clinical'
        ]
        
        text_lower = text.lower()
        found_terms = [term for term in scientific_terms if term in text_lower]
        return found_terms