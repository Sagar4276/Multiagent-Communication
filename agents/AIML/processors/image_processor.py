"""
Image Processing Module for AI/ML Agent
Current Date and Time (UTC): 2025-06-14 12:01:34
Current User's Login: Sagar4276

Handles MRI and medical image processing with advanced features.
"""

import os
import time
import numpy as np
from PIL import Image
from typing import Dict, Any, Optional, Tuple
import base64
import io

class ImageProcessor:
    """Advanced image processing for medical images"""
    
    def __init__(self, aiml_agent):
        self.aiml_agent = aiml_agent
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.dcm']
        self.mri_formats = ['.dcm', '.nii', '.nifti']
        
        print(f"🖼️ [ImageProcessor] Initialized medical image processor")
    
    def process_uploaded_image(self, user_id: str, image_path: str, image_type: str) -> Dict[str, Any]:
        """Process uploaded image based on type"""
        
        print(f"🖼️ [ImageProcessor] Processing {image_type} image: {os.path.basename(image_path)}")
        
        try:
            # Validate image
            if not self._validate_image(image_path, image_type):
                return {
                    'status': 'error',
                    'error': 'Invalid image format or file not found'
                }
            
            # Load and preprocess image
            processed_image = self._load_and_preprocess(image_path, image_type)
            
            if processed_image['status'] != 'success':
                return processed_image
            
            # Extract metadata
            metadata = self._extract_metadata(image_path, processed_image['image'])
            
            return {
                'status': 'success',
                'image_type': image_type,
                'image_path': image_path,
                'processed_image': processed_image['image'],
                'metadata': metadata,
                'preprocessing_info': processed_image.get('info', {}),
                'timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Image processing failed: {str(e)}"
            }
    
    def extract_mri_features(self, processing_result: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from MRI for classification"""
        
        if processing_result['status'] != 'success':
            return {
                'status': 'error',
                'error': 'Cannot extract features from failed processing'
            }
        
        try:
            image = processing_result['processed_image']
            metadata = processing_result['metadata']
            
            # Convert to numpy array
            img_array = np.array(image)
            
            # Extract comprehensive features
            features = self._extract_comprehensive_features(img_array, metadata)
            
            return {
                'status': 'success',
                'features': features,
                'feature_count': len(features),
                'image_quality': self._assess_quality(img_array),
                'extraction_time': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Feature extraction failed: {str(e)}"
            }
    
    def _validate_image(self, image_path: str, image_type: str) -> bool:
        """Validate image file exists and has correct format"""
        
        if not os.path.exists(image_path):
            return False
        
        file_ext = os.path.splitext(image_path)[1].lower()
        
        if image_type.lower() == 'mri':
            return file_ext in self.mri_formats or file_ext in self.supported_formats
        else:
            return file_ext in self.supported_formats
    
    def _load_and_preprocess(self, image_path: str, image_type: str) -> Dict[str, Any]:
        """Load and preprocess image based on type"""
        
        try:
            file_ext = os.path.splitext(image_path)[1].lower()
            
            # Handle DICOM files
            if file_ext == '.dcm':
                return self._process_dicom(image_path)
            
            # Handle standard image formats
            else:
                return self._process_standard_image(image_path, image_type)
                
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Image loading failed: {str(e)}"
            }
    
    def _process_dicom(self, image_path: str) -> Dict[str, Any]:
        """Process DICOM format files"""
        
        try:
            # Try to import pydicom
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_windowing
            
            # Read DICOM file
            ds = pydicom.dcmread(image_path)
            
            # Extract pixel data
            pixel_array = ds.pixel_array
            
            # Apply windowing if available
            if hasattr(ds, 'WindowCenter') and hasattr(ds, 'WindowWidth'):
                pixel_array = apply_windowing(pixel_array, ds)
            
            # Normalize to 0-255 range
            pixel_array = ((pixel_array - pixel_array.min()) / 
                          (pixel_array.max() - pixel_array.min()) * 255).astype(np.uint8)
            
            # Convert to PIL Image
            if len(pixel_array.shape) == 2:
                image = Image.fromarray(pixel_array, mode='L').convert('RGB')
            else:
                image = Image.fromarray(pixel_array).convert('RGB')
            
            # Resize for processing
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            return {
                'status': 'success',
                'image': image,
                'info': {
                    'format': 'DICOM',
                    'original_size': pixel_array.shape,
                    'processed_size': image.size,
                    'patient_id': getattr(ds, 'PatientID', 'Unknown'),
                    'study_date': getattr(ds, 'StudyDate', 'Unknown'),
                    'modality': getattr(ds, 'Modality', 'Unknown')
                }
            }
            
        except ImportError:
            print(f"⚠️ [ImageProcessor] pydicom not available, treating as standard image")
            return self._process_standard_image(image_path, 'mri')
        
        except Exception as e:
            return {
                'status': 'error',
                'error': f"DICOM processing failed: {str(e)}"
            }
    
    def _process_standard_image(self, image_path: str, image_type: str) -> Dict[str, Any]:
        """Process standard image formats"""
        
        try:
            # Open and convert image
            image = Image.open(image_path).convert('RGB')
            original_size = image.size
            
            # Resize for processing (standard ML input size)
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            return {
                'status': 'success',
                'image': image,
                'info': {
                    'format': 'Standard Image',
                    'original_size': original_size,
                    'processed_size': image.size,
                    'mode': 'RGB'
                }
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Standard image processing failed: {str(e)}"
            }
    
    def _extract_metadata(self, image_path: str, image: Image) -> Dict[str, Any]:
        """Extract image metadata"""
        
        return {
            'filename': os.path.basename(image_path),
            'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
            'image_size': image.size,
            'image_mode': image.mode,
            'processing_timestamp': time.time(),
            'file_extension': os.path.splitext(image_path)[1].lower()
        }
    
    def _extract_comprehensive_features(self, img_array: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive features for ML classification"""
        
        # Flatten image for processing
        if len(img_array.shape) == 3:
            gray_array = np.mean(img_array, axis=2)
        else:
            gray_array = img_array
        
        # Statistical features
        statistical_features = {
            'mean_intensity': float(np.mean(gray_array)),
            'std_intensity': float(np.std(gray_array)),
            'min_intensity': float(np.min(gray_array)),
            'max_intensity': float(np.max(gray_array)),
            'median_intensity': float(np.median(gray_array)),
            'skewness': float(self._calculate_skewness(gray_array)),
            'kurtosis': float(self._calculate_kurtosis(gray_array))
        }
        
        # Histogram features
        hist, _ = np.histogram(gray_array.flatten(), bins=50, range=(0, 255))
        histogram_features = {
            'histogram_bins': hist.tolist(),
            'histogram_peak': int(np.argmax(hist)),
            'histogram_entropy': float(self._calculate_entropy(hist))
        }
        
        # Texture features (simplified)
        texture_features = {
            'contrast': float(self._calculate_contrast(gray_array)),
            'homogeneity': float(self._calculate_homogeneity(gray_array)),
            'energy': float(np.sum(gray_array ** 2))
        }
        
        # Shape and size features
        shape_features = {
            'width': img_array.shape[1],
            'height': img_array.shape[0],
            'total_pixels': img_array.shape[0] * img_array.shape[1],
            'aspect_ratio': img_array.shape[1] / img_array.shape[0]
        }
        
        return {
            'statistical': statistical_features,
            'histogram': histogram_features,
            'texture': texture_features,
            'shape': shape_features,
            'metadata': metadata
        }
    
    def _assess_quality(self, img_array: np.ndarray) -> str:
        """Assess image quality for classification"""
        
        # Calculate quality metrics
        std_dev = np.std(img_array)
        mean_intensity = np.mean(img_array)
        
        # Quality assessment
        if std_dev < 10:
            return "Poor - Very low contrast"
        elif std_dev < 30:
            return "Fair - Low contrast"
        elif std_dev > 80:
            return "Good - High contrast"
        elif mean_intensity < 20 or mean_intensity > 235:
            return "Fair - Extreme brightness"
        else:
            return "Good - Adequate contrast and brightness"
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        return np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    def _calculate_entropy(self, hist: np.ndarray) -> float:
        """Calculate histogram entropy"""
        hist_norm = hist / np.sum(hist)
        hist_norm = hist_norm[hist_norm > 0]  # Remove zeros
        return -np.sum(hist_norm * np.log2(hist_norm))
    
    def _calculate_contrast(self, img_array: np.ndarray) -> float:
        """Calculate simple contrast measure"""
        return np.std(img_array)
    
    def _calculate_homogeneity(self, img_array: np.ndarray) -> float:
        """Calculate homogeneity measure"""
        return 1.0 / (1.0 + np.var(img_array))
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information"""
        return {
            'name': 'ImageProcessor',
            'version': '1.0.0',
            'supported_formats': self.supported_formats,
            'mri_formats': self.mri_formats,
            'capabilities': [
                'DICOM processing',
                'Standard image processing',
                'Feature extraction',
                'Quality assessment',
                'Metadata extraction'
            ]
        }