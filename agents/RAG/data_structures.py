"""
Enhanced Data Structures - Multimodal Support Added to Your Modern Structure
Current Date and Time (UTC): 2025-06-13 09:14:01
Current User's Login: Sagar4276
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

# ===== YOUR EXISTING STRUCTURES (KEPT UNCHANGED) =====

@dataclass
class ModernVectorizedChunk:
    """Modern vectorized chunk with enhanced metadata"""
    chunk_id: int
    doc_id: int
    chunk_index: int
    content: str
    vector: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'chunk_id': self.chunk_id,
            'doc_id': self.doc_id,
            'chunk_index': self.chunk_index,
            'content': self.content,
            'vector': self.vector.tolist() if isinstance(self.vector, np.ndarray) else self.vector,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModernVectorizedChunk':
        """Create from dictionary"""
        data['vector'] = np.array(data['vector'])
        return cls(**data)
    
    @property
    def content_length(self) -> int:
        """Get content length"""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get word count"""
        return len(self.content.split())

@dataclass
class ModernRetrievalResult:
    """Enhanced retrieval result with modern features"""
    content: str
    source: str
    similarity_score: float
    chunk_info: str
    matched_terms: List[str] = field(default_factory=list)
    doc_metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: int = 0
    
    # NEW: Multimodal support
    result_type: str = "text"  # "text", "image", "hybrid"
    image_data: Optional['MultimodalImageData'] = None
    clip_similarity: float = 0.0
    context_similarity: float = 0.0
    hybrid_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {
            'content': self.content,
            'source': self.source,
            'similarity_score': self.similarity_score,
            'chunk_info': self.chunk_info,
            'matched_terms': self.matched_terms,
            'doc_metadata': self.doc_metadata,
            'chunk_id': self.chunk_id,
            'result_type': self.result_type,
            'clip_similarity': self.clip_similarity,
            'context_similarity': self.context_similarity,
            'hybrid_score': self.hybrid_score
        }
        
        if self.image_data:
            result['image_data'] = self.image_data.to_dict()
        
        return result
    
    @property
    def similarity_percentage(self) -> str:
        """Get similarity as percentage"""
        return f"{self.similarity_score * 100:.1f}%"
    
    @property
    def short_content(self) -> str:
        """Get shortened content for display"""
        return self.content[:200] + "..." if len(self.content) > 200 else self.content

@dataclass
class ModernDocumentInfo:
    """Modern document information with enhanced features"""
    id: int
    filename: str
    filepath: str
    content: str
    chunks: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    processed_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # NEW: Multimodal support
    images_extracted: int = 0
    images_stored: int = 0
    has_multimodal_content: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'filepath': self.filepath,
            'content': self.content,
            'chunks': self.chunks,
            'metadata': self.metadata,
            'processed_at': self.processed_at,
            'images_extracted': self.images_extracted,
            'images_stored': self.images_stored,
            'has_multimodal_content': self.has_multimodal_content
        }
    
    @property
    def file_size_kb(self) -> int:
        """Get file size in KB"""
        return self.metadata.get('file_size', 0) // 1024
    
    @property
    def chunk_count(self) -> int:
        """Get number of chunks"""
        return len(self.chunks)
    
    @property
    def content_length(self) -> int:
        """Get total content length"""
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        """Get total word count"""
        return len(self.content.split())

# ===== NEW: SAFWAN'S MULTIMODAL STRUCTURES =====

@dataclass
class MultimodalImageData:
    """Image data structure based on Safwan's design"""
    id: str
    document_name: str
    page_number: int
    bbox: Tuple[float, float, float, float]  # x, y, width, height
    image_hash: str
    quality_score: float
    ocr_text: str
    ocr_confidence: float
    width: int
    height: int
    file_size: int
    clip_embedding: Optional[np.ndarray] = None
    context_text: str = ""
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def calculate_quality_score(self) -> float:
        """Calculate quality score based on Safwan's criteria"""
        # Size criteria (≥ 50k pixels)
        pixel_count = self.width * self.height
        size_score = min(pixel_count / 50000, 1.0)
        
        # Aspect ratio criteria
        aspect_ratio = max(self.width, self.height) / min(self.width, self.height)
        aspect_score = max(0, 1.0 - (aspect_ratio - 1.0) / 4.0)
        
        # OCR richness
        ocr_score = min(len(self.ocr_text) / 100, 1.0) if self.ocr_text else 0
        
        # Combined score (Safwan's formula)
        quality = (size_score * 0.4) + (aspect_score * 0.3) + (ocr_score * 0.3)
        return round(quality, 3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'document_name': self.document_name,
            'page_number': self.page_number,
            'bbox': self.bbox,
            'image_hash': self.image_hash,
            'quality_score': self.quality_score,
            'ocr_text': self.ocr_text,
            'ocr_confidence': self.ocr_confidence,
            'width': self.width,
            'height': self.height,
            'file_size': self.file_size,
            'context_text': self.context_text,
            'created_at': self.created_at,
            'clip_embedding': self.clip_embedding.tolist() if isinstance(self.clip_embedding, np.ndarray) else None
        }

@dataclass 
class MultimodalSearchQuery:
    """Enhanced search query with multimodal support"""
    text: str
    search_images: bool = False
    max_text_results: int = 5
    max_image_results: int = 4
    min_similarity: float = 0.1
    min_clip_similarity: float = 0.15  # Safwan's threshold
    min_quality_score: float = 0.3    # Safwan's threshold
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        'clip_visual': 0.5,     # Safwan's weights
        'context': 0.3,
        'proximity': 0.2
    })
    filters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'search_images': self.search_images,
            'max_text_results': self.max_text_results,
            'max_image_results': self.max_image_results,
            'min_similarity': self.min_similarity,
            'min_clip_similarity': self.min_clip_similarity,
            'min_quality_score': self.min_quality_score,
            'hybrid_weights': self.hybrid_weights,
            'filters': self.filters,
            'timestamp': self.timestamp
        }

@dataclass
class MultimodalConfig:
    """Configuration for multimodal RAG based on Safwan's settings"""
    # Text processing (your existing settings)
    chunk_size: int = 1500
    chunk_overlap: int = 200
    
    # Image processing (Safwan's settings)
    min_image_size: int = 80  # pixels
    max_images_per_query: int = 4
    min_pixel_count: int = 50000  # ≥ 50k pixels
    
    # Similarity thresholds (Safwan's values)
    text_similarity_threshold: float = 0.1
    clip_similarity_threshold: float = 0.15
    context_similarity_threshold: float = 0.25
    final_score_threshold: float = 0.3
    image_quality_threshold: float = 0.3
    
    # OCR settings
    ocr_confidence_threshold: float = 30.0
    
    # Model settings
    text_embedding_model: str = "all-MiniLM-L6-v2"  # Your choice
    clip_model: str = "openai/clip-vit-base-patch32"  # Safwan's choice
    
    # Hybrid scoring weights (Safwan's formula)
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {
        'clip_visual': 0.5,
        'context': 0.3,
        'proximity': 0.2
    })
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'min_image_size': self.min_image_size,
            'max_images_per_query': self.max_images_per_query,
            'min_pixel_count': self.min_pixel_count,
            'text_similarity_threshold': self.text_similarity_threshold,
            'clip_similarity_threshold': self.clip_similarity_threshold,
            'context_similarity_threshold': self.context_similarity_threshold,
            'final_score_threshold': self.final_score_threshold,
            'image_quality_threshold': self.image_quality_threshold,
            'ocr_confidence_threshold': self.ocr_confidence_threshold,
            'text_embedding_model': self.text_embedding_model,
            'clip_model': self.clip_model,
            'hybrid_weights': self.hybrid_weights
        }

# ===== EXISTING STRUCTURES (UNCHANGED) =====

@dataclass
class SearchQuery:
    """Search query with metadata"""
    text: str
    max_results: int = 5
    min_similarity: float = 0.1
    filters: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'text': self.text,
            'max_results': self.max_results,
            'min_similarity': self.min_similarity,
            'filters': self.filters,
            'timestamp': self.timestamp
        }

@dataclass
class SystemStats:
    """System statistics and health info"""
    documents_loaded: int
    total_chunks: int
    vector_dimension: int
    memory_usage_mb: float
    index_type: str
    embedding_model: str
    system_ready: bool
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # NEW: Multimodal stats
    images_indexed: int = 0
    multimodal_enabled: bool = False
    clip_model: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'documents_loaded': self.documents_loaded,
            'total_chunks': self.total_chunks,
            'vector_dimension': self.vector_dimension,
            'memory_usage_mb': self.memory_usage_mb,
            'index_type': self.index_type,
            'embedding_model': self.embedding_model,
            'system_ready': self.system_ready,
            'last_updated': self.last_updated,
            'images_indexed': self.images_indexed,
            'multimodal_enabled': self.multimodal_enabled,
            'clip_model': self.clip_model
        }
    
    @property
    def status(self) -> str:
        """Get system status"""
        if self.multimodal_enabled:
            return "🟢 Multimodal Ready" if self.system_ready else "🔴 Multimodal Not Ready"
        else:
            return "🟢 Text Ready" if self.system_ready else "🔴 Not Ready"
    
    @property
    def memory_usage_str(self) -> str:
        """Get formatted memory usage"""
        return f"{self.memory_usage_mb:.1f}MB"

@dataclass
class ProcessingResult:
    """Result of document processing"""
    success: bool
    message: str
    documents_processed: int = 0
    chunks_created: int = 0
    errors: List[str] = field(default_factory=list)
    processing_time_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    # NEW: Multimodal processing stats
    images_extracted: int = 0
    images_processed: int = 0
    ocr_operations: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'success': self.success,
            'message': self.message,
            'documents_processed': self.documents_processed,
            'chunks_created': self.chunks_created,
            'errors': self.errors,
            'processing_time_seconds': self.processing_time_seconds,
            'timestamp': self.timestamp,
            'images_extracted': self.images_extracted,
            'images_processed': self.images_processed,
            'ocr_operations': self.ocr_operations
        }
    
    @property
    def status_emoji(self) -> str:
        """Get status emoji"""
        return "✅" if self.success else "❌"

# ===== BACKWARD COMPATIBILITY =====
VectorizedChunk = ModernVectorizedChunk
RetrievalResult = ModernRetrievalResult
DocumentInfo = ModernDocumentInfo

# ===== TYPE HINTS =====
DocumentList = List[ModernDocumentInfo]
ResultList = List[ModernRetrievalResult]
ChunkList = List[ModernVectorizedChunk]
ImageList = List[MultimodalImageData]

# ===== CONSTANTS =====
DEFAULT_SIMILARITY_THRESHOLD = 0.1
DEFAULT_MAX_RESULTS = 5
DEFAULT_CHUNK_SIZE = 1500  # Updated for better context
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_CLIP_THRESHOLD = 0.15  # Safwan's setting
DEFAULT_QUALITY_THRESHOLD = 0.3  # Safwan's setting

# ===== UTILITY FUNCTIONS =====

def create_empty_result(message: str = "No results found") -> ModernRetrievalResult:
    """Create an empty result with a message"""
    return ModernRetrievalResult(
        content=message,
        source="System",
        similarity_score=0.0,
        chunk_info="No chunks",
        matched_terms=[],
        doc_metadata={'type': 'system_message'},
        chunk_id=-1,
        result_type="text"
    )

def create_multimodal_system_stats(documents: int, chunks: int, images: int, 
                                 vector_dim: int, memory_mb: float, 
                                 text_model: str, clip_model: Optional[str], 
                                 ready: bool) -> SystemStats:
    """Create multimodal system statistics"""
    return SystemStats(
        documents_loaded=documents,
        total_chunks=chunks,
        vector_dimension=vector_dim,
        memory_usage_mb=memory_mb,
        index_type="FAISS IndexFlatIP + CLIP",
        embedding_model=text_model,
        system_ready=ready,
        images_indexed=images,
        multimodal_enabled=clip_model is not None,
        clip_model=clip_model
    )

def validate_similarity_score(score: float) -> float:
    """Validate and normalize similarity score"""
    return max(0.0, min(1.0, score))

def calculate_hybrid_score(clip_sim: float, context_sim: float, 
                         proximity_score: float, quality_score: float,
                         weights: Dict[str, float] = None) -> float:
    """Calculate hybrid score using Safwan's formula"""
    if weights is None:
        weights = {'clip_visual': 0.5, 'context': 0.3, 'proximity': 0.2}
    
    base_score = (clip_sim * weights['clip_visual'] + 
                  context_sim * weights['context'] + 
                  proximity_score * weights['proximity'])
    
    # Multiply by quality score (Safwan's approach)
    final_score = base_score * quality_score
    
    return validate_similarity_score(final_score)

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes < 1024:
        return f"{size_bytes}B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes // 1024}KB"
    else:
        return f"{size_bytes // (1024 * 1024)}MB"

def calculate_chunk_stats(chunks: List[str]) -> Dict[str, Any]:
    """Calculate statistics for chunks"""
    if not chunks:
        return {'count': 0, 'avg_length': 0, 'total_words': 0}
    
    total_chars = sum(len(chunk) for chunk in chunks)
    total_words = sum(len(chunk.split()) for chunk in chunks)
    
    return {
        'count': len(chunks),
        'avg_length': total_chars // len(chunks),
        'total_words': total_words,
        'avg_words_per_chunk': total_words // len(chunks)
    }

def create_multimodal_config(enable_images: bool = True) -> MultimodalConfig:
    """Create default multimodal configuration"""
    return MultimodalConfig()

def detect_scientific_terms(text: str) -> List[str]:
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