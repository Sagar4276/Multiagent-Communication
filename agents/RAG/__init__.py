"""
Enhanced Multimodal RAG Module
Current Date and Time (UTC): 2025-06-13 09:40:12
Current User's Login: Sagar4276

A comprehensive multimodal RAG system combining:
- Your clean, efficient architecture
- Safwan's advanced CLIP-based image understanding
- Hybrid text + visual search capabilities
- Production-ready performance optimization

Built for: Research, exam preparation, and AI applications requiring 
both text and visual content understanding.
"""

__version__ = "2.0.0"
__author__ = "Sagar4276"
__date__ = "2025-06-13"

# ===== CORE IMPORTS =====

# Main RAG Agent (Primary Interface)
from .rag_agent import ModernEnhancedRAGAgent

# Core Components
from .text_processing import TextProcessor
from .vectorization import ModernVectorizationEngine
from .retrieval import ModernRetrievalEngine

# Data Structures
from .data_structures import (
    # Your existing structures
    ModernVectorizedChunk,
    ModernRetrievalResult,
    ModernDocumentInfo,
    SearchQuery,
    SystemStats,
    ProcessingResult,
    
    # New multimodal structures
    MultimodalImageData,
    MultimodalSearchQuery,
    MultimodalConfig,
    
    # Utility functions
    create_empty_result,
    create_multimodal_system_stats,
    calculate_hybrid_score,
    detect_scientific_terms,
    format_file_size,
    calculate_chunk_stats,
    
    # Type hints
    DocumentList,
    ResultList,
    ChunkList,
    ImageList,
    
    # Constants
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_MAX_RESULTS,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CLIP_THRESHOLD,
    DEFAULT_QUALITY_THRESHOLD
)

# ===== BACKWARD COMPATIBILITY =====

# Aliases for your existing code
VectorizedChunk = ModernVectorizedChunk
RetrievalResult = ModernRetrievalResult
DocumentInfo = ModernDocumentInfo
RAGAgent = ModernEnhancedRAGAgent
EnhancedRAGAgent = ModernEnhancedRAGAgent

# ===== VERSION INFO =====

def get_version_info():
    """Get comprehensive version and capability information"""
    return {
        'version': __version__,
        'author': __author__,
        'date': __date__,
        'current_user': 'Sagar4276',
        'current_time': '2025-06-13 09:40:12',
        'features': {
            'multimodal_support': True,
            'clip_integration': True,
            'ocr_capabilities': True,
            'hybrid_search': True,
            'faiss_optimization': True,
            'safwan_features': True,
            'your_architecture': True
        },
        'models_supported': {
            'text_embedding': 'SentenceTransformers (all-MiniLM-L6-v2)',
            'image_understanding': 'OpenAI CLIP (ViT-B/32)',
            'ocr_engines': ['EasyOCR', 'Tesseract'],
            'document_formats': ['PDF', 'TXT', 'MD', 'DOCX']
        },
        'search_capabilities': {
            'text_semantic_search': True,
            'image_visual_search': True,
            'hybrid_scoring': True,
            'quality_assessment': True,
            'context_extraction': True
        }
    }

def check_dependencies():
    """Check if all required dependencies are available"""
    dependencies = {
        'torch': False,
        'sentence_transformers': False,
        'faiss': False,
        'clip': False,
        'easyocr': False,
        'pytesseract': False,
        'fitz': False,  # PyMuPDF
        'PIL': False,
        'numpy': False
    }
    
    # Check core dependencies
    try:
        import torch
        dependencies['torch'] = True
    except ImportError:
        pass
    
    try:
        import sentence_transformers
        dependencies['sentence_transformers'] = True
    except ImportError:
        pass
    
    try:
        import faiss
        dependencies['faiss'] = True
    except ImportError:
        pass
    
    try:
        import clip
        dependencies['clip'] = True
    except ImportError:
        pass
    
    try:
        import easyocr
        dependencies['easyocr'] = True
    except ImportError:
        pass
    
    try:
        import pytesseract
        dependencies['pytesseract'] = True
    except ImportError:
        pass
    
    try:
        import fitz
        dependencies['fitz'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        dependencies['PIL'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        dependencies['numpy'] = True
    except ImportError:
        pass
    
    return dependencies

def print_system_info():
    """Print comprehensive system information"""
    print("🚀 ENHANCED MULTIMODAL RAG SYSTEM")
    print("=" * 50)
    
    version_info = get_version_info()
    print(f"📦 Version: {version_info['version']}")
    print(f"👤 Author: {version_info['author']}")
    print(f"📅 Date: {version_info['date']}")
    print(f"🕐 Current Time: {version_info['current_time']} UTC")
    print(f"👨‍💻 Current User: {version_info['current_user']}")
    
    print("\n🎯 Features:")
    for feature, enabled in version_info['features'].items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {feature.replace('_', ' ').title()}")
    
    print("\n🧠 AI Models:")
    for model_type, model_name in version_info['models_supported'].items():
        print(f"   📊 {model_type.replace('_', ' ').title()}: {model_name}")
    
    print("\n🔍 Search Capabilities:")
    for capability, enabled in version_info['search_capabilities'].items():
        status = "✅" if enabled else "❌"
        print(f"   {status} {capability.replace('_', ' ').title()}")
    
    # Check dependencies
    print("\n📦 Dependency Status:")
    deps = check_dependencies()
    for dep, available in deps.items():
        status = "✅" if available else "❌"
        print(f"   {status} {dep}")
    
    missing_deps = [dep for dep, available in deps.items() if not available]
    if missing_deps:
        print(f"\n⚠️ Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install -r requirements.txt")
    else:
        print("\n🎉 All dependencies available!")
    
    print("=" * 50)

# ===== QUICK START FUNCTION =====

def create_rag_agent(papers_folder: str = "knowledge_base/papers", 
                    enable_multimodal: bool = True):
    """
    Quick start function to create an enhanced RAG agent
    
    Args:
        papers_folder: Path to folder containing documents
        enable_multimodal: Enable CLIP-based image understanding
    
    Returns:
        ModernEnhancedRAGAgent: Ready-to-use RAG system
    """
    print(f"🚀 Creating Enhanced RAG Agent for {__author__}")
    print(f"📂 Papers folder: {papers_folder}")
    print(f"🖼️ Multimodal enabled: {enable_multimodal}")
    
    return ModernEnhancedRAGAgent(
        papers_folder=papers_folder,
        enable_multimodal=enable_multimodal
    )

# ===== MODULE EXPORTS =====

__all__ = [
    # Main classes
    'ModernEnhancedRAGAgent',
    'TextProcessor', 
    'ModernVectorizationEngine',
    'ModernRetrievalEngine',
    
    # Data structures
    'ModernVectorizedChunk',
    'ModernRetrievalResult', 
    'ModernDocumentInfo',
    'MultimodalImageData',
    'MultimodalSearchQuery',
    'MultimodalConfig',
    'SearchQuery',
    'SystemStats',
    'ProcessingResult',
    
    # Backward compatibility
    'VectorizedChunk',
    'RetrievalResult',
    'DocumentInfo', 
    'RAGAgent',
    'EnhancedRAGAgent',
    
    # Utility functions
    'create_empty_result',
    'create_multimodal_system_stats',
    'calculate_hybrid_score',
    'detect_scientific_terms',
    'format_file_size',
    'calculate_chunk_stats',
    'get_version_info',
    'check_dependencies',
    'print_system_info',
    'create_rag_agent',
    
    # Type hints
    'DocumentList',
    'ResultList', 
    'ChunkList',
    'ImageList',
    
    # Constants
    'DEFAULT_SIMILARITY_THRESHOLD',
    'DEFAULT_MAX_RESULTS',
    'DEFAULT_CHUNK_SIZE',
    'DEFAULT_CHUNK_OVERLAP',
    'DEFAULT_CLIP_THRESHOLD',
    'DEFAULT_QUALITY_THRESHOLD',
    
    # Module info
    '__version__',
    '__author__',
    '__date__'
]

# ===== MODULE INITIALIZATION =====

# Print welcome message when module is imported
print(f"📚 Enhanced Multimodal RAG v{__version__} loaded for {__author__}")
print(f"🕐 {__date__} | Current time: 2025-06-13 09:40:12 UTC")
print("🎯 Ready for text + image understanding!")

# Verify critical dependencies on import
critical_deps = check_dependencies()
missing_critical = [dep for dep in ['torch', 'sentence_transformers', 'faiss', 'numpy'] 
                   if not critical_deps.get(dep, False)]

if missing_critical:
    print(f"⚠️ Warning: Missing critical dependencies: {', '.join(missing_critical)}")
    print("Install with: pip install torch sentence-transformers faiss-cpu numpy")
else:
    print("✅ All critical dependencies available!")

# Check multimodal dependencies
multimodal_deps = ['clip', 'easyocr', 'fitz', 'PIL']
missing_multimodal = [dep for dep in multimodal_deps 
                     if not critical_deps.get(dep, False)]

if missing_multimodal:
    print(f"🖼️ Multimodal features need: {', '.join(missing_multimodal)}")
    print("Install with: pip install clip-by-openai easyocr PyMuPDF Pillow")