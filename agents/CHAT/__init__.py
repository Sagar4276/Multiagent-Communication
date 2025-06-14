"""
Enhanced Chat Agent Package - FIXED IMPORTS
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-12 12:40:02
Current User's Login: Sagar4276
"""

# Import all components with ORIGINAL NAMES (no breaking changes)
from .chat_agent import ChatAgent

# Import sub-modules
from .models import ModelLoader
from .processors import QueryAnalyzer, RAGProcessor, ResponseFormatter  
from .generators import LLMGenerator, StructuredGenerator

# Export everything with original names
__all__ = [
    'ChatAgent',           # Main class (original name)
    'ModelLoader',         # Model loading
    'QueryAnalyzer',       # Query analysis
    'RAGProcessor',        # RAG processing
    'ResponseFormatter',   # Response formatting
    'LLMGenerator',        # LLM generation
    'StructuredGenerator'  # Structured responses
]

# Version info
__version__ = "2.0.0"
__description__ = "Enhanced Chat Agent with Smart RAG Detection"