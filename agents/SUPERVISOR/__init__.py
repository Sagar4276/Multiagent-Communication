"""
SUPERVISOR Package - Main Entry Point
Current Date and Time (UTC): 2025-06-14 18:04:14
Current User's Login: Sagar4276

Main supervisor package with core functionality.
"""

from .supervisor_agent import EnhancedSupervisorAgent
from .core.enhanced_supervisor import EnhancedSupervisorCore as CoreSupervisorAgent
from .core.message_analysis import MessageAnalyzer, MessageType, Priority, MessageAnalysis
from .core.supervisor_config import SupervisorConfig

# Import components
from .components.error_handler import ErrorHandler
from .components.performance_monitor import PerformanceMonitor
from .components.session_manager import SessionManager
from .components.system_health import SystemHealthMonitor

# Import processors
from .processors.chat_processor import ChatProcessor
from .processors.rag_processor import RAGProcessor
from .processors.system_command import SystemCommandProcessor

# Import utils
from .utils.display_utils import Colors
from .utils.response_formatter import ResponseFormatter
from .utils.time_utils import TimeUtils

__version__ = "2.0.0-Enhanced"
__all__ = [
    # Main supervisor
    'EnhancedSupervisorAgent',
    'CoreSupervisorAgent',
    
    # Core components
    'MessageAnalyzer', 'MessageType', 'Priority', 'MessageAnalysis',
    'SupervisorConfig',
    
    # Components
    'ErrorHandler', 'PerformanceMonitor', 'SessionManager', 'SystemHealthMonitor',
    
    # Processors  
    'ChatProcessor', 'RAGProcessor', 'SystemCommandProcessor',
    
    # Utils
    'Colors', 'ResponseFormatter', 'TimeUtils'
]

print(f"📦 [SUPERVISOR] Package loaded v{__version__}")