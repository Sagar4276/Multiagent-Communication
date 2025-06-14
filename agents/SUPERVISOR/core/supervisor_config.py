"""
Supervisor Configuration Management
Current Date and Time (UTC): 2025-06-14 10:43:05
Current User's Login: Sagar4276

Centralized configuration for easy tweaking of system parameters.
"""

from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class SupervisorConfig:
    """Centralized supervisor configuration"""
    
    # Performance settings
    max_response_time: float = 30.0
    max_memory_usage_mb: int = 1024
    max_response_history: int = 100
    
    # Response generation settings - EASY TO MODIFY FOR LONGER RESPONSES
    response_length_tokens: int = 150  # ← INCREASE THIS FOR LONGER RESPONSES
    response_temperature: float = 0.7
    response_repetition_penalty: float = 1.1
    
    # Session management
    session_timeout_minutes: int = 30
    max_concurrent_sessions: int = 10
    
    # Feature toggles
    enable_detailed_logging: bool = True
    auto_recovery: bool = True
    performance_monitoring: bool = True
    enhanced_routing: bool = True
    error_reporting: bool = True
    backup_chat_agent: bool = True
    
    # Health monitoring
    health_check_interval: int = 60
    memory_warning_threshold: float = 80.0
    memory_critical_threshold: float = 95.0
    
    # RAG settings
    rag_confidence_threshold: float = 0.8
    research_query_confidence: float = 0.9
    force_rag_for_research: bool = True
    
    # System info
    version: str = "2.0.0"
    system_name: str = "Multi-Agent RAG Research System v2.0"
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration as dictionary"""
        return {
            'max_response_time': 30.0,
            'max_memory_usage_mb': 1024,
            'enable_detailed_logging': True,
            'auto_recovery': True,
            'session_timeout_minutes': 30,
            'max_concurrent_sessions': 10,
            'performance_monitoring': True,
            'enhanced_routing': True,
            'error_reporting': True,
            'health_check_interval': 60,
            'backup_chat_agent': True,
            # RESPONSE SETTINGS - EASY TO MODIFY
            'response_length_tokens': 150,  # ← CHANGE THIS
            'response_temperature': 0.7,
            'response_repetition_penalty': 1.1,
            'force_rag_for_research': True
        }
    
    @classmethod
    def create_high_quality_config(cls) -> 'SupervisorConfig':
        """Create configuration optimized for detailed responses"""
        return cls(
            response_length_tokens=300,  # Much longer responses
            response_temperature=0.7,
            response_repetition_penalty=1.1,
            max_response_time=45.0,  # Allow more time for detailed responses
            rag_confidence_threshold=0.7,  # Lower threshold for more RAG usage
            force_rag_for_research=True,
            enable_detailed_logging=True
        )