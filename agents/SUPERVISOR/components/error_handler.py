"""
Error Handling Component
Current Date and Time (UTC): 2025-06-14 10:52:09
Current User's Login: Sagar4276

Handles error processing, recovery, and reporting.
"""

import time
import traceback
from typing import Optional
from ..core.supervisor_config import SupervisorConfig
from ..core.message_analysis import MessageAnalysis
from ..utils.display_utils import Colors

class ErrorHandler:
    """Handles error processing, recovery, and reporting"""
    
    def __init__(self, config: SupervisorConfig):
        self.config = config
    
    def handle_uninitialized_request(self, user_id: str, user_message: str) -> str:
        """Handle requests when system is not properly initialized"""
        return f"""🔴 **SYSTEM INITIALIZATION ERROR**

🟢 **Supervisor Status:** Partially Initialized
❌ **Error:** System failed to initialize properly
👤 **User:** {user_id}
💬 **Query:** "{user_message}"

🆘 **Actions:**
• Restart the system
• Check system requirements
• Verify shared memory and chat agent setup
• Contact system administrator if problem persists

⚠️ **Note:** Some basic functions may still be available."""
    
    def handle_processing_error(self, user_id: str, user_message: str, error: Exception, response_time: float) -> str:
        """Enhanced error handling and recovery"""
        error_id = f"ERR_{int(time.time())}"
        
        # Log error details
        print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ PROCESSING ERROR {error_id}{Colors.END}")
        print(f"{Colors.RED}[ENHANCED SUPERVISOR] 👤 User: {user_id}{Colors.END}")
        print(f"{Colors.RED}[ENHANCED SUPERVISOR] ⏱️ Response Time: {response_time:.2f}s{Colors.END}")
        print(f"{Colors.RED}[ENHANCED SUPERVISOR] 🔍 Error: {str(error)}{Colors.END}")
        
        if self.config.enable_detailed_logging:
            print(f"{Colors.RED}[ENHANCED SUPERVISOR] 📜 Stack Trace:{Colors.END}")
            traceback.print_exc()
        
        # Attempt recovery if enabled
        recovery_response = None
        if self.config.auto_recovery:
            recovery_response = self._attempt_error_recovery(user_id, user_message, error)
            if recovery_response:
                return recovery_response
        
        # Format error response
        return f"""🔴 **ENHANCED SUPERVISOR - PROCESSING ERROR**

🟢 **Supervisor Status:** Active (v{self.config.version})
❌ **Error ID:** {error_id}
💬 **Your Query:** "{user_message[:100]}{'...' if len(user_message) > 100 else ''}"
👤 **User:** {user_id}
⏱️ **Processing Time:** {response_time:.2f}s

🔧 **Auto-Recovery:** {'Attempted' if self.config.auto_recovery else 'Disabled'}
📊 **System Health:** Available via `health check`

🆘 **Troubleshooting:**
• Try rephrasing your query
• Use `system status` to check component health
• Use `help` for available commands
• Try `health check` for detailed diagnostics

💡 **Error Details:** {str(error)[:200]}{'...' if len(str(error)) > 200 else ''}

🔄 **Next Steps:** The system is still operational for other queries."""
    
    def _attempt_error_recovery(self, user_id: str, user_message: str, error: Exception) -> Optional[str]:
        """Attempt to recover from processing errors"""
        try:
            print(f"{Colors.YELLOW}[ENHANCED SUPERVISOR] 🔄 Attempting error recovery...{Colors.END}")
            
            # Simple recovery: try to provide a basic response
            error_str = str(error).lower()
            
            if "timeout" in error_str:
                return f"🟡 **Recovery Mode:** Your query timed out, but I'm still here! Try a simpler version of: '{user_message[:50]}...'"
            
            elif "memory" in error_str:
                return f"🟡 **Recovery Mode:** Memory issue detected. Try `system status` to check system health, or ask a simpler question."
            
            elif "connection" in error_str or "network" in error_str:
                return f"🟡 **Recovery Mode:** Connection issue detected. The system is offline-capable, so this might be a temporary issue."
            
            else:
                return f"🟡 **Recovery Mode:** I encountered an issue but I'm still operational! Try rephrasing your question or use `help` for guidance."
            
        except Exception as recovery_error:
            print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ Recovery also failed: {str(recovery_error)}{Colors.END}")
            return None
    
    def handle_initialization_error(self, error: Exception):
        """Handle initialization errors"""
        print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ SYSTEM INITIALIZATION FAILED{Colors.END}")
        print(f"{Colors.RED}[ENHANCED SUPERVISOR] 🔍 Error: {str(error)}{Colors.END}")
        
        if self.config.enable_detailed_logging:
            traceback.print_exc()
    
    def handle_missing_chat_agent(self, user_id: str, message: str) -> str:
        """Handle requests when chat agent is not available"""
        return f"""🔴 **CHAT AGENT UNAVAILABLE**

🟢 **Supervisor Status:** Active v{self.config.version}
❌ **Chat Agent:** Not Available
👤 **User:** {user_id}
💬 **Query:** "{message[:100]}{'...' if len(message) > 100 else ''}"

🔧 **Available Actions:**
• Use `system status` to check component health
• Use `health check` for detailed diagnostics  
• Try `help` for available commands
• Restart the system if issues persist

⚠️ **Note:** The Enhanced Supervisor can still handle system commands directly."""
    
    def format_rag_error_response(self, user_id: str, message: str, analysis: MessageAnalysis, error: str) -> str:
        """Enhanced error formatting for RAG failures"""
        error_id = f"RAG_ERR_{int(time.time())}"
        
        return f"""🔴 **ENHANCED RAG PROCESSING ERROR**

🟢 **Supervisor Status:** Active v{self.config.version}
❌ **Error in:** Enhanced RAG Research System
💬 **Your Query:** "{message[:100]}{'...' if len(message) > 100 else ''}"
👤 **User:** {user_id}
🆔 **Error ID:** {error_id}

🎯 **Query Analysis:**
• **Type:** {analysis.type.value.replace('_', ' ').title()}
• **Priority:** {analysis.priority.value.title()}
• **Confidence:** {analysis.confidence:.1%}
• **Keywords:** {', '.join(analysis.keywords[:5])}

🔧 **Enhanced Troubleshooting:**
• Check document availability (`show papers`)
• Try rephrasing with different keywords
• Use `health check` for system diagnostics
• Use `performance` to check system load
• Try a simpler version of your research question

🆘 **Error Details:** {error[:200]}{'...' if len(error) > 200 else ''}

💡 **Recovery Options:**
• System commands still work (`system status`, `help`)
• Try general conversation (will route to chat system)
• The Enhanced Supervisor remains fully operational"""
    
    def format_chat_error_response(self, user_id: str, message: str, analysis: MessageAnalysis, error: str) -> str:
        """Enhanced error formatting for chat failures"""
        error_id = f"CHAT_ERR_{int(time.time())}"
        
        return f"""🔴 **ENHANCED CHAT PROCESSING ERROR**

🟢 **Supervisor Status:** Active v{self.config.version}
❌ **Error in:** Enhanced Chat System
💬 **Your Message:** "{message[:100]}{'...' if len(message) > 100 else ''}"
👤 **User:** {user_id}
🆔 **Error ID:** {error_id}

🎯 **Message Analysis:**
• **Type:** {analysis.type.value.replace('_', ' ').title()}
• **Confidence:** {analysis.confidence:.1%}

🔧 **Enhanced Fallback Response:**
I'm still here as your Enhanced Supervisor Agent! While the chat system has an issue, I can help with:
• Research queries (try academic or technical terms)
• System status and health checks (`system status`, `health check`)
• Performance monitoring (`performance`, `sessions`)
• Knowledge base reviews (`show papers`)
• Comprehensive diagnostics and help (`help`)

🆘 **Error Details:** {error[:200]}{'...' if len(error) > 200 else ''}

🔄 **Auto-Recovery:** {'Attempted' if self.config.auto_recovery else 'Disabled'}"""