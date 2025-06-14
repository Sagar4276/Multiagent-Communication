"""
Chat Processing Component
Current Date and Time (UTC): 2025-06-14 10:58:00
Current User's Login: Sagar4276

Handles general chat processing and response enhancement.
"""

from typing import Any
from ..core.supervisor_config import SupervisorConfig
from ..core.message_analysis import MessageAnalysis
from ..utils.response_formatter import ResponseFormatter
from ..utils.display_utils import Colors

class ChatProcessor:
    """Handles chat system processing and response enhancement"""
    
    def __init__(self, config: SupervisorConfig, response_formatter: ResponseFormatter):
        self.config = config
        self.response_formatter = response_formatter
    
    def process_chat_request(self, user_id: str, message: str, analysis: MessageAnalysis, chat_agent: Any) -> str:
        """Process general chat request"""
        
        try:
            if not chat_agent:
                return self._handle_missing_chat_agent(user_id, message)
            
            # Process with monitoring
            import time
            start_time = time.time()
            response = chat_agent.process_message(user_id, message)
            processing_time = time.time() - start_time
            
            # Light enhancement for chat responses
            enhanced_response = self.response_formatter.enhance_chat_response(
                response, user_id, analysis, processing_time
            )
            
            print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ✅ Chat processing completed in {processing_time:.2f}s{Colors.END}")
            return enhanced_response
            
        except Exception as e:
            print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ Chat processing error: {str(e)}{Colors.END}")
            return self._format_chat_error_response(user_id, message, analysis, str(e))
    
    def _handle_missing_chat_agent(self, user_id: str, message: str) -> str:
        """Handle missing chat agent"""
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
    
    def _format_chat_error_response(self, user_id: str, message: str, analysis: MessageAnalysis, error: str) -> str:
        """Format chat error response"""
        import time
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