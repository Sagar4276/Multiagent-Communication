"""
RAG Processing Component
Current Date and Time (UTC): 2025-06-14 10:52:09
Current User's Login: Sagar4276

Handles RAG system processing and response enhancement.
"""

import time
from typing import Any
from ..core.supervisor_config import SupervisorConfig
from ..core.message_analysis import MessageAnalysis
from ..utils.response_formatter import ResponseFormatter
from ..utils.time_utils import TimeUtils
from ..utils.display_utils import Colors

class RAGProcessor:
    """Handles RAG system processing and response enhancement"""
    
    def __init__(self, config: SupervisorConfig, response_formatter: ResponseFormatter, time_utils: TimeUtils):
        self.config = config
        self.response_formatter = response_formatter
        self.time_utils = time_utils
    
    def process_rag_request(self, user_id: str, message: str, analysis: MessageAnalysis, 
                          current_time: str, shared_memory: Any, chat_agent: Any) -> str:
        """Process RAG request with enhanced monitoring"""
        
        # Set processing flags
        shared_memory.set_flag(f'enhanced_rag_request_{user_id}', True)
        shared_memory.set_flag(f'research_mode', True)
        
        # Store enhanced context
        rag_context = {
            'message': message,
            'analysis': analysis.__dict__,
            'timestamp': time.time(),
            'user_id': user_id,
            'supervisor_version': self.config.version,
            'enhanced_processing': True,
            'response_config': {
                'length_tokens': self.config.response_length_tokens,
                'temperature': self.config.response_temperature,
                'repetition_penalty': self.config.response_repetition_penalty
            }
        }
        shared_memory.store_temp_data(f'enhanced_rag_context_{user_id}', rag_context)
        
        try:
            if not chat_agent:
                return self._handle_missing_chat_agent(user_id, message)
            
            # Force RAG mode for research queries
            if hasattr(chat_agent, 'force_rag_mode') and self.config.force_rag_for_research:
                chat_agent.force_rag_mode = True
            
            # Apply response length configuration to chat agent
            self._configure_chat_agent_response_length(chat_agent)
            
            # Process with enhanced monitoring
            start_time = time.time()
            response = chat_agent.process_message(user_id, message)
            processing_time = time.time() - start_time
            
            # Enhanced response formatting
            enhanced_response = self.response_formatter.enhance_rag_response(
                response, user_id, analysis, processing_time, current_time
            )
            
            print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ✅ RAG processing completed in {processing_time:.2f}s{Colors.END}")
            return enhanced_response
            
        except Exception as e:
            print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ RAG processing error: {str(e)}{Colors.END}")
            return self._format_rag_error_response(user_id, message, analysis, str(e))
        
        finally:
            # Clean up enhanced flags
            shared_memory.set_flag(f'enhanced_rag_request_{user_id}', False)
            shared_memory.set_flag(f'research_mode', False)
            shared_memory.clear_temp_data(f'enhanced_rag_context_{user_id}')
            
            # Reset RAG mode
            if hasattr(chat_agent, 'force_rag_mode'):
                chat_agent.force_rag_mode = False
    
    def _configure_chat_agent_response_length(self, chat_agent: Any):
        """Configure chat agent for longer responses based on config"""
        try:
            # Try to configure response length if chat agent supports it
            if hasattr(chat_agent, 'set_response_config'):
                chat_agent.set_response_config({
                    'max_tokens': self.config.response_length_tokens,
                    'temperature': self.config.response_temperature,
                    'repetition_penalty': self.config.response_repetition_penalty
                })
            
            # Alternative: Try to access LLM directly
            elif hasattr(chat_agent, 'llm') and hasattr(chat_agent.llm, 'generation_config'):
                chat_agent.llm.generation_config.update({
                    'max_new_tokens': self.config.response_length_tokens,
                    'temperature': self.config.response_temperature,
                    'repetition_penalty': self.config.response_repetition_penalty
                })
            
            # Print configuration info
            print(f"{Colors.CYAN}[RAG_PROCESSOR] 🔧 Configured for {self.config.response_length_tokens} token responses{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.YELLOW}[RAG_PROCESSOR] ⚠️ Could not configure response length: {str(e)}{Colors.END}")
    
    def _handle_missing_chat_agent(self, user_id: str, message: str) -> str:
        """Handle missing chat agent"""
        return f"""🔴 **CHAT AGENT UNAVAILABLE FOR RAG PROCESSING**

🟢 **Supervisor Status:** Active v{self.config.version}
❌ **RAG System:** Chat Agent Not Available
👤 **User:** {user_id}
💬 **Query:** "{message[:100]}{'...' if len(message) > 100 else ''}"

🔧 **Available Actions:**
• Use `system status` to check component health
• Use `health check` for detailed diagnostics  
• Try `help` for available commands
• Restart the system if issues persist

⚠️ **Note:** Research queries require the chat agent to access the knowledge base."""
    
    def _format_rag_error_response(self, user_id: str, message: str, analysis: MessageAnalysis, error: str) -> str:
        """Format RAG error response"""
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
• Try a simpler version of your research question

🆘 **Error Details:** {error[:200]}{'...' if len(error) > 200 else ''}

💡 **Recovery Options:**
• System commands still work (`system status`, `help`)
• The Enhanced Supervisor remains fully operational"""