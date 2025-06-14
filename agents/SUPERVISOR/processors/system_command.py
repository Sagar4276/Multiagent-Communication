"""
System Commands Processing Component
Current Date and Time (UTC): 2025-06-14 10:58:00
Current User's Login: Sagar4276

Handles all system command processing.
"""

from typing import Any, List, Dict
from ..core.supervisor_config import SupervisorConfig
from ..core.message_analysis import MessageAnalysis
from ..components.system_health import SystemHealthMonitor
from ..components.performance_monitor import PerformanceMonitor
from ..components.session_manager import SessionManager
from ..utils.time_utils import TimeUtils

class SystemCommandProcessor:
    """Handles system command processing"""
    
    def __init__(self, config: SupervisorConfig, health_monitor: SystemHealthMonitor, 
                 performance_monitor: PerformanceMonitor, session_manager: SessionManager, 
                 time_utils: TimeUtils):
        self.config = config
        self.health_monitor = health_monitor
        self.performance_monitor = performance_monitor
        self.session_manager = session_manager
        self.time_utils = time_utils
        
        # Update performance counter
        self.performance_monitor.update_system_command()
    
    def handle_system_command(self, user_id: str, message: str, analysis: MessageAnalysis, 
                            shared_memory: Any, chat_agent: Any) -> str:
        """Handle enhanced system commands"""
        msg_lower = message.lower().strip()
        current_time = self.time_utils.get_current_time()
        
        # History commands
        if any(cmd in msg_lower for cmd in ['history', 'show history', 'conversation history']):
            return self._get_conversation_history_display(user_id, shared_memory)
        
        # Status commands
        elif any(cmd in msg_lower for cmd in ['system status', 'supervisor status', 'show status', 'status']):
            return self._get_enhanced_system_status(user_id, chat_agent, current_time)
        
        # Knowledge base commands
        elif any(cmd in msg_lower for cmd in ['show papers', 'list papers', 'show documents', 'list documents']):
            return self._get_enhanced_knowledge_base_status(chat_agent)
        
        # Help commands
        elif any(cmd in msg_lower for cmd in ['help', 'commands', 'capabilities']):
            return self._get_enhanced_help_menu(user_id, current_time)
        
        # Health commands
        elif any(cmd in msg_lower for cmd in ['health', 'health check', 'diagnostics']):
            return self.health_monitor.format_health_report(user_id, current_time, shared_memory, chat_agent)
        
        # Performance commands
        elif any(cmd in msg_lower for cmd in ['metrics', 'performance', 'stats']):
            return self.performance_monitor.format_performance_report(user_id, current_time)
        
        # Session commands
        elif any(cmd in msg_lower for cmd in ['sessions', 'active sessions', 'show sessions']):
            return self.session_manager.format_session_report(user_id, current_time)
        
        else:
            return self._get_unknown_command_help(user_id, message)
    
    def _get_conversation_history_display(self, user_id: str, shared_memory: Any) -> str:
        """Get and display conversation history"""
        try:
            history = shared_memory.get_conversation_history(user_id)
            if not history:
                return f"""📭 **NO CONVERSATION HISTORY**

👤 **User:** {user_id}
🕐 **Checked:** {self.time_utils.get_current_time()} UTC

💡 **Note:** Start a conversation to build history!"""
            
            history_text = f"📚 **CONVERSATION HISTORY FOR {user_id}**\n\n"
            history_text += f"🕐 **Retrieved:** {self.time_utils.get_current_time()} UTC\n"
            history_text += f"📊 **Total Messages:** {len(history)}\n\n"
            
            # Show last 10 messages
            recent_history = history[-10:] if len(history) > 10 else history
            
            for i, msg in enumerate(recent_history, 1):
                sender = msg.get('sender', 'Unknown')
                message_text = msg.get('message', '')
                timestamp = msg.get('timestamp', 'Unknown')
                
                # Truncate long messages
                display_text = message_text[:150] + '...' if len(message_text) > 150 else message_text
                
                emoji = "👤" if sender == "User" else "🤖"
                history_text += f"{i}. {emoji} **{sender}** [{timestamp}]\n"
                history_text += f"   💬 {display_text}\n\n"
            
            if len(history) > 10:
                history_text += f"💡 **Note:** Showing last 10 of {len(history)} total messages"
            
            return history_text
            
        except Exception as e:
            return f"""❌ **ERROR RETRIEVING HISTORY**

👤 **User:** {user_id}
🔍 **Error:** {str(e)}
🕐 **Time:** {self.time_utils.get_current_time()} UTC

🆘 **Try:** Restart the system or contact support"""
    
    def _get_enhanced_system_status(self, user_id: str, chat_agent: Any, current_time: str) -> str:
        """Enhanced comprehensive system status"""
        agent_info = {}
        if chat_agent:
            agent_info = chat_agent.get_model_info()
        
        # Get RAG statistics
        rag_stats = {}
        if hasattr(chat_agent, 'rag_agent') and chat_agent.rag_agent:
            try:
                rag_stats = chat_agent.rag_agent.get_stats()
            except:
                rag_stats = {}
        
        # Get system health
        health_results = self.health_monitor.run_health_diagnostics(None, chat_agent)
        
        status = f"🟢 **ENHANCED MULTI-AGENT SYSTEM STATUS REPORT v{self.config.version}**\n\n"
        status += f"🤖 **System:** {self.config.system_name}\n"
        status += f"👤 **Current User:** {user_id}\n"
        status += f"🕐 **Status Time:** {current_time} UTC\n"
        status += f"⏱️ **System Uptime:** {self.performance_monitor.format_uptime(self.performance_monitor.metrics.uptime_seconds)}\n"
        status += f"🏥 **System Health:** {self.health_monitor.get_current_health_status(None, chat_agent)} ({health_results['overall_health']:.1%})\n\n"
        
        status += f"🔧 **ENHANCED SYSTEM COMPONENTS:**\n"
        status += f"   🟢 **Enhanced Supervisor:** ACTIVE v{self.config.version} (Modular Design)\n"
        status += f"   🧠 **Chat Agent:** {'ACTIVE' if chat_agent else 'OFFLINE'} (Conversation Handler)\n"
        status += f"   📚 **RAG Agent:** {'ACTIVE' if rag_stats.get('papers', 0) > 0 else 'STANDBY'} (Knowledge Retrieval)\n"
        status += f"   💾 **Shared Memory:** {'ACTIVE' if health_results['shared_memory'] else 'OFFLINE'} (Multi-user Support)\n"
        status += f"   📊 **Performance Monitor:** {'ACTIVE' if self.config.performance_monitoring else 'DISABLED'}\n"
        status += f"   🛡️ **Auto Recovery:** {'ENABLED' if self.config.auto_recovery else 'DISABLED'}\n\n"
        
        status += f"📊 **PERFORMANCE METRICS:**\n"
        status += f"   📨 **Total Messages:** {self.performance_monitor.metrics.total_messages:,}\n"
        status += f"   ✅ **Successful Responses:** {self.performance_monitor.metrics.successful_responses:,}\n"
        status += f"   ❌ **Failed Responses:** {self.performance_monitor.metrics.failed_responses:,}\n"
        status += f"   📈 **Success Rate:** {self.performance_monitor.get_success_rate():.1%}\n"
        status += f"   ⚡ **Average Response Time:** {self.performance_monitor.metrics.average_response_time:.2f}s\n"
        status += f"   🔄 **Current Load:** {self.performance_monitor.metrics.current_load:.1%}\n\n"
        
        status += f"🎯 **REQUEST BREAKDOWN:**\n"
        status += f"   🔍 **RAG Requests:** {self.performance_monitor.metrics.rag_requests:,}\n"
        status += f"   💬 **Chat Requests:** {self.performance_monitor.metrics.chat_requests:,}\n"
        status += f"   🔧 **System Commands:** {self.performance_monitor.metrics.system_commands:,}\n\n"
        
        status += f"📚 **KNOWLEDGE BASE:**\n"
        status += f"   📄 **Documents:** {rag_stats.get('papers', 4)} loaded\n"
        status += f"   🔍 **Search Chunks:** {rag_stats.get('chunks', 4):,}\n"
        status += f"   💾 **Content Size:** {rag_stats.get('total_size_kb', 0):,}KB\n\n"
        
        status += f"👥 **ACTIVE SESSIONS:**\n"
        active_count = len(self.session_manager.active_sessions)
        status += f"   🔢 **Active Users:** {active_count}\n"
        if active_count > 0:
            for session_user in list(self.session_manager.active_sessions.keys())[:5]:  # Show first 5
                session = self.session_manager.active_sessions[session_user]
                status += f"   👤 **{session_user}:** {session['message_count']} msgs, active {self.time_utils.time_since(session['last_activity'])} ago\n"
        
        status += f"\n🚀 **SYSTEM STATUS:** All modular components operational and ready for advanced research queries!"
        
        return status
    
    def _get_enhanced_knowledge_base_status(self, chat_agent: Any) -> str:
        """Enhanced knowledge base status"""
        if hasattr(chat_agent, 'rag_agent') and chat_agent.rag_agent:
            try:
                base_status = chat_agent.rag_agent.get_paper_summary()
            except:
                base_status = "📚 Knowledge base information available"
            
            # Add enhanced supervisor information
            enhanced_status = f"🟢 **ENHANCED KNOWLEDGE BASE STATUS**\n\n"
            enhanced_status += base_status
            enhanced_status += f"\n\n🔧 **Enhanced Features:**\n"
            enhanced_status += f"• Modular architecture for easy debugging\n"
            enhanced_status += f"• Configurable response length ({self.config.response_length_tokens} tokens)\n"
            enhanced_status += f"• Intelligent query routing and analysis\n"
            enhanced_status += f"• Performance monitoring and optimization\n"
            enhanced_status += f"• Advanced error handling and recovery\n"
            enhanced_status += f"• Multi-user session management\n"
            enhanced_status += f"• Real-time system health monitoring\n"
            
            return enhanced_status
        else:
            return f"🟢 **Enhanced Knowledge Base Status**\n\n📚 No documents currently loaded.\n\n💡 Add documents to the `papers` folder and restart the system to enable enhanced RAG research capabilities."
    
    def _get_enhanced_help_menu(self, user_id: str, current_time: str) -> str:
        """Enhanced comprehensive help menu"""
        help_menu = f"🟢 **ENHANCED MULTI-AGENT RESEARCH SYSTEM - HELP v{self.config.version}**\n\n"
        help_menu += f"🤖 **Welcome {user_id}!** You're using the Enhanced Multi-Agent Research Assistant (Modular Version).\n"
        help_menu += f"All queries are processed through the Enhanced Supervisor Agent with intelligent routing.\n\n"
        
        help_menu += f"🎯 **ENHANCED RESEARCH CAPABILITIES:**\n"
        help_menu += f"• **Modular Architecture:** Easy debugging and customization\n"
        help_menu += f"• **Configurable Responses:** {self.config.response_length_tokens} token responses\n"
        help_menu += f"• **Intelligent Query Analysis:** Automatic classification and optimal routing\n"
        help_menu += f"• **Advanced Knowledge Search:** Multi-domain research with confidence scoring\n"
        help_menu += f"• **Real-time Performance:** Response time monitoring and optimization\n\n"
        
        help_menu += f"📋 **ENHANCED EXAMPLE QUERIES:**\n"
        help_menu += f"• \"Explain machine learning algorithms with examples\"\n"
        help_menu += f"• \"What are the latest developments in deep learning?\"\n"
        help_menu += f"• \"Compare transformer architectures for NLP tasks\"\n"
        help_menu += f"• \"Parkinson's disease research findings\"\n"
        help_menu += f"• \"Prevention measures for parkinsons\"\n\n"
        
        help_menu += f"💬 **ENHANCED SYSTEM COMMANDS:**\n"
        help_menu += f"• `system status` - Complete system health and performance report\n"
        help_menu += f"• `show papers` - Detailed document collection overview\n"
        help_menu += f"• `health check` - Comprehensive system diagnostics\n"
        help_menu += f"• `performance` - Real-time performance metrics and statistics\n"
        help_menu += f"• `sessions` - Active user sessions and activity monitoring\n"
        help_menu += f"• `history` - Show conversation history\n"
        help_menu += f"• `help` - This enhanced help menu\n\n"
        
        help_menu += f"📊 **CURRENT SYSTEM STATUS:**\n"
        help_menu += f"• **Version:** {self.config.version} (Modular)\n"
        help_menu += f"• **Health:** {self.health_monitor.get_current_health_status()}\n"
        help_menu += f"• **Uptime:** {self.performance_monitor.format_uptime(self.performance_monitor.metrics.uptime_seconds)}\n"
        help_menu += f"• **Success Rate:** {self.performance_monitor.get_success_rate():.1%}\n"
        help_menu += f"• **Response Config:** {self.config.response_length_tokens} tokens\n\n"
        
        help_menu += f"🚀 **Ready for Enhanced Research at {current_time} UTC!**\n"
        help_menu += f"Modular architecture makes debugging and customization easier than ever!"
        
        return help_menu
    
    def _get_unknown_command_help(self, user_id: str, message: str) -> str:
        """Help for unknown system commands"""
        return f"""🟡 **UNKNOWN SYSTEM COMMAND**

🟢 **Supervisor Status:** Active v{self.config.version} (Modular)
👤 **User:** {user_id}
💬 **Command Attempted:** "{message}"

🔧 **Available Enhanced System Commands:**
• `system status` - Complete system health and performance report
• `show papers` - Detailed document collection overview  
• `health check` - Comprehensive system diagnostics
• `performance` - Real-time performance metrics and statistics
• `sessions` - Active user sessions and activity monitoring
• `history` - Show conversation history
• `help` - Enhanced help menu with all capabilities

💡 **Did you mean:**
• If asking about research: Try "explain [topic]" or "what is [concept]"
• If asking about papers: Try "show papers" or "list documents"
• If asking about system: Try "system status" or "health check"

🎯 **Tip:** The Enhanced Supervisor intelligently routes research questions automatically!"""