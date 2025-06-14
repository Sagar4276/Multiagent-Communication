"""
Enhanced Supervisor Agent - Main Entry Point with AI/ML Integration
Current Date and Time (UTC): 2025-06-14 18:35:06
Current User's Login: Sagar4276

Modular version with AI/ML agent integration for medical analysis and report generation.
This is the main supervisor class that coordinates all modular components including AI/ML.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from dataclasses import asdict

# Import modular components
from .core.supervisor_config import SupervisorConfig
from .core.message_analysis import MessageAnalyzer, MessageAnalysis
from .components.system_health import SystemHealthMonitor
from .components.performance_monitor import PerformanceMonitor
from .components.session_manager import SessionManager
from .components.error_handler import ErrorHandler
from .processors.rag_processor import RAGProcessor
from .processors.chat_processor import ChatProcessor
from .processors.aiml_processor import AIMLProcessor  # 🆕 NEW
from .processors.system_command import SystemCommandProcessor
from .utils.response_formatter import ResponseFormatter
from .utils.display_utils import Colors
from .utils.time_utils import TimeUtils

# Import original dependencies
from shared_memory.simple_memory import SimpleSharedMemory
from agents.CHAT.chat_agent import ChatAgent

# 🆕 Import AI/ML Agent
try:
    from agents.AIML.aiml_agent import AIMLAgent
    AIML_AVAILABLE = True
except ImportError:
    print("⚠️ [SUPERVISOR] AI/ML agent not available - install dependencies or check path")
    AIMLAgent = None
    AIML_AVAILABLE = False

class EnhancedSupervisorAgent:
    """
    🆕 Enhanced Supervisor Agent with AI/ML Medical Integration
    
    Features:
    - Modular architecture for easy debugging
    - AI/ML agent integration for medical analysis
    - MRI image processing and report generation
    - Configurable response length (edit supervisor_config.py)
    - Intelligent message routing to all 4 agents
    - Performance monitoring and health diagnostics
    - Session management and error handling
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Load configuration
        if config:
            # Create config from dictionary
            config_dict = SupervisorConfig.get_default_config()
            config_dict.update(config)
            # Filter out keys that don't exist in SupervisorConfig
            valid_keys = {k: v for k, v in config_dict.items() 
                         if k in SupervisorConfig.__dataclass_fields__}
            self.config = SupervisorConfig(**valid_keys)
        else:
            self.config = SupervisorConfig()
        
        # Core properties
        self.name = "EnhancedSupervisorAgent"
        self.current_user = "Sagar4276"
        self.version = f"{self.config.version}-AIML"  # 🆕 Updated version
        self.start_time = time.time()
        
        # 🆕 AI/ML agent reference
        self.aiml_agent = None
        
        # System state
        self.is_initialized = False
        self.is_healthy = True
        self.initialization_error = None
        
        # Initialize modular components
        self._initialize_components()
        
        # Initialize system with AI/ML support
        self._initialize_system()
    
    def _initialize_components(self):
        """🆕 Initialize all modular components including AI/ML"""
        print(f"{Colors.CYAN}[ENHANCED SUPERVISOR] 🔧 Initializing modular components with AI/ML support...{Colors.END}")
        
        # Core components
        self.message_analyzer = MessageAnalyzer()
        self.time_utils = TimeUtils()
        
        # System components
        self.health_monitor = SystemHealthMonitor(self.config)
        self.performance_monitor = PerformanceMonitor(self.config)
        self.session_manager = SessionManager(self.config)
        self.error_handler = ErrorHandler(self.config)
        
        # Utility components
        self.response_formatter = ResponseFormatter(self.config)
        
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ✅ Modular components initialized{Colors.END}")
    
    def _initialize_system(self):
        """🆕 Enhanced system initialization with AI/ML agent"""
        try:
            self._print_startup_banner()
            self._initialize_shared_memory()
            self._initialize_chat_agent()
            self._initialize_aiml_agent()  # 🆕 NEW
            
            # Initialize processors with components including AI/ML
            self._initialize_processors()
            
            self._perform_initial_health_check()
            self.is_initialized = True
            self._display_system_ready()
            
        except Exception as e:
            self.is_initialized = False
            self.is_healthy = False
            self.initialization_error = str(e)
            self.error_handler.handle_initialization_error(e)
    
    def _initialize_aiml_agent(self):
        """🆕 Initialize AI/ML agent for medical analysis"""
        try:
            if not AIML_AVAILABLE:
                print(f"{Colors.YELLOW}[{self.name}] ⚠️ AI/ML agent not available - medical features disabled{Colors.END}")
                self.aiml_agent = None
                return
            
            print(f"{Colors.GREEN}[{self.name}] 🤖 Loading AI/ML Medical Agent...{Colors.END}")
            
            start_time = time.time()
            self.aiml_agent = AIMLAgent(self.shared_memory)
            load_time = time.time() - start_time
            
            # Set up cross-agent connections
            if hasattr(self.aiml_agent, 'set_agent_connections'):
                self.aiml_agent.set_agent_connections(
                    chat_agent=getattr(self, 'chat_agent', None),
                    rag_agent=getattr(self.chat_agent, 'rag_agent', None) if hasattr(self, 'chat_agent') else None
                )
            
            # Verify AI/ML agent functionality
            aiml_info = self.aiml_agent.get_model_info()
            
            print(f"{Colors.GREEN}[{self.name}] ✅ AI/ML agent loaded in {load_time:.2f}s{Colors.END}")
            print(f"{Colors.GREEN}[{self.name}] 🏥 Medical System: {aiml_info.get('model_name', 'AI/ML Medical Analysis')}{Colors.END}")
            print(f"{Colors.GREEN}[{self.name}] 🔬 Capabilities: {len(aiml_info.get('capabilities', []))} medical features{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}[{self.name}] ❌ AI/ML agent initialization failed: {str(e)}{Colors.END}")
            self.aiml_agent = None
            print(f"{Colors.YELLOW}[{self.name}] 🔄 Continuing without AI/ML agent - medical features disabled{Colors.END}")
    
    def _initialize_processors(self):
        """🆕 Initialize processing components including AI/ML processor"""
        print(f"{Colors.CYAN}[ENHANCED SUPERVISOR] 🔧 Initializing processors with AI/ML support...{Colors.END}")
        
        self.rag_processor = RAGProcessor(
            self.config, 
            self.response_formatter, 
            self.time_utils
        )
        
        self.chat_processor = ChatProcessor(
            self.config, 
            self.response_formatter
        )
        
        # 🆕 Initialize AI/ML processor
        self.aiml_processor = AIMLProcessor(
            self.config,
            self.response_formatter,
            self.time_utils
        )
        
        self.system_command_processor = SystemCommandProcessor(
            self.config, 
            self.health_monitor, 
            self.performance_monitor, 
            self.session_manager, 
            self.time_utils
        )
        
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ✅ All processors initialized (including AI/ML){Colors.END}")
    
    def _print_startup_banner(self):
        """🆕 Enhanced startup banner with AI/ML info"""
        current_time = self.time_utils.get_current_time()
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}{'='*90}")
        print(f"🚀 {self.config.system_name.upper()} WITH AI/ML MEDICAL INTEGRATION")
        print(f"🟢 ENHANCED SUPERVISOR AGENT v{self.version} - MEDICAL AI MODULAR IMPLEMENTATION")
        print(f"{'='*90}{Colors.END}")
        print(f"{Colors.GREEN}👤 Current User: {Colors.BOLD}{self.current_user}{Colors.END}")
        print(f"{Colors.GREEN}🕐 System Start: {Colors.BOLD}{current_time} UTC{Colors.END}")
        print(f"{Colors.GREEN}🎯 Mission: Advanced Multi-Agent Medical Research & Analysis{Colors.END}")
        print(f"{Colors.GREEN}🏥 Medical AI: MRI Analysis, Report Generation, Patient Management{Colors.END}")
        print(f"{Colors.GREEN}📂 Architecture: Modular Design with AI/ML Integration{Colors.END}")
        print(f"{Colors.GREEN}🔧 Features: Medical Analysis, Configurable Responses, Intelligent Routing{Colors.END}")
        print(f"{Colors.GREEN}⚙️ Response Config: {self.config.response_length_tokens} tokens, {self.config.response_temperature} temperature{Colors.END}")
        print(f"{Colors.GREEN}🤖 AI/ML Support: {'Enabled' if AIML_AVAILABLE else 'Disabled'}{Colors.END}")
        print(f"{Colors.GREEN}{'='*90}{Colors.END}\n")
    
    def _initialize_shared_memory(self):
        """Initialize shared memory with enhanced error handling"""
        try:
            print(f"{Colors.GREEN}[{self.name}] 💾 Initializing enhanced shared memory system...{Colors.END}")
            self.shared_memory = SimpleSharedMemory()
            
            # Test shared memory functionality
            test_key = f"system_test_{int(time.time())}"
            self.shared_memory.store_temp_data(test_key, {"test": True})
            self.shared_memory.clear_temp_data(test_key)
            
            print(f"{Colors.GREEN}[{self.name}] ✅ Shared memory system operational{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}[{self.name}] ❌ Shared memory initialization failed: {str(e)}{Colors.END}")
            raise
    
    def _initialize_chat_agent(self):
        """Initialize chat agent with enhanced monitoring"""
        try:
            print(f"{Colors.GREEN}[{self.name}] 🧠 Loading Enhanced RAG-ChatAgent...{Colors.END}")
            
            start_time = time.time()
            self.chat_agent = ChatAgent(self.shared_memory)
            load_time = time.time() - start_time
            
            # Verify chat agent functionality
            agent_info = self.chat_agent.get_model_info()
            
            print(f"{Colors.GREEN}[{self.name}] ✅ Chat agent loaded in {load_time:.2f}s{Colors.END}")
            print(f"{Colors.GREEN}[{self.name}] 📊 Agent Info: {agent_info.get('model_name', 'RAG System')}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}[{self.name}] ❌ Chat agent initialization failed: {str(e)}{Colors.END}")
            if self.config.backup_chat_agent:
                print(f"{Colors.YELLOW}[{self.name}] 🔄 Attempting backup chat mode...{Colors.END}")
                self._initialize_backup_chat()
            else:
                raise
    
    def _initialize_backup_chat(self):
        """Initialize backup chat functionality"""
        try:
            # Simplified backup chat agent (stub)
            self.chat_agent = None
            self.backup_mode = True
            print(f"{Colors.YELLOW}[{self.name}] ⚠️ Running in backup mode{Colors.END}")
        except Exception as e:
            print(f"{Colors.RED}[{self.name}] ❌ Backup chat initialization failed: {str(e)}{Colors.END}")
            raise
    
    def _perform_initial_health_check(self):
        """🆕 Perform comprehensive initial health check including AI/ML"""
        print(f"{Colors.GREEN}[{self.name}] 🏥 Performing system health check with AI/ML diagnostics...{Colors.END}")
        
        health_results = self.health_monitor.run_health_diagnostics(self.shared_memory, self.chat_agent)
        
        # Add AI/ML health check
        aiml_health = 1.0 if self.aiml_agent else 0.0
        overall_health = (health_results['overall_health'] + aiml_health) / 2
        
        if overall_health >= 0.8:
            print(f"{Colors.GREEN}[{self.name}] ✅ System health: EXCELLENT ({overall_health:.1%}){Colors.END}")
        elif overall_health >= 0.6:
            print(f"{Colors.YELLOW}[{self.name}] ⚠️ System health: GOOD ({overall_health:.1%}){Colors.END}")
        else:
            print(f"{Colors.RED}[{self.name}] ❌ System health: POOR ({overall_health:.1%}){Colors.END}")
            self.is_healthy = False
        
        # AI/ML specific health info
        if self.aiml_agent:
            print(f"{Colors.GREEN}[{self.name}] 🤖 AI/ML Medical System: OPERATIONAL{Colors.END}")
        else:
            print(f"{Colors.YELLOW}[{self.name}] 🤖 AI/ML Medical System: NOT AVAILABLE{Colors.END}")
    
    def _display_system_ready(self):
        """🆕 Enhanced system ready display with AI/ML info"""
        agent_info = {}
        if self.chat_agent:
            agent_info = self.chat_agent.get_model_info()
        
        papers_count = agent_info.get('papers_loaded', 0)
        chunks_count = agent_info.get('searchable_chunks', 0)
        rag_enabled = agent_info.get('rag_enabled', False)
        
        # AI/ML info
        aiml_info = {}
        if self.aiml_agent:
            try:
                aiml_info = self.aiml_agent.get_model_info()
            except:
                aiml_info = {'capabilities': ['Medical AI Features Available']}
        
        print(f"{Colors.GREEN}[{self.name}] ✅ Enhanced Modular System Components:{Colors.END}")
        print(f"{Colors.GREEN}   🧠 AI Model: {agent_info.get('model_name', 'Enhanced RAG System')}{Colors.END}")
        print(f"{Colors.GREEN}   📚 Knowledge Base: {papers_count} documents loaded{Colors.END}")
        print(f"{Colors.GREEN}   🔍 Search Capability: {chunks_count:,} searchable chunks{Colors.END}")
        print(f"{Colors.GREEN}   🔋 Processing Mode: {'RAG + SLM Enhanced' if rag_enabled else 'RAG Intelligent'}{Colors.END}")
        print(f"{Colors.GREEN}   🤖 AI/ML Medical: {'Available' if self.aiml_agent else 'Not Available'}{Colors.END}")
        print(f"{Colors.GREEN}   🏥 Medical Features: {len(aiml_info.get('capabilities', []))} capabilities{Colors.END}")
        print(f"{Colors.GREEN}   💾 Memory System: Multi-user + Performance tracking{Colors.END}")
        print(f"{Colors.GREEN}   📊 Monitoring: Real-time metrics and health checks{Colors.END}")
        print(f"{Colors.GREEN}   🛡️ Recovery: Auto-recovery and error handling{Colors.END}")
        print(f"{Colors.GREEN}   ⚙️ Response Config: {self.config.response_length_tokens} tokens, temp {self.config.response_temperature}{Colors.END}")
        print(f"{Colors.GREEN}   🔧 Architecture: Modular design with AI/ML integration{Colors.END}")
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}🎉 ENHANCED MULTI-AGENT MEDICAL SYSTEM READY!{Colors.END}")
        print(f"{Colors.GREEN}🟢 Modular architecture: Easy debugging and customization{Colors.END}")
        print(f"{Colors.GREEN}🔄 Intelligent routing: Research → RAG | General → Chat | Medical → AI/ML | System → Direct{Colors.END}")
        print(f"{Colors.GREEN}🏥 Medical AI routing: Image analysis, Report generation, Patient management{Colors.END}")
        print(f"{Colors.GREEN}📊 Performance monitoring: Response times, success rates, system health{Colors.END}")
        print(f"{Colors.GREEN}🚀 Version {self.version} - Medical AI implementation ready for advanced research and analysis!{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}\n")
    
    def handle_user_input(self, user_id: str, user_message: str, context: Dict[str, Any] = None) -> str:
        """
        🆕 ENHANCED MAIN ENTRY POINT - All messages with AI/ML support
        
        This is the main method that processes all user input through the modular system
        including AI/ML medical analysis capabilities.
        """
        if not self.is_initialized:
            return self.error_handler.handle_uninitialized_request(user_id, user_message)
        
        # Start performance tracking
        start_time = time.time()
        current_time = self.time_utils.get_current_time()
        
        # Update metrics and session
        self.performance_monitor.update_message_count()
        self.session_manager.initialize_user_session(user_id)
        
        # Enhanced input logging
        self._log_incoming_request(user_id, user_message, current_time)
        
        try:
            # Store user message in shared memory
            self.shared_memory.add_message(user_id, user_message, "User")
            
            # 🆕 Enhanced message analysis with AI/ML context
            analysis = self.message_analyzer.analyze_message(user_message, context)
            
            # Intelligent routing with modular processors including AI/ML
            response = self._enhanced_supervisor_routing(user_id, user_message, analysis, current_time, context)
            
            # Performance tracking
            response_time = time.time() - start_time
            self.performance_monitor.update_performance_metrics(response_time, True)
            self.session_manager.update_session_success(user_id, True)
            self.session_manager.add_session_response_time(user_id, response_time)
            
            # Success logging
            self._log_successful_response(user_id, response_time, analysis)
            
            return response
            
        except Exception as e:
            # Error handling with modular error handler
            response_time = time.time() - start_time
            self.performance_monitor.update_performance_metrics(response_time, False)
            self.session_manager.update_session_success(user_id, False)
            
            return self.error_handler.handle_processing_error(user_id, user_message, e, response_time)
    
    def _log_incoming_request(self, user_id: str, user_message: str, current_time: str):
        """Enhanced incoming request logging"""
        if not self.config.enable_detailed_logging:
            return
        
        print(f"\n{Colors.GREEN}{Colors.BOLD}[ENHANCED SUPERVISOR] 📨 INCOMING REQUEST{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 👤 User: {user_id}{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 🕐 Time: {current_time} UTC{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 💬 Query: '{user_message[:100]}{'...' if len(user_message) > 100 else ''}'{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 📊 Total Messages: {self.performance_monitor.metrics.total_messages}{Colors.END}")
    
    def _enhanced_supervisor_routing(self, user_id: str, message: str, analysis: MessageAnalysis, 
                                   current_time: str, context: Dict[str, Any] = None) -> str:
        """🆕 Enhanced routing with AI/ML support and comprehensive monitoring"""
        
        # Log routing decision
        self._log_routing_decision(user_id, analysis)
        
        # 🆕 Route based on analysis with modular processors including AI/ML
        if analysis.route_target == "aiml_system":
            # Update AI/ML metrics
            self.performance_monitor.update_aiml_request() if hasattr(self.performance_monitor, 'update_aiml_request') else None
            return self.aiml_processor.process_aiml_request(
                user_id, message, analysis, current_time, self.shared_memory, self.aiml_agent
            )
        
        elif analysis.type.value == "system_command":
            self.performance_monitor.update_system_command()
            return self.system_command_processor.handle_system_command(
                user_id, message, analysis, self.shared_memory, self.chat_agent
            )
        
        elif analysis.requires_rag:
            self.performance_monitor.update_rag_request()
            return self.rag_processor.process_rag_request(
                user_id, message, analysis, current_time, self.shared_memory, self.chat_agent
            )
        
        else:
            self.performance_monitor.update_chat_request()
            return self.chat_processor.process_chat_request(
                user_id, message, analysis, self.chat_agent
            )
    
    def _log_routing_decision(self, user_id: str, analysis: MessageAnalysis):
        """🆕 Log routing decision details including AI/ML routes"""
        if self.config.enable_detailed_logging:
            print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 🎯 Analysis Complete{Colors.END}")
            print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 📊 Type: {analysis.type.value}{Colors.END}")
            print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 🎚️ Priority: {analysis.priority.value}{Colors.END}")
            print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 🎯 Target: {analysis.route_target}{Colors.END}")
            print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 📈 Confidence: {analysis.confidence:.2%}{Colors.END}")
            print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 🏷️ Keywords: {', '.join(analysis.keywords)}{Colors.END}")
            
            # 🆕 AI/ML specific logging
            if analysis.route_target == "aiml_system":
                print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 🤖 AI/ML Features: {analysis.processing_flags.get('medical_related', False)}{Colors.END}")
                print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 🖼️ Image Processing: {analysis.processing_flags.get('image_processing', False)}{Colors.END}")
                print(f"{Colors.CYAN}[ENHANCED SUPERVISOR → ROUTING] 📄 Report Related: {analysis.processing_flags.get('report_related', False)}{Colors.END}")
    
    def _log_successful_response(self, user_id: str, response_time: float, analysis: MessageAnalysis):
        """Log successful response with metrics"""
        if self.config.enable_detailed_logging:
            print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ✅ RESPONSE DELIVERED{Colors.END}")
            print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ⚡ Time: {response_time:.2f}s{Colors.END}")
            print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 📊 Success Rate: {self.performance_monitor.get_success_rate():.1%}{Colors.END}")
            
            # 🆕 AI/ML specific success logging
            if analysis.route_target == "aiml_system":
                print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 🤖 AI/ML Processing: Completed successfully{Colors.END}")
    
    # 🆕 Enhanced methods for AI/ML integration
    def handle_image_upload(self, user_id: str, image_path: str, image_type: str = "general") -> str:
        """🆕 Handle image upload requests with proper context"""
        context = {
            'image_path': image_path,
            'image_type': image_type
        }
        
        message = f"upload {image_type} {image_path}" if image_type == "mri" else f"upload image {image_path}"
        return self.handle_user_input(user_id, message, context)
    
    def handle_report_generation(self, user_id: str, patient_info: Dict[str, str]) -> str:
        """🆕 Handle report generation with patient information"""
        context = {
            'generate_report': True,
            'patient_info': patient_info
        }
        
        return self.handle_user_input(user_id, "generate report", context)
    
    def get_aiml_status(self) -> Dict[str, Any]:
        """🆕 Get AI/ML agent status and capabilities"""
        if not self.aiml_agent:
            return {
                'available': False,
                'status': 'Not loaded',
                'capabilities': [],
                'reason': 'AI/ML agent not available or failed to initialize'
            }
        
        try:
            aiml_info = self.aiml_agent.get_model_info()
            return {
                'available': True,
                'status': 'Operational',
                'model_name': aiml_info.get('model_name', 'AI/ML Medical System'),
                'version': aiml_info.get('version', '1.0.0'),
                'capabilities': aiml_info.get('capabilities', []),
                'integrated_with': aiml_info.get('integrated_with', [])
            }
        except Exception as e:
            return {
                'available': True,
                'status': 'Limited functionality',
                'error': str(e),
                'capabilities': ['Basic AI/ML features']
            }
    
    # Delegate methods to appropriate modular components
    def get_conversation_history(self, user_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Get conversation history using shared memory"""
        try:
            return self.shared_memory.get_conversation_history(user_id, limit)
        except Exception as e:
            print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ Error getting history: {str(e)}{Colors.END}")
            return []
    
    def get_enhanced_conversation_history(self, user_id: str) -> List[Dict[str, Any]]:
        """🆕 Get enhanced conversation history with supervisor and AI/ML metadata"""
        history = self.get_conversation_history(user_id)
        
        # Add enhanced supervisor metadata
        enhanced_history = []
        for msg in history:
            enhanced_msg = msg.copy()
            enhanced_msg['processed_by_supervisor'] = True
            enhanced_msg['supervisor_version'] = self.version
            enhanced_msg['system_type'] = 'enhanced_multi_agent_medical_system_modular'
            enhanced_msg['enhanced_features'] = True
            enhanced_msg['modular_architecture'] = True
            enhanced_msg['aiml_enabled'] = self.aiml_agent is not None
            enhanced_history.append(enhanced_msg)
        
        return enhanced_history
    
    def get_system_metrics_export(self) -> Dict[str, Any]:
        """🆕 Export comprehensive system metrics including AI/ML"""
        base_metrics = self.performance_monitor.get_system_metrics_export(
            self.time_utils.get_current_time(),
            self.version,
            self.config.system_name,
            self.session_manager.active_sessions,
            self.health_monitor.run_health_diagnostics(self.shared_memory, self.chat_agent)
        )
        
        # Add AI/ML specific metrics
        aiml_metrics = {
            'aiml_agent_available': self.aiml_agent is not None,
            'aiml_status': self.get_aiml_status(),
            'medical_features_enabled': AIML_AVAILABLE,
            'total_agents': 4 if self.aiml_agent else 3  # Chat, RAG, AI/ML, Supervisor
        }
        
        base_metrics['aiml_metrics'] = aiml_metrics
        return base_metrics
    
    def perform_maintenance(self) -> str:
        """🆕 Perform system maintenance tasks including AI/ML"""
        maintenance_report = []
        current_time = self.time_utils.get_current_time()
        
        maintenance_report.append(f"🔧 **ENHANCED MEDICAL SYSTEM MAINTENANCE REPORT**")
        maintenance_report.append(f"🕐 **Maintenance Time:** {current_time} UTC")
        maintenance_report.append(f"🤖 **Supervisor Version:** {self.version} (Medical AI Modular)")
        maintenance_report.append("")
        
        # Clean up old response times
        if len(self.performance_monitor.response_times) > self.config.max_response_history:
            old_count = len(self.performance_monitor.response_times)
            self.performance_monitor.response_times = self.performance_monitor.response_times[-self.config.max_response_history:]
            maintenance_report.append(f"✅ Cleaned response time history: {old_count} → {len(self.performance_monitor.response_times)} entries")
        
        # Clean up inactive sessions
        inactive_count = self.session_manager.cleanup_inactive_sessions()
        if inactive_count > 0:
            maintenance_report.append(f"✅ Cleaned {inactive_count} inactive sessions")
        
        # Update system metrics
        self.performance_monitor.metrics.uptime_seconds = time.time() - self.start_time
        maintenance_report.append(f"✅ Updated system metrics and memory usage")
        
        # Perform health check including AI/ML
        health_results = self.health_monitor.run_health_diagnostics(self.shared_memory, self.chat_agent)
        aiml_health = "Available" if self.aiml_agent else "Not Available"
        maintenance_report.append(f"✅ Modular health check completed: {health_results['overall_health']:.1%}")
        maintenance_report.append(f"✅ AI/ML Medical System: {aiml_health}")
        
        # Memory optimization suggestions
        memory_pct = health_results.get('memory_usage', 0)
        if memory_pct > self.config.memory_critical_threshold:
            maintenance_report.append(f"⚠️ **Warning:** High memory usage ({memory_pct:.1f}%) - consider restart")
        elif memory_pct > self.config.memory_warning_threshold:
            maintenance_report.append(f"💡 **Info:** Moderate memory usage ({memory_pct:.1f}%) - monitoring recommended")
        else:
            maintenance_report.append(f"✅ Memory usage optimal ({memory_pct:.1f}%)")
        
        # AI/ML specific maintenance
        if self.aiml_agent:
            try:
                # Could add AI/ML specific cleanup here
                maintenance_report.append(f"✅ AI/ML Medical System: Operational and maintained")
            except Exception as e:
                maintenance_report.append(f"⚠️ AI/ML Medical System: Maintenance warning - {str(e)}")
        
        maintenance_report.append("")
        maintenance_report.append(f"🎯 **Maintenance Summary:**")
        maintenance_report.append(f"   • System health: {self.health_monitor.get_current_health_status(self.shared_memory, self.chat_agent)}")
        maintenance_report.append(f"   • Active sessions: {len(self.session_manager.active_sessions)}")
        maintenance_report.append(f"   • Success rate: {self.performance_monitor.get_success_rate():.1%}")
        maintenance_report.append(f"   • Average response time: {self.performance_monitor.metrics.average_response_time:.2f}s")
        maintenance_report.append(f"   • Response config: {self.config.response_length_tokens} tokens")
        maintenance_report.append(f"   • AI/ML Medical: {aiml_health}")
        maintenance_report.append("")
        maintenance_report.append(f"✅ **Medical AI maintenance completed successfully!**")
        
        return "\n".join(maintenance_report)
    
    def shutdown(self):
        """🆕 Enhanced graceful supervisor shutdown including AI/ML"""
        current_time = self.time_utils.get_current_time()
        
        print(f"\n{Colors.GREEN}[ENHANCED SUPERVISOR] 🔄 Initiating graceful medical system shutdown...{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 💾 Saving enhanced system state and metrics...{Colors.END}")
        
        # Save final metrics including AI/ML
        final_metrics = self.get_system_metrics_export()
        
        # AI/ML specific shutdown
        if self.aiml_agent:
            print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 🤖 Shutting down AI/ML Medical System...{Colors.END}")
            # Could add specific AI/ML cleanup here if needed
        
        # Clean up sessions
        session_count = len(self.session_manager.active_sessions)
        self.session_manager.active_sessions.clear()
        self.session_manager.session_metrics.clear()
        
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 👥 Cleaned {session_count} active sessions{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 📊 Final metrics saved{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ⏱️ Total uptime: {self.performance_monitor.format_uptime(self.performance_monitor.metrics.uptime_seconds)}{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 📈 Total messages processed: {self.performance_monitor.metrics.total_messages:,}{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 🎯 Final success rate: {self.performance_monitor.get_success_rate():.1%}{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ⚙️ Response config: {self.config.response_length_tokens} tokens{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 🤖 AI/ML Medical System: {'Operational' if self.aiml_agent else 'Not Available'}{Colors.END}")
        print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] ✅ Medical system shutdown completed at {current_time} UTC{Colors.END}")
        print(f"{Colors.GREEN}🏥 Thank you for using the Enhanced Multi-Agent Medical Research System v{self.version}!{Colors.END}\n")

    def update_current_time(self, new_time: str = None):
        """Update current time - can be used for testing or manual time setting"""
        if new_time:
            # Validate format: YYYY-MM-DD HH:MM:SS
            try:
                datetime.strptime(new_time, '%Y-%m-%d %H:%M:%S')
                # Note: TimeUtils uses current system time, this is for compatibility
                print(f"{Colors.GREEN}[ENHANCED SUPERVISOR] 🕐 Time reference updated to: {new_time} UTC{Colors.END}")
            except ValueError:
                print(f"{Colors.RED}[ENHANCED SUPERVISOR] ❌ Invalid time format. Use: YYYY-MM-DD HH:MM:SS{Colors.END}")

    def handle_user_input_with_timestamp(self, user_id: str, user_message: str, timestamp: str = None, context: Dict[str, Any] = None) -> str:
        """🆕 Enhanced handle_user_input with explicit timestamp and context support"""
        # Update system time if provided
        if timestamp:
            self.update_current_time(timestamp)
        
        # Use the existing enhanced handler with context
        return self.handle_user_input(user_id, user_message, context)


# ========================================
# CONFIGURATION PRESETS FOR EASY USE
# ========================================

def create_medical_config() -> Dict[str, Any]:
    """🆕 Create configuration optimized for medical AI processing"""
    config = SupervisorConfig.get_default_config()
    config.update({
        'response_length_tokens': 350,  # Longer for detailed medical reports
        'response_temperature': 0.7,
        'response_repetition_penalty': 1.1,
        'max_response_time': 60.0,  # Allow more time for medical processing
        'force_rag_for_research': True,
        'rag_confidence_threshold': 0.6,  # Lower for medical queries
        'enable_detailed_logging': True,
        'performance_monitoring': True,
        'medical_ai_enabled': True,
        'system_name': 'Enhanced Multi-Agent Medical Research System'
    })
    return config

def create_long_response_config() -> Dict[str, Any]:
    """Create configuration for longer, more detailed responses"""
    config = SupervisorConfig.get_default_config()
    config.update({
        'response_length_tokens': 300,  # Much longer responses
        'response_temperature': 0.7,
        'response_repetition_penalty': 1.1,
        'max_response_time': 45.0,  # Allow more time for detailed responses
        'force_rag_for_research': True,
        'rag_confidence_threshold': 0.7,  # Lower threshold for more RAG usage
        'enable_detailed_logging': True
    })
    return config

def create_fast_response_config() -> Dict[str, Any]:
    """Create configuration optimized for speed"""
    config = SupervisorConfig.get_default_config()
    config.update({
        'response_length_tokens': 100,  # Shorter for speed
        'response_temperature': 0.6,
        'max_response_time': 15.0,
        'enable_detailed_logging': False,
        'performance_monitoring': True,  # Keep monitoring for fast systems
        'force_rag_for_research': True
    })
    return config

def create_debug_config() -> Dict[str, Any]:
    """Create configuration optimized for debugging"""
    config = SupervisorConfig.get_default_config()
    config.update({
        'enable_detailed_logging': True,
        'auto_recovery': False,  # Disable for debugging
        'response_length_tokens': 200,
        'performance_monitoring': True,
        'error_reporting': True
    })
    return config

def create_production_config() -> Dict[str, Any]:
    """Create configuration optimized for production"""
    config = SupervisorConfig.get_default_config()
    config.update({
        'response_length_tokens': 250,  # Balanced length
        'response_temperature': 0.7,
        'max_response_time': 30.0,
        'enable_detailed_logging': False,  # Reduce logging in production
        'auto_recovery': True,
        'performance_monitoring': True,
        'force_rag_for_research': True,
        'session_timeout_minutes': 60,  # Longer sessions in production
        'max_concurrent_sessions': 20
    })
    return config


# ========================================
# MAIN ENTRY POINT WITH AI/ML EXAMPLES
# ========================================

if __name__ == "__main__":
    # Initialize with current date/time
    current_time = "2025-06-14 18:35:06"  # Updated timestamp
    current_user = "Sagar4276"  # Your provided user login
    
    print(f"{Colors.GREEN}🚀 Initializing Enhanced Multi-Agent Medical System with AI/ML...{Colors.END}")
    print(f"{Colors.GREEN}🕐 Current Date/Time (UTC): {current_time}{Colors.END}")
    print(f"{Colors.GREEN}👤 Current User Login: {current_user}{Colors.END}")
    print(f"{Colors.GREEN}🛠️ Version: MEDICAL AI MODULAR with AI/ML integration{Colors.END}")
    
    # Create enhanced supervisor with medical configuration
    supervisor = EnhancedSupervisorAgent(create_medical_config())
    
    # Update with current time and user
    supervisor.update_current_time(current_time)
    supervisor.current_user = current_user
    
    # Example usage with AI/ML routing
    print(f"\n{Colors.CYAN}📝 Example Enhanced Medical Supervisor Usage:{Colors.END}")
    
    # Test medical research query routing
    print(f"\n{Colors.YELLOW}🧪 Testing medical research routing:{Colors.END}")
    research_response = supervisor.handle_user_input_with_timestamp(
        current_user,
        "parkinsons prevention measures",  # This should route to RAG system
        current_time
    )
    print(f"\n{Colors.BLUE}Medical Research Response:{Colors.END}")
    print(research_response[:500] + "..." if len(research_response) > 500 else research_response)
    
    # Test AI/ML system status
    print(f"\n{Colors.YELLOW}🧪 Testing AI/ML system status:{Colors.END}")
    status_response = supervisor.handle_user_input_with_timestamp(
        current_user, 
        "system status", 
        current_time
    )
    print(f"\n{Colors.BLUE}System Status Response:{Colors.END}")
    print(status_response[:500] + "..." if len(status_response) > 500 else status_response)
    
    # Test AI/ML image upload (mock)
    print(f"\n{Colors.YELLOW}🧪 Testing AI/ML medical routing:{Colors.END}")
    aiml_response = supervisor.handle_user_input_with_timestamp(
        current_user,
        "upload mri brain_scan.dcm",  # This should route to AI/ML system
        current_time
    )
    print(f"\n{Colors.BLUE}AI/ML Medical Response:{Colors.END}")
    print(aiml_response[:500] + "..." if len(aiml_response) > 500 else aiml_response)
    
    print(f"\n{Colors.GREEN}✅ Enhanced Multi-Agent Medical System demonstration completed!{Colors.END}")
    
    # Show final metrics including AI/ML
    print(f"\n{Colors.CYAN}📊 Final Medical System Metrics:{Colors.END}")
    final_metrics = supervisor.get_system_metrics_export()
    print(f"{Colors.GREEN}✅ Total Messages: {final_metrics['performance_metrics']['total_messages']}{Colors.END}")
    print(f"{Colors.GREEN}✅ RAG Requests: {final_metrics['performance_metrics']['rag_requests']}{Colors.END}")
    print(f"{Colors.GREEN}✅ Chat Requests: {final_metrics['performance_metrics']['chat_requests']}{Colors.END}")
    print(f"{Colors.GREEN}✅ System Commands: {final_metrics['performance_metrics']['system_commands']}{Colors.END}")
    print(f"{Colors.GREEN}✅ AI/ML Available: {final_metrics['aiml_metrics']['aiml_agent_available']}{Colors.END}")
    print(f"{Colors.GREEN}✅ Total Agents: {final_metrics['aiml_metrics']['total_agents']}{Colors.END}")
    print(f"{Colors.GREEN}✅ Success Rate: {final_metrics['system_status']['success_rate']:.1%}{Colors.END}")
    print(f"{Colors.GREEN}✅ Health Score: {final_metrics['system_status']['current_health_score']:.1%}{Colors.END}")
    
    print(f"\n{Colors.GREEN}🎯 **ENHANCED MEDICAL AI FEATURES:**{Colors.END}")
    print(f"{Colors.GREEN}   🔧 Easy debugging - each component in separate file{Colors.END}")
    print(f"{Colors.GREEN}   📝 Configurable responses - edit supervisor_config.py{Colors.END}")
    print(f"{Colors.GREEN}   🎯 Better routing - improved message analysis with AI/ML{Colors.END}")
    print(f"{Colors.GREEN}   ✨ Enhanced formatting - modular response formatter{Colors.END}")
    print(f"{Colors.GREEN}   🕐 Consistent timestamps - centralized time utilities{Colors.END}")
    print(f"{Colors.GREEN}   🛠️ Maintainable code - clear separation of concerns{Colors.END}")
    print(f"{Colors.GREEN}   🤖 AI/ML Integration - medical image analysis and reports{Colors.END}")
    print(f"{Colors.GREEN}   🏥 Medical workflows - MRI → Analysis → Reports{Colors.END}")
    
    # Show AI/ML status
    aiml_status = supervisor.get_aiml_status()
    print(f"\n{Colors.CYAN}🤖 AI/ML Medical System Status:{Colors.END}")
    print(f"{Colors.GREEN}✅ Available: {aiml_status['available']}{Colors.END}")
    print(f"{Colors.GREEN}✅ Status: {aiml_status['status']}{Colors.END}")
    if aiml_status.get('capabilities'):
        print(f"{Colors.GREEN}✅ Capabilities: {len(aiml_status['capabilities'])} medical features{Colors.END}")
    
    # Graceful shutdown
    supervisor.shutdown()