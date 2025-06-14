"""
Enhanced Chat Agent - Clean Modular Architecture (FINAL - NO BREAKING CHANGES)
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-12 12:46:43
Current User's Login: Sagar4276
"""

import time
from datetime import datetime, timezone
from shared_memory.simple_memory import SimpleSharedMemory

# Import modular components (FIXED to use original names)
from .models.model_loader import ModelLoader
from .processors.query_analyzer import QueryAnalyzer
from .processors.rag_processor import RAGProcessor
from .generators.llm_generator import LLMGenerator
from .generators.structured_generator import StructuredGenerator
from .processors.response_formatter import ResponseFormatter

class ChatAgent:
    """SAME CLASS NAME - Enhanced Chat Agent with smart RAG detection and error resilience"""
    
    def __init__(self, shared_memory: SimpleSharedMemory):
        self.shared_memory = shared_memory
        self.name = "ChatAgent"
        self.current_user = "Sagar4276"
        self.version = "2.0.0"
        
        print(f"[{self.name}] 🚀 Loading Enhanced RAG + AI System v{self.version}...")
        print(f"[{self.name}] 👤 Current User: {self.current_user}")
        print(f"[{self.name}] 🕐 Current Time: {self.get_current_time()} UTC")
        
        # Smart RAG system initialization (adapts to what's available)
        print(f"[{self.name}] 📚 Initializing Smart RAG System...")
        self.rag_agent = self._initialize_smart_rag_system()
        
        # Initialize modular components
        self._initialize_components()
        
        # Show system status
        self._display_system_status()
        
        print(f"[{self.name}] ✅ Enhanced Chat Agent ready for conversations!")
    
    def _initialize_smart_rag_system(self):
        """Smart RAG initialization - tries modern first, falls back gracefully"""
        rag_order = [
            ("Modern AI RAG (FAISS)", "ModernEnhancedRAGAgent", "🧠"),
            ("Enhanced Vectorized RAG", "EnhancedRAGAgent", "🔢"),
            ("Legacy Enhanced RAG", "RAGAgent", "📋")
        ]
        
        for rag_name, rag_class, emoji in rag_order:
            try:
                print(f"[{self.name}] {emoji} Attempting {rag_name} initialization...")
                exec(f"from agents.RAG.rag_agent import {rag_class}")
                rag_agent = eval(f'{rag_class}("knowledge_base/papers")')
                print(f"[{self.name}] ✅ Successfully initialized {rag_name}")
                return rag_agent
            except Exception as e:
                print(f"[{self.name}] ⚠️ {rag_name} unavailable: {str(e)[:50]}...")
                continue
        
        # If all fail, create a minimal fallback
        print(f"[{self.name}] 🆘 Creating minimal fallback RAG system...")
        return self._create_fallback_rag()
    
    def _create_fallback_rag(self):
        """Create a minimal fallback RAG system if all others fail"""
        class FallbackRAG:
            def __init__(self, path):
                self.documents = []
                self.path = path
                print(f"[ChatAgent] 📂 Fallback RAG initialized with path: {path}")
            
            def get_stats(self):
                return {
                    'documents': 0,
                    'chunks': 0,
                    'total_chunks': 0,
                    'status': 'fallback_mode'
                }
            
            def search_papers(self, query, max_results=3):
                return []
        
        return FallbackRAG("knowledge_base/papers")
    
    def _initialize_components(self):
        """SAME FUNCTION NAME - Initialize all modular components with error handling"""
        print(f"[{self.name}] 🔧 Initializing modular components...")
        
        try:
            # Core components (FIXED to use original names)
            self.model_loader = ModelLoader(self.name)
            self.query_analyzer = QueryAnalyzer(self.name)
            self.rag_processor = RAGProcessor(self.rag_agent, self.name)
            self.llm_generator = LLMGenerator(self.model_loader, self.name)
            self.structured_generator = StructuredGenerator(self.rag_agent, self.name)
            self.response_formatter = ResponseFormatter(self.rag_agent, self.name)
            
            # Load model (SAME FUNCTION NAME)
            print(f"[{self.name}] 🤖 Loading language model...")
            self.llm = self.model_loader.load_offline_slm()
            
            print(f"[{self.name}] ✅ All components initialized successfully")
            
        except Exception as e:
            print(f"[{self.name}] ❌ Component initialization error: {e}")
            print(f"[{self.name}] 🔄 Creating fallback components...")
            self._create_fallback_components()
    
    def _create_fallback_components(self):
        """Create fallback components if initialization fails"""
        class FallbackComponent:
            def __init__(self, name):
                self.name = name
                self.agent_name = "ChatAgent"
            
            def __getattr__(self, name):
                return lambda *args, **kwargs: f"⚠️ Fallback mode - {self.name} component not fully available"
        
        self.model_loader = FallbackComponent("ModelLoader")
        self.query_analyzer = FallbackComponent("QueryAnalyzer") 
        self.rag_processor = FallbackComponent("RAGProcessor")
        self.llm_generator = FallbackComponent("LLMGenerator")
        self.structured_generator = FallbackComponent("StructuredGenerator")
        self.response_formatter = FallbackComponent("ResponseFormatter")
        self.llm = None
    
    def _display_system_status(self):
        """Enhanced system status display with smart detection and error handling"""
        try:
            rag_stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
            
            print(f"[{self.name}] 📊 Enhanced RAG Status:")
            print(f"   📚 Documents: {rag_stats.get('documents', len(getattr(self.rag_agent, 'documents', [])))}")
            print(f"   📄 Chunks: {rag_stats.get('total_chunks', rag_stats.get('chunks', 0)):,}")
            
            # Smart system detection for status
            if hasattr(self.rag_agent, 'get_system_info'):
                try:
                    system_info = self.rag_agent.get_system_info()
                    print(f"   🧠 Modern AI: {'✅ Ready' if rag_stats.get('system_ready') else '❌ Not Ready'}")
                    print(f"   🤖 AI Model: {system_info.get('embedding_model', 'Unknown')}")
                    print(f"   📊 Vector Dims: {system_info.get('vector_dimension', 0):,}")
                    print(f"   ⚡ Search Type: {system_info.get('index_type', 'Unknown')}")
                    print(f"   💾 Memory: {system_info.get('memory_usage_mb', 0):.1f}MB")
                except Exception as e:
                    print(f"   ⚠️ Modern AI status unavailable: {str(e)[:30]}...")
            elif hasattr(self.rag_agent, 'is_vectorized') and getattr(self.rag_agent, 'is_vectorized', False):
                print(f"   🔢 Vectorized: ✅ Active")
                print(f"   📊 Vector Dims: {rag_stats.get('vector_dimensions', 0):,}")
                print(f"   ⚡ Search Type: Mathematical similarity")
            else:
                status = rag_stats.get('status', 'enhanced_keyword')
                print(f"   📋 Search Type: {status.replace('_', ' ').title()}")
            
            # Model status
            try:
                model_info = self.model_loader.get_model_info()
                print(f"   🤖 LLM: {model_info['model_name']} ({'✅ Loaded' if model_info['loaded'] else '❌ Not Available'})")
            except Exception as e:
                print(f"   🤖 LLM: Status unknown")
                
        except Exception as e:
            print(f"[{self.name}] ⚠️ Status display error: {e}")
            print(f"[{self.name}] 📝 System running in basic mode")
    
    def get_current_time(self):
        """SAME FUNCTION NAME - Get current UTC time with error handling"""
        try:
            return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            return "2025-06-12 12:46:43"  # Fallback timestamp
    
    def process_message(self, user_id: str, message: str) -> str:
        """SAME FUNCTION NAME - Enhanced message processing with comprehensive error handling"""
        current_time = self.get_current_time()
        print(f"[{self.name}] 📨 [{current_time}] Processing for {user_id}: '{message[:50]}{'...' if len(message) > 50 else ''}'")
        
        try:
            # Get conversation history
            history = []
            try:
                history = self.shared_memory.get_conversation_history(user_id)
            except Exception as e:
                print(f"[{self.name}] ⚠️ History retrieval error: {e}")
            
            # Step 1: Analyze the query
            try:
                analysis = self.query_analyzer.analyze_query(message)
            except Exception as e:
                print(f"[{self.name}] ⚠️ Query analysis error: {e}")
                # Create fallback analysis
                analysis = type('QueryAnalysis', (), {
                    'is_research_query': '?' in message or any(word in message.lower() for word in ['what', 'how', 'why', 'explain']),
                    'query_type': 'unknown',
                    'confidence': 0.5
                })()
            
            # Step 2: Route based on analysis
            if analysis.is_research_query:
                print(f"[{self.name}] 🔍 Research query detected - using Enhanced RAG...")
                response = self._process_research_query(user_id, message, history, analysis)
            elif self._is_system_command(message):
                print(f"[{self.name}] 🔧 System command detected...")
                response = self._process_system_command(user_id, message)
            else:
                print(f"[{self.name}] 💬 General conversation...")
                response = self._process_general_conversation(user_id, message, history)
            
            # Step 3: Store and return
            try:
                self.shared_memory.add_message(user_id, response, self.name)
            except Exception as e:
                print(f"[{self.name}] ⚠️ Message storage error: {e}")
            
            print(f"[{self.name}] ✅ Response generated for {user_id}")
            return response
            
        except Exception as e:
            print(f"[{self.name}] ❌ Critical processing error: {e}")
            return self._create_error_response(user_id, message, str(e))
    
    def _is_system_command(self, message: str) -> bool:
        """Simple system command detection as fallback"""
        try:
            return self.query_analyzer.is_system_command(message)
        except:
            cmd_keywords = ['show papers', 'system status', 'help', 'ai info', 'status']
            return any(cmd in message.lower() for cmd in cmd_keywords)
    
    def _create_error_response(self, user_id: str, message: str, error: str) -> str:
        """Create user-friendly error response"""
        current_time = self.get_current_time()
        
        return f"""🛠️ **System Notice for {user_id}**

I encountered a technical issue while processing your message: *"{message[:50]}{'...' if len(message) > 50 else ''}"*

**Error Details:** {error[:100]}{'...' if len(error) > 100 else ''}

**What you can try:**
• Rephrase your question in simpler terms
• Try asking "help" to see available commands
• Check if documents are properly loaded with "show papers"
• Try "system status" to check system health

**Time:** {current_time} UTC
**Status:** I'm still here and ready to help! 🤖"""
    
    def _process_research_query(self, user_id: str, message: str, history: list, analysis) -> str:
        """Enhanced research query processing - FIXED to always try RAG search"""
        try:
            print(f"[{self.name}] 🔍 Processing research query for {user_id}: '{message}'")
        
        # ALWAYS try RAG search for research queries (no matter what the message is)
            print(f"[{self.name}] 🧠 Using modern FAISS semantic search...")
            rag_result = self.rag_processor.process_research_query(user_id, message, history)
        
        # Debug RAG results
            if rag_result and rag_result.get('success') and rag_result.get('retrieval_results'):
                print(f"[{self.name}] 📚 Found {len(rag_result['retrieval_results'])} high-quality results")
                for i, result in enumerate(rag_result['retrieval_results'][:3]):
                    print(f"[{self.name}]    {i+1}. {result.source} (similarity: {result.similarity_score:.3f})")
        
            if not rag_result['success']:
                if rag_result['reason'] == 'no_results':
                    return self._handle_no_results(user_id, message)
                else:
                    return f"❌ **Search Error**: {rag_result['message']}. Please try rephrasing your question or check if documents are loaded."
        
         # Step 2: Generate Response with Enhanced Context
            print(f"[{self.name}] 🧠 Generating Enhanced RAG response...")
        
        # Try LLM generation with enhanced context
            try:
                llm_result = self.llm_generator.generate_rag_response(
                rag_result['context'], 
                message, 
                rag_result['retrieval_results']
            )
            
                if llm_result['success']:
                # Format with LLM response + enhanced metadata
                    return self.response_formatter.format_rag_response(
                    llm_result['answer'],
                    rag_result['retrieval_results'],
                    message,
                    self.get_current_time()
                )
            except Exception as e:
                print(f"[{self.name}] ⚠️ LLM generation error: {e}")
        
        # Fallback to structured response with enhanced results
            print(f"[{self.name}] 📋 Using enhanced structured response...")
            return self.structured_generator.create_structured_rag_response(
            message,
            rag_result['retrieval_results'],
            self.get_current_time()
        )
        
        except Exception as e:
            print(f"[{self.name}] ❌ Research query processing error: {e}")
            return self._create_research_fallback_response(user_id, message)
    
    def _create_research_fallback_response(self, user_id: str, message: str) -> str:
        """Create fallback response for research queries when systems fail"""
        current_time = self.get_current_time()
        
        return f"""🔍 **Research Query Response for {user_id}**

I understand you're asking about: *"{message}"*

Unfortunately, I'm experiencing technical difficulties with the research system right now.

**What I can suggest:**
• Try rephrasing your question with simpler terms
• Check if research documents are loaded with `show papers`
• Try `system status` to see what systems are available
• Ask `help` for alternative ways to get information

**Technical Status:** Research systems temporarily unavailable
**Time:** {current_time} UTC

I apologize for the inconvenience and appreciate your patience! 🤖"""
    
    def _handle_no_results(self, user_id: str, message: str) -> str:
        """Enhanced no-results handling with error resilience"""
        try:
            return self.rag_processor.handle_no_results(user_id, message)
        except Exception as e:
            print(f"[{self.name}] ⚠️ No-results handler error: {e}")
            current_time = self.get_current_time()
            
            return f"""🔍 **No Results Found for {user_id}**

I couldn't find information about: *"{message}"*

**Suggestions:**
• Try different keywords or phrases
• Use more specific terms
• Check available documents with `show papers`
• Try broader search terms

**Time:** {current_time} UTC"""
    
    def _process_system_command(self, user_id: str, message: str) -> str:
        """Enhanced system command processing with error handling"""
        try:
            command_type = self.query_analyzer.get_command_type(message)
        except:
            # Simple fallback command detection
            msg_lower = message.lower()
            if 'papers' in msg_lower:
                command_type = "show_papers"
            elif 'status' in msg_lower:
                command_type = "system_status"
            elif 'help' in msg_lower:
                command_type = "help"
            elif 'ai' in msg_lower or 'vector' in msg_lower:
                command_type = "ai_info"
            else:
                command_type = "unknown"
        
        current_time = self.get_current_time()
        
        try:
            if command_type == "show_papers":
                return self._format_paper_list(user_id, current_time)
            elif command_type == "system_status":
                return self.response_formatter.format_system_status(user_id, current_time)
            elif command_type in ["vectorization_info", "ai_info"]:
                return self.response_formatter.format_vectorization_info(current_time)
            elif command_type == "help":
                return self._create_enhanced_help(user_id, current_time)
            else:
                return self._create_command_help(user_id, message, current_time)
                
        except Exception as e:
            print(f"[{self.name}] ⚠️ System command error: {e}")
            return self._create_command_fallback(user_id, message, current_time)
    
    def _create_command_help(self, user_id: str, message: str, current_time: str) -> str:
        """Create help for unknown commands"""
        return f"""🔧 **Unknown Command for {user_id}:** '{message}'

**Available commands:**
• `show papers` - View document library
• `system status` - Check system health  
• `ai info` - View AI capabilities
• `help` - Get detailed help

**Time:** {current_time} UTC"""
    
    def _create_command_fallback(self, user_id: str, message: str, current_time: str) -> str:
        """Fallback for system command errors"""
        return f"""🔧 **System Command Status for {user_id}**

Command: *"{message}"*
Status: Experiencing technical difficulties

**Basic commands that might work:**
• `help` - Get help
• `status` - Check status

**Time:** {current_time} UTC"""
    
    def _format_paper_list(self, user_id: str, current_time: str) -> str:
        """Enhanced paper list formatting with error handling"""
        try:
            if hasattr(self.response_formatter, 'format_paper_list'):
                return self.response_formatter.format_paper_list(user_id, current_time)
        except Exception as e:
            print(f"[{self.name}] ⚠️ Paper list formatting error: {e}")
        
        # Fallback formatting
        try:
            documents = getattr(self.rag_agent, 'documents', [])
            if not documents:
                return f"📚 **No papers loaded for {user_id}** • {current_time} UTC\n\n💡 Add PDF documents to enable Enhanced RAG!"
            
            response = f"📚 **Document Library for {user_id}** • {current_time} UTC\n\n"
            for i, doc in enumerate(documents, 1):
                doc_name = getattr(doc, 'name', f'Document {i}')
                response += f"   {i}. {doc_name}\n"
            
            response += f"\n📊 **Total:** {len(documents)} papers available"
            return response
            
        except Exception as e:
            return f"📚 **Document Status for {user_id}**\n\nError accessing document list: {str(e)[:50]}...\n\n**Time:** {current_time} UTC"
    
    def _create_enhanced_help(self, user_id: str, current_time: str) -> str:
        """Enhanced help with system-aware content and error handling"""
        try:
            stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
            
            # Detect system type for appropriate help
            if hasattr(self.rag_agent, 'get_system_info'):
                system_type = "Modern AI RAG (FAISS)"
                ai_description = "semantic AI understanding"
            elif hasattr(self.rag_agent, 'is_vectorized') and getattr(self.rag_agent, 'is_vectorized', False):
                system_type = "Enhanced Vectorized RAG"
                ai_description = "mathematical similarity matching"
            else:
                system_type = "Enhanced RAG"
                ai_description = "intelligent keyword matching"
            
            help_text = f"""📖 **Enhanced RAG Assistant Help**

👤 **User:** {user_id}
🕐 **Time:** {current_time} UTC
🧠 **System:** {system_type}
📚 **Documents:** {stats.get('documents', len(getattr(self.rag_agent, 'documents', [])))} loaded

🔍 **Research Queries:**
• Ask questions about your documents
• Use natural language ({ai_description})
• Examples: 'What is machine learning?', 'Explain neural networks'

🔧 **System Commands:**
• `show papers` - View document library
• `system status` - Check system health
• `ai info` - View AI capabilities

🚀 **Enhanced Features:**
• **Smart Search:** {ai_description}
• **Context Aware:** Maintains conversation context  
• **Fast Retrieval:** Optimized search performance
• **Smart Ranking:** Best results ranked by relevance

💡 **Tips for better results:**
• Be specific in your questions
• Use domain-specific terminology
• Ask follow-up questions for deeper exploration

✨ **Version:** {self.version} • **Status:** Ready to help!"""

            return help_text
            
        except Exception as e:
            return f"""📖 **Help for {user_id}**

**Basic Commands:**
• Ask questions about research topics
• Try `show papers` to see documents
• Try `system status` to check health

**Time:** {current_time} UTC
**Status:** Basic help mode (error: {str(e)[:30]}...)"""
    
    def _process_general_conversation(self, user_id: str, message: str, history: list) -> str:
        """Enhanced general conversation processing with error handling"""
        try:
            return self.structured_generator.create_general_responses(
                user_id, 
                message, 
                self.get_current_time(), 
                self.rag_agent
            )
        except Exception as e:
            print(f"[{self.name}] ⚠️ General conversation error: {e}")
            current_time = self.get_current_time()
            
            # Simple fallback responses
            msg_lower = message.lower()
            if any(word in msg_lower for word in ['hello', 'hi', 'hey']):
                return f"Hello {user_id}! I'm your Enhanced RAG assistant. How can I help you today? • {current_time} UTC"
            elif any(phrase in msg_lower for phrase in ['how are you', 'how\'re you']):
                return f"I'm doing well, {user_id}! Ready to help with research queries. • {current_time} UTC"
            else:
                return f"I understand, {user_id}. Try asking me research questions or use `help` for assistance. • {current_time} UTC"
    
    def get_model_info(self):
        """SAME FUNCTION NAME - Enhanced model info with comprehensive error handling"""
        current_time = self.get_current_time()
        
        try:
            model_info = self.model_loader.get_model_info()
            stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
            
            base_info = {
                'model_type': 'enhanced_rag_smart_detection',
                'timestamp': current_time,
                'current_user': self.current_user,
                'version': self.version,
                'internet_required': False,
                'offline_capable': True,
                'multi_user_support': True,
                'rag_enabled': True,
                'documents_loaded': stats.get('documents', len(getattr(self.rag_agent, 'documents', []))),
                'searchable_chunks': stats.get('total_chunks', stats.get('chunks', 0)),
                'architecture': 'enhanced_modular_smart',
                'error_resilient': True
            }
            
            # Smart system detection for model info
            try:
                if hasattr(self.rag_agent, 'get_system_info'):
                    system_info = self.rag_agent.get_system_info()
                    base_info.update({
                        'rag_type': 'modern_semantic_search_rag',
                        'ai_engine': 'faiss_sentence_transformers',
                        'embedding_model': system_info.get('embedding_model', 'Unknown'),
                        'vector_dimension': system_info.get('vector_dimension', 0),
                        'search_engine': system_info.get('index_type', 'Modern AI'),
                        'memory_usage_mb': system_info.get('memory_usage_mb', 0),
                        'system_status': 'modern_ai_ready' if stats.get('system_ready') else 'modern_ai_loading'
                    })
                elif hasattr(self.rag_agent, 'is_vectorized') and getattr(self.rag_agent, 'is_vectorized', False):
                    base_info.update({
                        'rag_type': 'enhanced_vectorized_rag', 
                        'ai_engine': 'numpy_vectorization',
                        'vector_dimension': stats.get('vector_dimensions', 0),
                        'search_engine': 'TF-IDF + Cosine Similarity',
                        'system_status': 'vectorized_ready'
                    })
                else:
                    base_info.update({
                        'rag_type': 'enhanced_keyword_rag',
                        'ai_engine': 'intelligent_keyword_matching',
                        'search_engine': 'Enhanced Keyword Search',
                        'system_status': 'legacy_ready'
                    })
            except Exception as e:
                base_info.update({
                    'rag_type': 'fallback_mode',
                    'ai_engine': 'basic_processing',
                    'search_engine': 'Fallback Mode',
                    'system_status': 'limited_functionality',
                    'error_info': str(e)[:50]
                })
            
            # Add LLM info
            try:
                if model_info['loaded']:
                    base_info.update({
                        'model_name': f"{model_info['model_name']} + Enhanced RAG",
                        'status': f"{base_info['system_status']}_with_llm",
                        'description': f'Enhanced RAG: {model_info["model_name"]} with smart search across {base_info["documents_loaded"]} documents'
                    })
                else:
                    base_info.update({
                        'model_name': 'Enhanced Structured Response System',
                        'status': f"{base_info['system_status']}_structured",
                        'description': f'Enhanced RAG: Smart search + structured responses from {base_info["documents_loaded"]} documents'
                    })
            except Exception as e:
                base_info.update({
                    'model_name': 'Fallback Response System',
                    'status': 'basic_mode',
                    'description': 'Basic text processing with error resilience'
                })
            
            return base_info
            
        except Exception as e:
            print(f"[{self.name}] ⚠️ Model info error: {e}")
            return {
                'model_type': 'fallback_system',
                'timestamp': current_time,
                'current_user': self.current_user,
                'version': self.version,
                'status': 'basic_mode',
                'error_info': str(e)[:100],
                'description': 'System running in fallback mode due to initialization issues'
            }