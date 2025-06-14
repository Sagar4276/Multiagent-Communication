"""
Enhanced Chat Agent - Response Formatting Module (UPDATED)
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-12 12:31:49
Current User's Login: Sagar4276
"""

from typing import List, Dict, Any

class ResponseFormatter:
    """SAME CLASS NAME - Enhanced response formatting with rich metadata and visualization"""
    
    def __init__(self, rag_agent, agent_name: str = "ResponseFormatter"):
        self.agent_name = agent_name
        self.rag_agent = rag_agent
        self.current_user = "Sagar4276"
    
    def format_rag_response(self, answer: str, retrieval_results: list, query: str, current_time: str) -> str:
        """SAME FUNCTION NAME - Enhanced RAG response formatting with smart system detection"""
        
        # Smart system detection for appropriate headers
        system_type, system_emoji = self._detect_system_type()
        
        # Enhanced response header with smart formatting
        header = f"{system_emoji} **{system_type} Analysis:** {answer}\n\n"
        
        # Enhanced sources section with improved visualization
        sources_section = self._create_enhanced_sources_section(retrieval_results, system_type)
        
        # Enhanced footer with comprehensive system info
        footer = self._create_enhanced_footer(retrieval_results, current_time, system_type)
        
        return header + sources_section + footer
    
    def _detect_system_type(self) -> tuple:
        """NEW helper - Smart detection of system capabilities"""
        
        # Check for modern FAISS system
        if hasattr(self.rag_agent, 'get_system_info'):
            try:
                system_info = self.rag_agent.get_system_info()
                if system_info.get('system_ready', False):
                    embedding_model = system_info.get('embedding_model', 'Modern AI')
                    return f"Modern AI RAG ({embedding_model})", "🧠"
            except:
                pass
        
        # Check for vectorized system
        if hasattr(self.rag_agent, 'is_vectorized') and getattr(self.rag_agent, 'is_vectorized', False):
            return "Enhanced Vectorized RAG", "🔢"
        
        # Fallback to legacy
        return "Enhanced RAG", "📋"
    
    def _create_enhanced_sources_section(self, retrieval_results: list, system_type: str) -> str:
        """NEW helper - Create enhanced sources section with smart visualization"""
        
        if not retrieval_results:
            return "📚 **Sources:** No sources retrieved\n"
        
        # Dynamic section title based on system type
        if "Modern AI" in system_type:
            sources_section = "📚 **Semantic AI Sources:**\n"
        elif "Vectorized" in system_type:
            sources_section = "📚 **Vectorized Sources:**\n"
        else:
            sources_section = "📚 **Research Sources:**\n"
        
        for i, result in enumerate(retrieval_results, 1):
            # Enhanced similarity visualization
            similarity_percentage = result.similarity_score * 100
            
            # Smart confidence indicators
            if similarity_percentage >= 80:
                confidence_emoji = "🟢"
                confidence_text = "Excellent Match"
            elif similarity_percentage >= 60:
                confidence_emoji = "🟡" 
                confidence_text = "Very Good Match"
            elif similarity_percentage >= 40:
                confidence_emoji = "🟠"
                confidence_text = "Good Match"
            elif similarity_percentage >= 20:
                confidence_emoji = "🔴"
                confidence_text = "Fair Match"
            else:
                confidence_emoji = "⚪"
                confidence_text = "Low Match"
            
            # Enhanced similarity bar
            bar_length = 10
            filled_bars = int(similarity_percentage / 10)
            similarity_bar = "█" * filled_bars + "░" * (bar_length - filled_bars)
            
            # Format source entry
            sources_section += f"   {i}. **{result.source}** "
            sources_section += f"{confidence_emoji} {confidence_text} ({similarity_percentage:.1f}%)\n"
            sources_section += f"      📊 Relevance: {similarity_bar}\n"
            
            # Enhanced matched terms display
            if hasattr(result, 'matched_terms') and result.matched_terms:
                clean_terms = [term for term in result.matched_terms[:4] if len(term) > 2]
                if clean_terms:
                    sources_section += f"      🔑 Key Terms: `{', '.join(clean_terms)}`\n"
            
            # Chunk information
            if hasattr(result, 'chunk_info') and result.chunk_info:
                sources_section += f"      📄 {result.chunk_info}\n"
            
        return sources_section + "\n"
    
    def _create_enhanced_footer(self, retrieval_results: list, current_time: str, system_type: str) -> str:
        """NEW helper - Create enhanced footer with comprehensive system info"""
        
        # Get system statistics
        stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
        
        footer = f"🔍 **Enhanced RAG Summary:** "
        footer += f"Retrieved {len(retrieval_results)} relevant results"
        
        # Add system-specific information
        if "Modern AI" in system_type:
            try:
                system_info = self.rag_agent.get_system_info()
                embedding_model = system_info.get('embedding_model', 'AI')
                memory_usage = system_info.get('memory_usage_mb', 0)
                footer += f" • AI: {embedding_model} • Memory: {memory_usage:.1f}MB"
            except:
                footer += f" • Modern AI enabled"
        elif "Vectorized" in system_type:
            vector_dims = stats.get('vector_dimensions', 0)
            idf_terms = stats.get('idf_terms', 0)
            footer += f" • Vector dims: {vector_dims:,} • IDF terms: {idf_terms:,}"
        else:
            total_chunks = stats.get('chunks', 0)
            footer += f" • Searched {total_chunks:,} chunks"
        
        footer += f" • {current_time} UTC"
        
        return footer
    
    def format_system_status(self, user_id: str, current_time: str) -> str:
        """SAME FUNCTION NAME - Enhanced system status with smart detection"""
        
        # Get comprehensive stats
        stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
        system_type, system_emoji = self._detect_system_type()
        
        # Enhanced header
        status = f"🔧 **Enhanced RAG System Status** {system_emoji}\n"
        status += f"📅 **Timestamp:** {current_time} UTC\n"
        status += f"👤 **Current User:** {user_id}\n\n"
        
        # Core system information
        papers_count = len(getattr(self.rag_agent, 'documents', []))
        chunks_count = stats.get('total_chunks', stats.get('chunks', 0))
        
        status += f"📚 **Knowledge Base:**\n"
        status += f"   • Papers loaded: **{papers_count}**\n"
        status += f"   • Searchable chunks: **{chunks_count:,}**\n"
        
        if stats.get('index_terms'):
            status += f"   • Index terms: **{stats['index_terms']:,}**\n"
        
        total_size = stats.get('total_size_kb', 0)
        if total_size > 0:
            if total_size > 1024:
                status += f"   • Total content: **{total_size/1024:.1f}MB**\n"
            else:
                status += f"   • Total content: **{total_size:,}KB**\n"
        
        status += f"\n🧠 **AI System:** {system_type}\n"
        
        # System-specific status information
        if "Modern AI" in system_type:
            status += self._format_modern_ai_status(stats)
        elif "Vectorized" in system_type:
            status += self._format_vectorized_status(stats)
        else:
            status += self._format_legacy_status(stats)
        
        # Ready status
        status += f"\n✅ **System Status:** Fully operational and ready for queries!\n"
        status += f"💡 **Usage:** Ask any research questions about your loaded papers."
        
        return status
    
    def _format_modern_ai_status(self, stats: dict) -> str:
        """NEW helper - Format modern AI system status"""
        try:
            system_info = self.rag_agent.get_system_info()
            
            ai_status = f"   • **Engine:** {system_info.get('embedding_model', 'Modern AI')}\n"
            ai_status += f"   • **Vector Dimensions:** {system_info.get('vector_dimension', 0):,}\n"
            ai_status += f"   • **Memory Usage:** {system_info.get('memory_usage_mb', 0):.1f}MB\n"
            ai_status += f"   • **Search Type:** Semantic similarity (FAISS)\n"
            ai_status += f"   • **Performance:** Ultra-fast neural search\n"
            
            return ai_status
        except:
            return f"   • **Status:** Modern AI enabled\n"
    
    def _format_vectorized_status(self, stats: dict) -> str:
        """NEW helper - Format vectorized system status"""
        vector_status = f"   • **Status:** ✅ Vectorization ACTIVE\n"
        vector_status += f"   • **Vector dimensions:** {stats.get('vector_dimensions', 0):,}\n"
        vector_status += f"   • **Vectorized chunks:** {stats.get('vectorized_chunks', 0):,}\n"
        vector_status += f"   • **IDF terms:** {stats.get('idf_terms', 0):,}\n"
        
        # Memory usage if available
        if hasattr(self.rag_agent, 'get_vectorization_info'):
            try:
                vector_info = self.rag_agent.get_vectorization_info()
                memory_usage = vector_info.get('memory_usage_mb', 0)
                vector_status += f"   • **Vector memory:** {memory_usage:.1f}MB\n"
            except:
                pass
        
        vector_status += f"   • **Search Type:** Mathematical similarity (cosine)\n"
        
        return vector_status
    
    def _format_legacy_status(self, stats: dict) -> str:
        """NEW helper - Format legacy system status"""
        legacy_status = f"   • **Status:** Enhanced keyword search\n"
        legacy_status += f"   • **Search Type:** Intelligent keyword matching\n"
        legacy_status += f"   • **Performance:** Fast and reliable\n"
        
        return legacy_status
    
    def format_vectorization_info(self, current_time: str) -> str:
        """SAME FUNCTION NAME - Enhanced vectorization information with smart detection"""
        
        # Check for modern AI system first
        if hasattr(self.rag_agent, 'get_system_info'):
            try:
                system_info = self.rag_agent.get_system_info()
                if system_info.get('system_ready', False):
                    return self._format_modern_ai_info(system_info, current_time)
            except:
                pass
        
        # Check for legacy vectorization
        if hasattr(self.rag_agent, 'get_vectorization_info'):
            try:
                vector_info = self.rag_agent.get_vectorization_info()
                return self._format_legacy_vectorization_info(vector_info, current_time)
            except:
                pass
        
        # Fallback for no vectorization
        return f"🔢 **Vectorization Status:** Not available • {current_time} UTC\n\n" \
               f"💡 **Current Mode:** Using enhanced keyword search\n" \
               f"🚀 **Upgrade:** Consider enabling vectorization for better results!"
    
    def _format_modern_ai_info(self, system_info: dict, current_time: str) -> str:
        """NEW helper - Format modern AI system information"""
        response = f"🧠 **Modern AI System Details** at {current_time} UTC:\n\n"
        response += f"✅ **Status:** Fully Operational (Modern AI)\n"
        response += f"🤖 **AI Engine:** {system_info.get('embedding_model', 'Advanced Neural Network')}\n"
        response += f"📊 **Vector Dimensions:** {system_info.get('vector_dimension', 0):,}\n"
        response += f"💾 **Memory Usage:** {system_info.get('memory_usage_mb', 0):.1f}MB\n"
        response += f"⚡ **Search Speed:** Ultra-fast semantic search\n"
        response += f"🎯 **Accuracy:** Neural similarity matching\n\n"
        
        response += f"💡 **Modern AI Benefits:**\n"
        response += f"   • **Semantic Understanding:** Understands meaning, not just keywords\n"
        response += f"   • **Context Awareness:** Finds related concepts automatically\n"
        response += f"   • **High Precision:** Advanced neural similarity scoring\n"
        response += f"   • **Fast Performance:** Optimized FAISS vector search\n"
        
        return response
    
    def _format_legacy_vectorization_info(self, vector_info: dict, current_time: str) -> str:
        """NEW helper - Format legacy vectorization information"""
        if vector_info.get('status') == 'vectorized':
            response = f"🔢 **Vectorization Details** at {current_time} UTC:\n\n"
            response += f"✅ **Status:** Fully Vectorized (Legacy Mode)\n"
            response += f"📊 **Vocabulary Size:** {vector_info.get('vocabulary_size', 0):,} unique terms\n"
            response += f"📄 **Vectorized Chunks:** {vector_info.get('vectorized_chunks', 0):,}\n"
            response += f"🎯 **Vector Dimensions:** {vector_info.get('vector_dimensions', 0):,}\n"
            response += f"📈 **IDF Terms:** {vector_info.get('idf_terms', 0):,}\n"
            response += f"💾 **Memory Usage:** {vector_info.get('memory_usage_mb', 0):.1f}MB\n"
            
            matrix_shape = vector_info.get('vector_matrix_shape', 'N/A')
            response += f"🔧 **Matrix Shape:** {matrix_shape}\n\n"
            
            response += f"💡 **Vectorization Benefits:**\n"
            response += f"   • **Mathematical Similarity:** Cosine similarity scoring\n"
            response += f"   • **Improved Accuracy:** Better than keyword matching\n"
            response += f"   • **Efficient Search:** Fast vector operations\n"
            response += f"   • **TF-IDF Weighting:** Smart term importance\n"
            
            return response
        else:
            reason = vector_info.get('reason', 'Not available')
            return f"🔢 **Vectorization Status:** {reason} • {current_time} UTC\n\n" \
                   f"💡 **Current Mode:** Using enhanced keyword search"
    
    # NEW enhanced functions (optional to use)
    def format_paper_list(self, user_id: str, current_time: str) -> str:
        """NEW function - Format comprehensive paper list"""
        documents = getattr(self.rag_agent, 'documents', [])
        
        if not documents:
            return f"📚 **Paper Collection for {user_id}**\n\n" \
                   f"🔍 **Status:** No papers loaded\n" \
                   f"💡 **Next Step:** Add PDF research papers to enable RAG capabilities\n" \
                   f"🕐 **Checked:** {current_time} UTC"
        
        response = f"📚 **Research Paper Collection for {user_id}**\n"
        response += f"📅 **Last Updated:** {current_time} UTC\n\n"
        
        for i, doc in enumerate(documents, 1):
            doc_name = getattr(doc, 'name', f'Document {i}')
            response += f"   {i}. **{doc_name}**\n"
            
            # Add metadata if available
            if hasattr(doc, 'metadata'):
                metadata = doc.metadata
                if metadata.get('pages'):
                    response += f"      📄 Pages: {metadata['pages']}\n"
                if metadata.get('size_kb'):
                    response += f"      💾 Size: {metadata['size_kb']}KB\n"
        
        stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
        chunks_count = stats.get('total_chunks', stats.get('chunks', 0))
        
        response += f"\n📊 **Collection Stats:**\n"
        response += f"   • Total papers: **{len(documents)}**\n"
        response += f"   • Searchable chunks: **{chunks_count:,}**\n"
        response += f"   • System ready: **{'Yes' if chunks_count > 0 else 'No'}**"
        
        return response
    
    def get_formatting_stats(self) -> Dict[str, Any]:
        """NEW function - Get response formatting statistics"""
        system_type, _ = self._detect_system_type()
        stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
        
        return {
            'system_type': system_type,
            'papers_loaded': len(getattr(self.rag_agent, 'documents', [])),
            'total_chunks': stats.get('total_chunks', stats.get('chunks', 0)),
            'supports_modern_ai': hasattr(self.rag_agent, 'get_system_info'),
            'supports_vectorization': hasattr(self.rag_agent, 'get_vectorization_info'),
            'current_user': self.current_user,
            'last_updated': '2025-06-12 12:31:49'
        }