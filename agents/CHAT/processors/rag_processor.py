"""
Enhanced Chat Agent - RAG Processing Module (UPDATED)
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-12 12:25:15
Current User's Login: Sagar4276
"""

from typing import List, Dict, Any, Optional
from agents.RAG.rag_agent import RetrievalResult

class RAGProcessor:
    """SAME CLASS NAME - Enhanced RAG processing and context creation"""
    
    def __init__(self, rag_agent, agent_name: str = "RAGProcessor"):
        self.agent_name = agent_name
        self.rag_agent = rag_agent
        self.current_user = "Sagar4276"
    
    def process_research_query(self, user_id: str, message: str, history: list) -> Dict[str, Any]:
        """SAME FUNCTION NAME - Enhanced research query processing with smart RAG detection"""
        try:
            print(f"[{self.agent_name}] 🔍 Processing research query for {user_id}: '{message}'")
            
            # STEP 1: SMART RETRIEVE with enhanced detection
            retrieval_results = self._perform_smart_retrieval(message)
            
            # Enhanced result validation
            if not retrieval_results or all(result.similarity_score <= 0.01 for result in retrieval_results):
                return {
                    'success': False,
                    'reason': 'no_results',
                    'retrieval_results': [],
                    'context': '',
                    'message': f"No relevant content found for '{message}'",
                    'search_attempted': True,
                    'user_id': user_id
                }
            
            # Enhanced quality filtering with adaptive thresholds
            quality_results = self._filter_quality_results(retrieval_results, message)
            
            print(f"[{self.agent_name}] 📚 Found {len(quality_results)} high-quality results")
            for i, result in enumerate(quality_results, 1):
                print(f"   {i}. {result.source} (similarity: {result.similarity_score:.3f})")
            
            # STEP 2: ENHANCED AUGMENT with smart context creation
            enhanced_context = self._create_enhanced_context(quality_results, message)
            
            return {
                'success': True,
                'retrieval_results': quality_results,
                'context': enhanced_context,
                'message': 'RAG processing successful',
                'search_type': self._get_search_type(),
                'user_id': user_id
            }
            
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Enhanced RAG processing error: {str(e)}")
            return {
                'success': False,
                'reason': 'processing_error',
                'retrieval_results': [],
                'context': '',
                'message': f"RAG processing error: {str(e)}",
                'error_details': str(e),
                'user_id': user_id
            }
    
    def _perform_smart_retrieval(self, message: str) -> List[RetrievalResult]:
        """NEW helper - Smart retrieval with multiple fallback strategies"""
        
        # Try modern FAISS first (if available)
        if hasattr(self.rag_agent, 'get_system_info') and self.rag_agent.get_stats().get('system_ready', False):
            print(f"[{self.agent_name}] 🧠 Using modern FAISS semantic search...")
            try:
                return self.rag_agent.retrieve_for_chat_agent(message, max_results=5)
            except Exception as e:
                print(f"[{self.agent_name}] ⚠️ FAISS search failed: {e}, falling back...")
        
        # Try vectorized search (if available)
        elif hasattr(self.rag_agent, 'is_vectorized') and self.rag_agent.is_vectorized:
            print(f"[{self.agent_name}] 🔢 Using vectorized retrieval...")
            try:
                return self.rag_agent.retrieve_for_chat_agent(message, max_results=4)
            except Exception as e:
                print(f"[{self.agent_name}] ⚠️ Vectorized search failed: {e}, falling back...")
        
        # Fallback to legacy search
        print(f"[{self.agent_name}] 🔄 Using legacy search...")
        try:
            legacy_results = self.rag_agent.search_papers(message, max_results=3)
            return self._convert_legacy_to_retrieval_results(legacy_results)
        except Exception as e:
            print(f"[{self.agent_name}] ❌ All search methods failed: {e}")
            return []
    
    def _get_search_type(self) -> str:
        """NEW helper - Determine which search type was used"""
        if hasattr(self.rag_agent, 'get_system_info') and self.rag_agent.get_stats().get('system_ready', False):
            return "modern_faiss"
        elif hasattr(self.rag_agent, 'is_vectorized') and self.rag_agent.is_vectorized:
            return "vectorized"
        else:
            return "legacy"
    
    def _filter_quality_results(self, retrieval_results: List[RetrievalResult], message: str) -> List[RetrievalResult]:
        """NEW helper - Enhanced quality filtering with adaptive thresholds"""
        
        # Adaptive threshold based on best result
        if not retrieval_results:
            return []
        
        best_score = max(result.similarity_score for result in retrieval_results)
        
        # Dynamic threshold calculation
        if best_score > 0.7:
            threshold = 0.3  # High best score allows lower threshold
        elif best_score > 0.5:
            threshold = 0.15  # Medium best score
        elif best_score > 0.2:
            threshold = 0.08  # Low best score, very permissive
        else:
            threshold = 0.05  # Very low scores, take what we can get
        
        # Filter with adaptive threshold
        quality_results = [r for r in retrieval_results if r.similarity_score > threshold]
        
        # Ensure we have at least one result if any exist
        if not quality_results and retrieval_results:
            quality_results = [max(retrieval_results, key=lambda x: x.similarity_score)]
        
        # Limit to top 4 results for processing efficiency
        return quality_results[:4]
    
    def _convert_legacy_to_retrieval_results(self, legacy_results) -> List[RetrievalResult]:
        """SAME FUNCTION NAME - Enhanced legacy result conversion"""
        retrieval_results = []
        
        for i, result in enumerate(legacy_results):
            if result.get('relevance', 0) > 0:
                # Enhanced RetrievalResult creation
                retrieval_result = RetrievalResult(
                    content=result['content'],
                    source=result['source'],
                    similarity_score=result['relevance'],
                    chunk_info=f"Legacy chunk {i+1}",
                    matched_terms=result.get('matched_terms', []),
                    doc_metadata={
                        'legacy_search': True,
                        'original_relevance': result['relevance'],
                        'search_method': 'keyword_matching'
                    },
                    chunk_id=i
                )
                retrieval_results.append(retrieval_result)
        
        return retrieval_results
    
    def _create_enhanced_context(self, retrieval_results: List[RetrievalResult], query: str) -> str:
        """SAME FUNCTION NAME - Enhanced context creation with better formatting"""
        context_parts = []
        
        for i, result in enumerate(retrieval_results, 1):
            # Enhanced similarity info with confidence levels
            similarity_percentage = result.similarity_score * 100
            
            # Confidence level determination
            if similarity_percentage >= 70:
                confidence_level = "🟢 High Confidence"
            elif similarity_percentage >= 50:
                confidence_level = "🟡 Medium Confidence"
            elif similarity_percentage >= 30:
                confidence_level = "🟠 Low Confidence"
            else:
                confidence_level = "🔴 Very Low Confidence"
            
            similarity_info = f"({confidence_level}: {similarity_percentage:.1f}%"
            
            # Enhanced matched terms display
            if result.matched_terms:
                clean_terms = [term for term in result.matched_terms[:4] if len(term) > 2]
                if clean_terms:
                    similarity_info += f", Key Terms: {', '.join(clean_terms)}"
            
            similarity_info += ")"
            
            # Enhanced context part formatting
            context_part = f"**Source {i}: {result.source}** {similarity_info}\n"
            context_part += f"{result.content.strip()}\n"
            
            # Enhanced chunk info
            chunk_details = result.chunk_info
            if hasattr(result, 'doc_metadata') and result.doc_metadata:
                if result.doc_metadata.get('legacy_search'):
                    chunk_details += " [Legacy Search]"
                elif result.doc_metadata.get('faiss_search'):
                    chunk_details += " [FAISS Search]"
            
            context_part += f"*{chunk_details}*\n"
            context_parts.append(context_part)
        
        # Smart context length management
        full_context = "\n".join(context_parts)
        
        # Enhanced truncation with better logic
        max_length = 1400  # Slightly increased for better context
        if len(full_context) > max_length:
            full_context = self._smart_truncate_context(context_parts, max_length)
        
        return full_context
    
    def _smart_truncate_context(self, context_parts: List[str], max_length: int) -> str:
        """NEW helper - Smart context truncation preserving important information"""
        truncated_parts = []
        current_length = 0
        
        for part in context_parts:
            if current_length + len(part) > max_length:
                # Calculate remaining space
                remaining_length = max_length - current_length
                
                if remaining_length > 150:  # Only truncate if meaningful space left
                    # Try to truncate at sentence boundary
                    sentences = part.split('. ')
                    truncated_content = ""
                    
                    for sentence in sentences:
                        if len(truncated_content + sentence + ". ") < remaining_length - 50:
                            truncated_content += sentence + ". "
                        else:
                            break
                    
                    if truncated_content:
                        truncated_part = truncated_content + "*(truncated for length)*"
                        truncated_parts.append(truncated_part)
                break
            
            truncated_parts.append(part)
            current_length += len(part)
        
        return "\n".join(truncated_parts)
    
    def create_rag_prompt(self, context: str, query: str) -> str:
        """SAME FUNCTION NAME - Enhanced prompt creation with smart adaptation"""
        
        # Determine search type for prompt customization
        search_type = self._get_search_type()
        
        if search_type == "modern_faiss":
            context_description = "semantic AI research context"
        elif search_type == "vectorized":
            context_description = "vectorized research context"
        else:
            context_description = "research context"
        
        # Enhanced prompt with better instructions
        return f"""Based on this {context_description}:

{context}

Research Question: {query}

INSTRUCTIONS:
- Provide a comprehensive, accurate answer based on the retrieved research
- Use clear, accessible language while maintaining technical accuracy
- Reference the sources when making specific claims
- Structure your response logically with key points
- If the context doesn't fully answer the question, indicate what information is available

ANSWER:"""
    
    def handle_no_results(self, user_id: str, query: str) -> str:
        """SAME FUNCTION NAME - Enhanced no-results handling with better guidance"""
        papers_count = len(getattr(self.rag_agent, 'documents', []))
        
        if papers_count > 0:
            stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
            search_type = self._get_search_type()
            
            # Enhanced response with search type awareness
            response = f"🔍 **Enhanced RAG Search Results for {user_id}**\n\n"
            
            # Search type specific messaging
            if search_type == "modern_faiss":
                response += f"I used **modern semantic AI search** through your **{papers_count} research papers** "
                response += f"({stats.get('total_chunks', stats.get('chunks', 0)):,} chunks) "
                response += f"for '{query}', but couldn't find highly relevant content.\n\n"
            elif search_type == "vectorized":
                response += f"I used **vectorized search** through your **{papers_count} research papers** "
                response += f"({stats.get('chunks', 0):,} chunks, {stats.get('vector_dimensions', 0):,} dimensions) "
                response += f"for '{query}', but couldn't find highly relevant content.\n\n"
            else:
                response += f"I searched through your **{papers_count} research papers** "
                response += f"({stats.get('chunks', 0):,} chunks) for '{query}', but couldn't find relevant content.\n\n"
            
            # Enhanced search suggestions
            response += f"💡 **Smart Search Suggestions:**\n"
            response += f"• **Be more specific:** Use technical terms and domain keywords\n"
            response += f"• **Try synonyms:** Different terms for the same concept\n"
            response += f"• **Check coverage:** Use 'show papers' to verify topic areas\n"
            response += f"• **Broader terms:** Try more general concepts first\n"
            response += f"• **Ask differently:** Rephrase your question with different keywords\n\n"
            
            # Enhanced statistics
            response += f"📊 **Search Performance:** "
            response += f"Processed {stats.get('total_chunks', stats.get('chunks', 0)):,} document sections"
            
            if search_type == "modern_faiss":
                response += f" using semantic AI understanding"
            elif search_type == "vectorized":
                response += f" using {stats.get('vector_dimensions', 0):,}-dimensional vectors"
            else:
                response += f" using keyword matching"
            
            response += f"\n🕐 **Search completed:** 2025-06-12 12:25:15 UTC"
            
            return response
        else:
            return f"📚 **No Research Papers Found**\n\nHello {user_id}! Your collection appears to be empty. Please add research papers to enable Enhanced RAG capabilities.\n\n💡 **Next Steps:** Load PDF documents to start using intelligent research search!"
    
    # NEW enhanced functions (optional to use)
    def get_processing_stats(self) -> Dict[str, Any]:
        """NEW function - Get RAG processing statistics"""
        stats = self.rag_agent.get_stats() if hasattr(self.rag_agent, 'get_stats') else {}
        
        return {
            'search_type': self._get_search_type(),
            'documents_loaded': len(getattr(self.rag_agent, 'documents', [])),
            'total_chunks': stats.get('total_chunks', stats.get('chunks', 0)),
            'vector_dimensions': stats.get('vector_dimensions', 0),
            'system_ready': stats.get('system_ready', False),
            'last_updated': '2025-06-12 12:25:15'
        }
    
    def validate_query_for_rag(self, message: str) -> Dict[str, Any]:
        """NEW function - Validate if query is suitable for RAG processing"""
        msg_lower = message.lower().strip()
        
        # Check query length
        word_count = len(msg_lower.split())
        
        # Check for research indicators
        research_terms = ['what', 'how', 'why', 'explain', 'describe', 'define', 'compare']
        has_research_terms = any(term in msg_lower for term in research_terms)
        
        # Calculate suitability score
        suitability_score = 0.5
        
        if word_count >= 3:
            suitability_score += 0.2
        if word_count >= 6:
            suitability_score += 0.1
        if has_research_terms:
            suitability_score += 0.3
        if any(term in msg_lower for term in ['research', 'study', 'paper', 'analysis']):
            suitability_score += 0.2
        
        return {
            'suitable_for_rag': suitability_score >= 0.7,
            'suitability_score': min(suitability_score, 1.0),
            'word_count': word_count,
            'has_research_terms': has_research_terms,
            'recommendation': 'suitable' if suitability_score >= 0.7 else 'marginal' if suitability_score >= 0.5 else 'not_suitable'
        }