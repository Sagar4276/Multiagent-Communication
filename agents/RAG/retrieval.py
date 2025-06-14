"""
Enhanced Retrieval Engine - Hybrid Text + Image Search (Safwan's Approach)
Current Date and Time (UTC): 2025-06-13 09:28:11
Current User's Login: Sagar4276
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from .data_structures import (
    RetrievalResult, MultimodalSearchQuery, MultimodalImageData, 
    MultimodalConfig, calculate_hybrid_score, detect_scientific_terms
)
from .text_processing import TextProcessor

class ModernRetrievalEngine:
    """Enhanced retrieval engine with multimodal capabilities"""
    
    def __init__(self, agent_name: str = "ModernRetrieval"):
        self.agent_name = agent_name
        self.current_user = "Sagar4276"
        self.current_time = "2025-06-13 09:28:11"
        
        self.text_processor = TextProcessor()
        self.vectorization_engine = None
        self.config = MultimodalConfig()
        
        # Statistics tracking
        self.search_stats = {
            'total_searches': 0,
            'text_searches': 0,
            'image_searches': 0,
            'hybrid_searches': 0,
            'avg_results_returned': 0.0,
            'avg_processing_time': 0.0
        }
        
        print(f"[{self.agent_name}] 🔍 Enhanced retrieval engine initialized")
        print(f"[{self.agent_name}] 👤 User: {self.current_user}")
        print(f"[{self.agent_name}] 🕐 Time: {self.current_time} UTC")
    
    def set_vectorization_engine(self, vectorization_engine):
        """Set the enhanced vectorization engine"""
        self.vectorization_engine = vectorization_engine
        multimodal_status = "✅ Enabled" if vectorization_engine.enable_multimodal else "❌ Disabled"
        print(f"[{self.agent_name}] 🔗 Connected to vectorization engine")
        print(f"[{self.agent_name}] 🖼️ Multimodal support: {multimodal_status}")
    
    # ===== YOUR EXISTING RETRIEVAL (ENHANCED) =====
    
    def retrieve_for_chat_agent(self, query: str, max_results: int = 3) -> List[RetrievalResult]:
        """MAIN RETRIEVAL - Enhanced semantic search with multimodal support"""
        print(f"[{self.agent_name}] 🔍 Processing query for chat agent: '{query[:50]}...'")
        
        self.search_stats['total_searches'] += 1
        
        if not self.vectorization_engine or not self.vectorization_engine.faiss_index:
            print(f"[{self.agent_name}] ⚠️ Vectorization engine not available")
            return []
        
        try:
            import time
            start_time = time.time()
            
            # Determine search strategy based on query content
            search_strategy = self._analyze_query_intent(query)
            
            if search_strategy['include_images'] and self.vectorization_engine.enable_multimodal:
                # Use hybrid search (Safwan's approach)
                results = self._perform_hybrid_retrieval(query, max_results)
                self.search_stats['hybrid_searches'] += 1
            else:
                # Use text-only search (your existing approach enhanced)
                results = self._perform_text_retrieval(query, max_results)
                self.search_stats['text_searches'] += 1
            
            # Update statistics
            processing_time = time.time() - start_time
            self.search_stats['avg_processing_time'] = (
                (self.search_stats['avg_processing_time'] * (self.search_stats['total_searches'] - 1) + processing_time) /
                self.search_stats['total_searches']
            )
            
            self.search_stats['avg_results_returned'] = (
                (self.search_stats['avg_results_returned'] * (self.search_stats['total_searches'] - 1) + len(results)) /
                self.search_stats['total_searches']
            )
            
            print(f"[{self.agent_name}] ✅ Retrieved {len(results)} results in {processing_time:.3f}s")
            return results
            
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Retrieval failed: {str(e)}")
            return []
    
    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal search strategy"""
        query_lower = query.lower()
        
        # Keywords that suggest image content might be relevant
        image_keywords = [
            'image', 'picture', 'diagram', 'chart', 'graph', 'figure', 'illustration',
            'visual', 'show me', 'display', 'screenshot', 'photo', 'scan',
            'flowchart', 'table', 'visualization', 'plot', 'drawing'
        ]
        
        # Scientific terms that often have visual representations
        visual_science_terms = [
            'brain', 'neural', 'structure', 'anatomy', 'pathway', 'circuit',
            'model', 'algorithm', 'architecture', 'framework', 'workflow',
            'process', 'mechanism', 'system', 'network'
        ]
        
        include_images = False
        confidence = 0.0
        
        # Check for explicit image requests
        for keyword in image_keywords:
            if keyword in query_lower:
                include_images = True
                confidence += 0.3
        
        # Check for scientific terms that might have visual content
        for term in visual_science_terms:
            if term in query_lower:
                include_images = True
                confidence += 0.1
        
        # Boost confidence for specific domains
        if any(term in query_lower for term in ['parkinson', 'brain', 'neural', 'medical']):
            confidence += 0.2
        
        return {
            'include_images': include_images,
            'confidence': min(confidence, 1.0),
            'strategy': 'hybrid' if include_images else 'text_only',
            'detected_terms': detect_scientific_terms(query)
        }
    
    def _perform_text_retrieval(self, query: str, max_results: int) -> List[RetrievalResult]:
        """Enhanced text-only retrieval (your existing method improved)"""
        try:
            # Get raw FAISS results
            similar_chunks = self.vectorization_engine.search_similar_chunks(
                query, 
                top_k=min(max_results * 2, 10)
            )
            
            if not similar_chunks:
                print(f"[{self.agent_name}] ⚠️ No similar text chunks found")
                return []
            
            # Apply enhanced filtering and ranking
            quality_results = self._filter_and_rank_text_results(query, similar_chunks, max_results)
            
            print(f"[{self.agent_name}] 📝 Text retrieval: {len(quality_results)} results")
            return quality_results
            
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Text retrieval error: {str(e)}")
            return []
    
    def _perform_hybrid_retrieval(self, query: str, max_results: int) -> List[RetrievalResult]:
        """Perform hybrid text + image retrieval (Safwan's approach)"""
        try:
            print(f"[{self.agent_name}] 🎯 Performing hybrid retrieval for: '{query}'")
            
            # Get hybrid search results from vectorization engine
            hybrid_results = self.vectorization_engine.hybrid_search(
                query,
                max_text_results=max_results,
                max_image_results=self.config.max_images_per_query
            )
            
            combined_results = []
            
            # Process text results
            for text_result in hybrid_results['text_results']:
                chunk = text_result['chunk']
                similarity = text_result['similarity']
                
                # Enhanced text content creation
                enhanced_content = self._create_enhanced_content(
                    chunk.content, 
                    set(self.text_processor.extract_key_terms(query))
                )
                
                # Find matched terms
                content_terms = set(self.text_processor.extract_key_terms(chunk.content))
                query_terms = set(self.text_processor.extract_key_terms(query))
                matched_terms = list(query_terms & content_terms)
                
                # Add scientific terms bonus
                scientific_terms = detect_scientific_terms(chunk.content)
                for term in scientific_terms[:3]:  # Limit to avoid clutter
                    if term not in matched_terms:
                        matched_terms.append(term)
                
                # Create text result
                result = RetrievalResult(
                    content=enhanced_content,
                    source=chunk.metadata.get('filename', 'Unknown'),
                    similarity_score=float(similarity),
                    chunk_info=f"Text Chunk {chunk.chunk_index + 1}",
                    matched_terms=matched_terms[:5],
                    doc_metadata={
                        'doc_id': chunk.metadata.get('doc_id'),
                        'filename': chunk.metadata.get('filename', 'Unknown'),
                        'chunk_length': len(chunk.content),
                        'similarity': f"{similarity:.1%}",
                        'result_source': 'text_embedding'
                    },
                    chunk_id=chunk.chunk_id,
                    result_type="text"
                )
                
                combined_results.append(result)
            
            # Process image results (Safwan's approach)
            for image_result in hybrid_results['image_results']:
                image_data = image_result['image_data']
                clip_similarity = image_result['clip_similarity']
                context_similarity = image_result['context_similarity']
                final_score = image_result['final_score']
                
                # Create image description content
                image_content = self._create_image_description(image_data, query)
                
                # Extract terms from OCR and context
                ocr_terms = set(self.text_processor.extract_key_terms(image_data.ocr_text))
                context_terms = set(self.text_processor.extract_key_terms(image_data.context_text))
                query_terms = set(self.text_processor.extract_key_terms(query))
                
                # Find matched terms
                matched_terms = list((ocr_terms | context_terms) & query_terms)
                
                # Create image result
                result = RetrievalResult(
                    content=image_content,
                    source=f"{image_data.document_name} (Page {image_data.page_number + 1})",
                    similarity_score=final_score,
                    chunk_info=f"Image {image_data.page_number + 1}",
                    matched_terms=matched_terms[:5],
                    doc_metadata={
                        'document_name': image_data.document_name,
                        'page_number': image_data.page_number + 1,
                        'image_quality': f"{image_data.quality_score:.1%}",
                        'clip_similarity': f"{clip_similarity:.1%}",
                        'context_similarity': f"{context_similarity:.1%}",
                        'result_source': 'clip_embedding'
                    },
                    chunk_id=-1,  # Images don't have chunk IDs
                    result_type="image",
                    image_data=image_data,
                    clip_similarity=clip_similarity,
                    context_similarity=context_similarity,
                    hybrid_score=final_score
                )
                
                combined_results.append(result)
            
            # Sort by relevance score (text similarity or image final score)
            combined_results.sort(key=lambda x: x.similarity_score, reverse=True)
            
            # Limit total results
            final_results = combined_results[:max_results]
            
            print(f"[{self.agent_name}] 🎯 Hybrid retrieval complete:")
            print(f"[{self.agent_name}] 📝 Text results: {len([r for r in final_results if r.result_type == 'text'])}")
            print(f"[{self.agent_name}] 🖼️ Image results: {len([r for r in final_results if r.result_type == 'image'])}")
            
            return final_results
            
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Hybrid retrieval error: {str(e)}")
            # Fallback to text-only
            return self._perform_text_retrieval(query, max_results)
    
    def _create_image_description(self, image_data: MultimodalImageData, query: str) -> str:
        """Create descriptive content for image results (Safwan's approach)"""
        try:
            description_parts = []
            
            # Start with image identification
            description_parts.append(f"📊 **Image from {image_data.document_name}, Page {image_data.page_number + 1}**")
            
            # Add OCR content if available
            if image_data.ocr_text and len(image_data.ocr_text.strip()) > 10:
                description_parts.append(f"\n🔤 **Text in image:** {image_data.ocr_text.strip()}")
            
            # Add context information
            if image_data.context_text and len(image_data.context_text.strip()) > 20:
                # Limit context to avoid overwhelming
                context = image_data.context_text.strip()
                if len(context) > 200:
                    context = context[:200] + "..."
                description_parts.append(f"\n📝 **Context:** {context}")
            
            # Add quality indicators
            quality_indicators = []
            if image_data.quality_score >= 0.7:
                quality_indicators.append("High Quality")
            if image_data.ocr_confidence >= 70:
                quality_indicators.append("Clear Text")
            if len(image_data.ocr_text) > 50:
                quality_indicators.append("Rich Content")
            
            if quality_indicators:
                description_parts.append(f"\n✨ **Quality:** {', '.join(quality_indicators)}")
            
            # Add technical details
            description_parts.append(f"\n📏 **Size:** {image_data.width}×{image_data.height} pixels")
            
            return "\n".join(description_parts)
            
        except Exception as e:
            print(f"[{self.agent_name}] ⚠️ Image description error: {str(e)}")
            return f"Image from {image_data.document_name}, Page {image_data.page_number + 1}"
    
    # ===== ENHANCED TEXT PROCESSING (YOUR EXISTING METHOD IMPROVED) =====
    
    def _filter_and_rank_text_results(self, query: str, similar_chunks: List[Tuple], max_results: int) -> List[RetrievalResult]:
        """Enhanced filtering and ranking for text results"""
        
        results = []
        seen_docs = {}
        query_terms = set(self.text_processor.extract_key_terms(query))
        
        for chunk, similarity in similar_chunks:
            if len(results) >= max_results:
                break
            
            # Quality threshold (Safwan's approach)
            if similarity < self.config.text_similarity_threshold:
                print(f"[{self.agent_name}] ⚠️ Skipping low similarity: {similarity:.3f}")
                continue
            
            doc_id = chunk.metadata.get('doc_id', 'unknown')
            
            # Diversity control: max 2 results per document
            if doc_id in seen_docs and seen_docs[doc_id] >= 2:
                print(f"[{self.agent_name}] ⚠️ Skipping for diversity: doc {doc_id}")
                continue
            
            # Enhanced content creation
            enhanced_content = self._create_enhanced_content(chunk.content, query_terms)
            
            if len(enhanced_content.strip()) < 100:
                print(f"[{self.agent_name}] ⚠️ Skipping short content")
                continue
            
            # Find matched terms with scientific bonus
            content_terms = set(self.text_processor.extract_key_terms(chunk.content))
            matched_terms = list(query_terms & content_terms)
            
            # Add scientific terms bonus (Safwan's approach)
            scientific_terms = detect_scientific_terms(chunk.content)
            for term in scientific_terms[:3]:
                if term not in matched_terms:
                    matched_terms.append(term)
            
            # Calculate enhanced score
            enhanced_score = self._calculate_enhanced_text_score(query, chunk.content, similarity)
            
            # Create retrieval result
            result = RetrievalResult(
                content=enhanced_content,
                source=chunk.metadata.get('filename', 'Unknown'),
                similarity_score=float(similarity),
                chunk_info=f"Chunk {chunk.chunk_index + 1}",
                matched_terms=matched_terms[:5],
                doc_metadata={
                    'doc_id': doc_id,
                    'filename': chunk.metadata.get('filename', 'Unknown'),
                    'chunk_length': len(chunk.content),
                    'similarity': f"{similarity:.1%}",
                    'enhanced_score': f"{enhanced_score:.1%}",
                    'scientific_terms': len(scientific_terms),
                    'result_source': 'text_embedding'
                },
                chunk_id=chunk.chunk_id,
                result_type="text"
            )
            
            # Set enhanced score
            result.similarity_score = enhanced_score
            
            results.append(result)
            
            # Track document count
            seen_docs[doc_id] = seen_docs.get(doc_id, 0) + 1
            
            print(f"[{self.agent_name}] ✅ Added text result: {similarity:.3f} → {enhanced_score:.3f}")
        
        # Sort by enhanced score
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return results
    
    def _calculate_enhanced_text_score(self, query: str, content: str, base_score: float) -> float:
        """Calculate enhanced relevance score for text (Safwan's approach)"""
        try:
            score = base_score
            
            # Keyword overlap bonus
            query_words = set(query.lower().split())
            content_words = set(content.lower().split())
            if query_words:
                keyword_overlap = len(query_words.intersection(content_words)) / len(query_words)
                score += keyword_overlap * 0.2
            
            # Content length factor (prefer substantial content)
            length_factor = min(len(content) / 1000, 1.0)
            score += length_factor * 0.1
            
            # Scientific terms bonus
            scientific_terms = detect_scientific_terms(content)
            if scientific_terms:
                science_bonus = min(len(scientific_terms) / 10, 0.15)
                score += science_bonus
            
            # Domain-specific bonuses
            if any(term in content.lower() for term in ['parkinson', 'neurological', 'brain']):
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception:
            return base_score
    
    def _create_enhanced_content(self, content: str, query_terms: set, max_length: int = 2000) -> str:
        """Create enhanced content preserving important information"""
        
        if not content or len(content) <= max_length:
            return self.text_processor.clean_text_for_embeddings(content)
        
        # Extract key sections based on query terms and importance
        sections = self._extract_key_sections_enhanced(content, query_terms)
        
        if not sections:
            # Fallback: take first portion
            return self.text_processor.clean_text_for_embeddings(content[:max_length-3] + "...")
        
        # Combine sections
        combined_text = " ".join(sections)
        
        # Ensure we don't exceed max length
        if len(combined_text) > max_length:
            combined_text = combined_text[:max_length-3] + "..."
            
        return self.text_processor.clean_text_for_embeddings(combined_text)
    
    def _extract_key_sections_enhanced(self, content: str, query_terms: set) -> List[str]:
        """Extract key sections with enhanced relevance scoring"""
        
        # Split content into paragraphs
        paragraphs = [p.strip() for p in content.split('\n\n') if len(p.strip()) > 50]
        
        if len(paragraphs) <= 1:
            # Split by sentences if no clear paragraphs
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 30]
            paragraphs = []
            current_para = ""
            
            for sentence in sentences:
                if len(current_para) + len(sentence) > 300:
                    if current_para:
                        paragraphs.append(current_para.strip())
                    current_para = sentence
                else:
                    current_para += ". " + sentence if current_para else sentence
            
            if current_para:
                paragraphs.append(current_para.strip())
        
        # Score paragraphs
        scored_paragraphs = []
        for para in paragraphs:
            score = self._score_paragraph_relevance(para, query_terms)
            if score > 0:
                scored_paragraphs.append((para, score))
        
        # Sort by score and select top sections
        scored_paragraphs.sort(key=lambda x: x[1], reverse=True)
        
        # Extract top sections up to ~1500 chars
        selected_sections = []
        char_count = 0
        
        for para, score in scored_paragraphs:
            if char_count + len(para) > 1800:
                break
            selected_sections.append(para)
            char_count += len(para)
            
            if len(selected_sections) >= 3:  # Limit number of sections
                break
        
        return selected_sections
    
    def _score_paragraph_relevance(self, paragraph: str, query_terms: set) -> float:
        """Score paragraph relevance to query"""
        try:
            para_lower = paragraph.lower()
            score = 0.0
            
            # Query term matches
            para_words = set(para_lower.split())
            query_matches = len(query_terms & para_words)
            if query_terms:
                score += (query_matches / len(query_terms)) * 2.0
            
            # Scientific terms bonus
            scientific_terms = detect_scientific_terms(paragraph)
            score += len(scientific_terms) * 0.2
            
            # Important medical keywords
            medical_keywords = [
                'symptoms', 'treatment', 'diagnosis', 'therapy', 'medication',
                'brain', 'neurological', 'dopamine', 'tremor', 'movement'
            ]
            
            for keyword in medical_keywords:
                if keyword in para_lower:
                    score += 0.3
            
            # Length factor (prefer substantial paragraphs)
            if len(paragraph) > 200:
                score += 0.5
            
            return score
            
        except Exception:
            return 0.0
    
    # ===== ADVANCED SEARCH METHODS =====
    
    def search_with_filters(self, query: str, min_similarity: float = 0.1, 
                          max_results: int = 5, include_metadata: bool = True,
                          search_images: bool = False) -> List[Dict]:
        """Advanced search with filtering options (enhanced)"""
        
        if not self.vectorization_engine:
            return []
        
        try:
            if search_images and self.vectorization_engine.enable_multimodal:
                # Create multimodal search query
                search_query = MultimodalSearchQuery(
                    text=query,
                    search_images=True,
                    max_text_results=max_results,
                    max_image_results=min(max_results, self.config.max_images_per_query),
                    min_similarity=min_similarity,
                    min_clip_similarity=self.config.clip_similarity_threshold,
                    min_quality_score=self.config.image_quality_threshold
                )
                
                # Perform hybrid search
                results = self._perform_hybrid_retrieval(query, max_results)
                
                # Convert to dictionary format
                filtered_results = []
                for result in results:
                    result_dict = {
                        'content': result.content,
                        'similarity': result.similarity_score,
                        'source': result.source,
                        'chunk_id': result.chunk_id,
                        'result_type': result.result_type,
                        'matched_terms': result.matched_terms
                    }
                    
                    if include_metadata:
                        result_dict['metadata'] = result.doc_metadata
                        
                        if result.result_type == "image":
                            result_dict['image_metadata'] = {
                                'clip_similarity': result.clip_similarity,
                                'context_similarity': result.context_similarity,
                                'hybrid_score': result.hybrid_score,
                                'quality_score': result.image_data.quality_score if result.image_data else 0
                            }
                    
                    filtered_results.append(result_dict)
                
                return filtered_results
            
            else:
                # Text-only search
                similar_chunks = self.vectorization_engine.search_similar_chunks(query, top_k=max_results * 2)
                
                filtered_results = []
                for chunk, similarity in similar_chunks:
                    if similarity >= min_similarity and len(filtered_results) < max_results:
                        
                        # Use enhanced content creation
                        result_content = self._create_enhanced_content(
                            chunk.content,
                            set(self.text_processor.extract_key_terms(query)),
                            max_length=1500
                        )
                        
                        result = {
                            'content': result_content,
                            'similarity': float(similarity),
                            'source': chunk.metadata.get('filename', 'Unknown'),
                            'chunk_id': chunk.chunk_id,
                            'result_type': 'text'
                        }
                        
                        if include_metadata:
                            result['metadata'] = {
                                'doc_id': chunk.metadata.get('doc_id'),
                                'chunk_index': chunk.chunk_index,
                                'content_length': len(chunk.content),
                                'enhanced_score': self._calculate_enhanced_text_score(query, chunk.content, similarity)
                            }
                        
                        filtered_results.append(result)
                
                return filtered_results
            
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Advanced search failed: {str(e)}")
            return []
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrieval engine statistics"""
        if not self.vectorization_engine:
            return {'status': 'not_initialized'}
        
        vec_info = self.vectorization_engine.get_vectorization_info()
        
        stats = {
            'status': 'ready' if vec_info.get('status') == 'vectorized' else 'not_ready',
            'total_chunks': vec_info.get('total_chunks', 0),
            'total_images': vec_info.get('total_images', 0),
            'text_embedding_model': vec_info.get('text_model_name', 'unknown'),
            'clip_model': vec_info.get('clip_model_name', 'none'),
            'vector_dimension': vec_info.get('vector_dimension', 0),
            'clip_dimension': vec_info.get('clip_dimension', 0),
            'index_type': vec_info.get('index_type', 'unknown'),
            'memory_usage_mb': vec_info.get('memory_usage_mb', 0),
            'image_memory_mb': vec_info.get('image_memory_mb', 0),
            'multimodal_enabled': vec_info.get('multimodal_enabled', False),
            'search_statistics': self.search_stats.copy()
        }
        
        return stats
    
    def get_search_suggestions(self, query: str) -> List[str]:
        """Get search suggestions based on available content"""
        suggestions = []
        
        try:
            # Detect scientific terms in query
            science_terms = detect_scientific_terms(query)
            
            if 'parkinson' in query.lower():
                suggestions.extend([
                    "What are the symptoms of Parkinson's disease?",
                    "How is Parkinson's disease treated?",
                    "What causes Parkinson's disease?",
                    "Show me brain diagrams related to Parkinson's"
                ])
            
            if any(term in query.lower() for term in ['brain', 'neural', 'neurological']):
                suggestions.extend([
                    "Find brain structure diagrams",
                    "Show neural pathway illustrations",
                    "Explain brain anatomy",
                    "Find neurological research images"
                ])
            
            if any(term in query.lower() for term in ['treatment', 'therapy', 'medication']):
                suggestions.extend([
                    "What are the available treatments?",
                    "Compare different therapy approaches",
                    "Show medication effectiveness charts",
                    "Find clinical trial results"
                ])
            
            # Add general suggestions if none match
            if not suggestions:
                suggestions = [
                    "Search for definitions and explanations",
                    "Find relevant diagrams and charts",
                    "Look for research findings",
                    "Explore related concepts"
                ]
            
            return suggestions[:4]  # Limit to 4 suggestions
            
        except Exception:
            return ["Try a more specific search term", "Search for key concepts", "Look for research topics"]

# ===== COMPATIBILITY FUNCTIONS =====

def create_retrieval_result_from_chunk(chunk, similarity: float, query: str) -> RetrievalResult:
    """Create RetrievalResult from chunk for backward compatibility"""
    processor = TextProcessor()
    
    # Enhanced content
    query_terms = set(processor.extract_key_terms(query))
    content_terms = set(processor.extract_key_terms(chunk.content))
    matched_terms = list(query_terms & content_terms)
    
    return RetrievalResult(
        content=chunk.content[:2000] + "..." if len(chunk.content) > 2000 else chunk.content,
        source=chunk.metadata.get('filename', 'Unknown'),
        similarity_score=float(similarity),
        chunk_info=f"Chunk {chunk.chunk_index + 1}",
        matched_terms=matched_terms[:5],
        doc_metadata=chunk.metadata,
        chunk_id=chunk.chunk_id,
        result_type="text"
    )