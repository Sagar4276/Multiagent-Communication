"""
Enhanced RAG Agent - Multimodal Orchestration (Safwan's Architecture + Your Clean Design)
Current Date and Time (UTC): 2025-06-13 09:28:11
Current User's Login: Sagar4276
"""

import os
import time
from typing import List, Dict, Any, Optional, Tuple
from .data_structures import (
    RetrievalResult, DocumentInfo, MultimodalConfig, SystemStats, 
    ProcessingResult, create_multimodal_system_stats
)
from .text_processing import TextProcessor
from .vectorization import ModernVectorizationEngine
from .retrieval import ModernRetrievalEngine

class ModernEnhancedRAGAgent:
    """Enhanced RAG Agent with multimodal capabilities - Built on your clean architecture"""
    
    def __init__(self, papers_folder: str = "knowledge_base/papers", enable_multimodal: bool = True):
        self.name = "ModernEnhancedRAGAgent"
        self.papers_folder = papers_folder
        self.current_time = "2025-06-13 09:28:11"
        self.user = "Sagar4276"
        self.enable_multimodal = enable_multimodal
        
        print(f"[{self.name}] 🚀 Initializing Enhanced Multimodal RAG System...")
        print(f"[{self.name}] 👤 User: {self.user}")
        print(f"[{self.name}] 🕐 Time: {self.current_time} UTC")
        print(f"[{self.name}] 📂 Papers folder: {papers_folder}")
        print(f"[{self.name}] 🖼️ Multimodal enabled: {enable_multimodal}")
        
        # Configuration
        self.config = MultimodalConfig()
        
        # Initialize enhanced components
        self.text_processor = TextProcessor(self.user, enable_multimodal)
        self.vectorization_engine = ModernVectorizationEngine(
            cache_dir=papers_folder,
            enable_multimodal=enable_multimodal
        )
        self.retrieval_engine = ModernRetrievalEngine(self.name)
        
        # Connect components
        self.retrieval_engine.set_vectorization_engine(self.vectorization_engine)
        
        # Core data
        self.documents = []
        self.image_data = []  # NEW: Store extracted images
        self.is_ready = False
        
        # Processing statistics
        self.processing_stats = {
            'documents_processed': 0,
            'total_chunks': 0,
            'images_extracted': 0,
            'images_stored': 0,
            'last_processing_time': 0.0,
            'multimodal_active': enable_multimodal
        }
        
        # Initialize the enhanced system
        self._initialize_enhanced_system()
    
    def _initialize_enhanced_system(self):
        """Initialize the complete enhanced multimodal RAG system"""
        try:
            start_time = time.time()
            
            # Step 1: Load and process documents (enhanced)
            processing_result = self._load_and_process_documents()
            
            if self.documents:
                # Step 2: Create text embeddings with FAISS
                print(f"[{self.name}] 🔄 Creating text embeddings...")
                text_success = self.vectorization_engine.vectorize_documents(self.documents)
                
                # Step 3: Create image embeddings if multimodal enabled
                image_success = True
                if self.enable_multimodal and self.image_data:
                    print(f"[{self.name}] 🖼️ Creating image embeddings...")
                    image_success = self.vectorization_engine.vectorize_images(self.image_data)
                
                if text_success:
                    self.is_ready = True
                    processing_time = time.time() - start_time
                    self.processing_stats['last_processing_time'] = processing_time
                    
                    print(f"[{self.name}] ✅ Enhanced RAG System Ready!")
                    print(f"[{self.name}] ⏱️ Total initialization time: {processing_time:.2f}s")
                    self._print_enhanced_system_stats()
                else:
                    print(f"[{self.name}] ❌ Vectorization failed - system in fallback mode")
            else:
                print(f"[{self.name}] ⚠️ No documents loaded - system in standby mode")
                
        except Exception as e:
            print(f"[{self.name}] ❌ Enhanced system initialization failed: {str(e)}")
            self.is_ready = False
    
    def _load_and_process_documents(self) -> ProcessingResult:
        """Enhanced document loading with multimodal processing"""
        try:
            start_time = time.time()
            
            # Ensure papers folder exists
            os.makedirs(self.papers_folder, exist_ok=True)
            
            # Find supported files
            supported_extensions = ('.pdf', '.txt', '.md', '.docx')
            paper_files = [f for f in os.listdir(self.papers_folder) 
                          if f.lower().endswith(supported_extensions)]
            
            print(f"[{self.name}] 📄 Found {len(paper_files)} document(s)")
            
            if not paper_files:
                print(f"[{self.name}] ⚠️ No documents found in {self.papers_folder}")
                return ProcessingResult(
                    success=False,
                    message="No documents found in papers folder"
                )
            
            # Process each document with multimodal extraction
            processed_docs = 0
            total_chunks = 0
            total_images_extracted = 0
            total_images_stored = 0
            processing_errors = []
            
            for i, filename in enumerate(paper_files):
                filepath = os.path.join(self.papers_folder, filename)
                
                try:
                    print(f"[{self.name}] 🔄 Processing: {filename}")
                    
                    # Enhanced multimodal extraction
                    if self.enable_multimodal:
                        text_content, extracted_images = self.text_processor.extract_multimodal_content(filepath)
                    else:
                        text_content = self.text_processor.extract_text(filepath)
                        extracted_images = []
                    
                    if not text_content or len(text_content.strip()) < 200:
                        print(f"[{self.name}] ⚠️ Skipped {filename}: insufficient text content")
                        continue
                    
                    # Clean text for embeddings
                    cleaned_text = self.text_processor.clean_text_for_embeddings(text_content)
                    
                    # Create smart chunks
                    chunks = self.text_processor.create_smart_chunks(
                        cleaned_text,
                        chunk_size=self.config.chunk_size,
                        overlap=self.config.chunk_overlap
                    )
                    
                    if not chunks:
                        print(f"[{self.name}] ⚠️ Skipped {filename}: no valid chunks created")
                        continue
                    
                    # Filter quality images
                    quality_images = [img for img in extracted_images 
                                    if img.quality_score >= self.config.image_quality_threshold]
                    
                    # Create enhanced document object
                    document = {
                        'id': i,
                        'filename': filename,
                        'filepath': filepath,
                        'content': cleaned_text,
                        'chunks': chunks,
                        'metadata': {
                            'file_size': os.path.getsize(filepath),
                            'modified_time': os.path.getmtime(filepath),
                            'processed_time': self.current_time,
                            'chunk_count': len(chunks),
                            'content_length': len(cleaned_text),
                            'images_extracted': len(extracted_images),
                            'images_stored': len(quality_images),
                            'multimodal_processing': self.enable_multimodal
                        }
                    }
                    
                    self.documents.append(document)
                    self.image_data.extend(quality_images)
                    
                    # Update statistics
                    processed_docs += 1
                    total_chunks += len(chunks)
                    total_images_extracted += len(extracted_images)
                    total_images_stored += len(quality_images)
                    
                    print(f"[{self.name}] ✅ Processed: {filename}")
                    print(f"   📝 Chunks: {len(chunks)}")
                    print(f"   🖼️ Images: {len(extracted_images)} extracted, {len(quality_images)} stored")
                    
                except Exception as e:
                    error_msg = f"Failed to process {filename}: {str(e)}"
                    processing_errors.append(error_msg)
                    print(f"[{self.name}] ❌ {error_msg}")
            
            # Update processing statistics
            self.processing_stats.update({
                'documents_processed': processed_docs,
                'total_chunks': total_chunks,
                'images_extracted': total_images_extracted,
                'images_stored': total_images_stored
            })
            
            processing_time = time.time() - start_time
            
            if self.documents:
                success_message = (f"Successfully processed {processed_docs} documents with "
                                 f"{total_chunks} chunks and {total_images_stored} quality images")
                
                print(f"[{self.name}] 🎉 {success_message}")
                
                return ProcessingResult(
                    success=True,
                    message=success_message,
                    documents_processed=processed_docs,
                    chunks_created=total_chunks,
                    images_extracted=total_images_extracted,
                    images_processed=total_images_stored,
                    processing_time_seconds=processing_time,
                    errors=processing_errors
                )
            else:
                return ProcessingResult(
                    success=False,
                    message="No documents could be processed successfully",
                    errors=processing_errors,
                    processing_time_seconds=processing_time
                )
            
        except Exception as e:
            error_msg = f"Document loading failed: {str(e)}"
            print(f"[{self.name}] ❌ {error_msg}")
            return ProcessingResult(
                success=False,
                message=error_msg,
                errors=[error_msg]
            )
    
    def _print_enhanced_system_stats(self):
        """Print comprehensive system statistics"""
        stats = self.get_enhanced_stats()
        vec_info = self.vectorization_engine.get_vectorization_info()
        
        print(f"[{self.name}] 📊 Enhanced System Statistics:")
        print(f"   📚 Documents: {stats['documents']}")
        print(f"   📄 Text chunks: {stats['total_chunks']}")
        print(f"   🖼️ Images indexed: {stats.get('images_stored', 0)}")
        print(f"   🔢 Text vector dimension: {vec_info.get('vector_dimension', 0)}")
        print(f"   🎯 CLIP dimension: {vec_info.get('clip_dimension', 0)}")
        print(f"   🧠 Text model: {vec_info.get('text_model_name', 'Unknown')}")
        print(f"   🖼️ CLIP model: {vec_info.get('clip_model_name', 'None')}")
        print(f"   💾 Memory usage: {vec_info.get('memory_usage_mb', 0):.1f}MB")
        print(f"   🎮 Multimodal: {'✅ Active' if self.enable_multimodal else '❌ Disabled'}")
    
    # ===== ENHANCED PUBLIC API METHODS =====
    
    def retrieve_for_chat_agent(self, query: str, max_results: int = 3) -> List[RetrievalResult]:
        """MAIN RETRIEVAL - Enhanced semantic search with multimodal support"""
        if not self.is_ready:
            print(f"[{self.name}] ⚠️ System not ready for retrieval")
            return []
        
        return self.retrieval_engine.retrieve_for_chat_agent(query, max_results)
    
    def search_documents(self, query: str, max_results: int = 5, min_similarity: float = 0.1,
                        include_images: bool = None) -> List[Dict]:
        """Enhanced document search with multimodal support"""
        if not self.is_ready:
            return []
        
        # Auto-detect image search if not specified
        if include_images is None:
            include_images = self.enable_multimodal and any(
                keyword in query.lower() 
                for keyword in ['image', 'diagram', 'chart', 'figure', 'visual', 'show me']
            )
        
        return self.retrieval_engine.search_with_filters(
            query=query,
            max_results=max_results,
            min_similarity=min_similarity,
            include_metadata=True,
            search_images=include_images
        )
    
    def search_images_only(self, query: str, max_results: int = 4) -> List[Dict]:
        """Search for images specifically"""
        if not self.is_ready or not self.enable_multimodal:
            return []
        
        try:
            # Get image search results
            image_results = self.vectorization_engine.search_similar_images(query, max_results)
            
            formatted_results = []
            for image_data, clip_sim, context_sim, final_score in image_results:
                result = {
                    'content': f"Image from {image_data.document_name}, Page {image_data.page_number + 1}",
                    'source': image_data.document_name,
                    'page_number': image_data.page_number + 1,
                    'clip_similarity': clip_sim,
                    'context_similarity': context_sim,
                    'final_score': final_score,
                    'quality_score': image_data.quality_score,
                    'ocr_text': image_data.ocr_text,
                    'context_text': image_data.context_text[:200] + "..." if len(image_data.context_text) > 200 else image_data.context_text,
                    'image_size': f"{image_data.width}×{image_data.height}",
                    'result_type': 'image'
                }
                formatted_results.append(result)
            
            return formatted_results
            
        except Exception as e:
            print(f"[{self.name}] ❌ Image search failed: {str(e)}")
            return []
    
    def get_document_summary(self) -> str:
        """Get comprehensive document library summary with multimodal info"""
        if not self.documents:
            return self._get_empty_library_message()
        
        total_chunks = sum(len(doc['chunks']) for doc in self.documents)
        total_size_kb = sum(doc['metadata']['file_size'] for doc in self.documents) // 1024
        total_images = len(self.image_data)
        
        summary = f"📚 **Enhanced Research Library for {self.user}**\n\n"
        summary += f"📂 **Location:** {self.papers_folder}\n"
        summary += f"📄 **Documents:** {len(self.documents)}\n"
        summary += f"🔍 **Smart chunks:** {total_chunks:,}\n"
        summary += f"🖼️ **Images indexed:** {total_images}\n"
        summary += f"💾 **Total size:** {total_size_kb:,}KB\n"
        summary += f"🕐 **Last updated:** {self.current_time} UTC\n\n"
        
        # Enhanced AI status
        if self.is_ready:
            vec_info = self.vectorization_engine.get_vectorization_info()
            summary += f"🚀 **Enhanced AI Status:** ✅ MULTIMODAL ACTIVE\n"
            summary += f"🧠 **Text Model:** {vec_info.get('text_model_name', 'Unknown')}\n"
            summary += f"🖼️ **Vision Model:** {vec_info.get('clip_model_name', 'None')}\n"
            summary += f"📊 **Text Dimensions:** {vec_info.get('vector_dimension', 0):,}\n"
            summary += f"🎯 **CLIP Dimensions:** {vec_info.get('clip_dimension', 0):,}\n"
            summary += f"⚡ **Search Engine:** Hybrid FAISS (Text + Images)\n"
            summary += f"💾 **Memory Usage:** {vec_info.get('memory_usage_mb', 0):.1f}MB\n\n"
        else:
            summary += f"🚀 **Enhanced AI Status:** ❌ NOT READY\n\n"
        
        # Document list with multimodal info
        summary += "📋 **Your Documents:**\n"
        for i, doc in enumerate(self.documents, 1):
            file_size_kb = doc['metadata']['file_size'] // 1024
            images_info = ""
            if doc['metadata'].get('images_stored', 0) > 0:
                images_info = f", {doc['metadata']['images_stored']} images"
            
            summary += f"{i:2d}. **{doc['filename']}** ({len(doc['chunks'])} chunks{images_info}, {file_size_kb}KB)\n"
        
        # Enhanced usage suggestions
        summary += f"\n💡 **Try asking:**\n"
        summary += f"   • 'What is the main topic?'\n"
        summary += f"   • 'Show me diagrams about...'\n"
        summary += f"   • 'Find brain structure images'\n"
        summary += f"   • 'Explain with visual examples'\n"
        summary += f"   • 'Compare different approaches'"
        
        if self.enable_multimodal:
            summary += f"\n\n🎯 **Multimodal Capabilities:**\n"
            summary += f"   • Text and image search\n"
            summary += f"   • OCR text extraction\n"
            summary += f"   • Visual context understanding\n"
            summary += f"   • Hybrid relevance scoring"
        
        return summary
    
    def _get_empty_library_message(self) -> str:
        """Enhanced empty library message"""
        multimodal_note = "\n🖼️ **Multimodal Support:** Images and diagrams will be automatically extracted" if self.enable_multimodal else ""
        
        return (f"📚 **Empty Enhanced Library for {self.user}**\n\n"
                f"📂 **Folder:** {self.papers_folder}\n\n"
                f"🔧 **To get started:**\n"
                f"1. Add your documents (PDF, TXT, MD, DOCX) to the papers folder\n"
                f"2. Restart the system for automatic processing\n"
                f"3. Search for both text and visual content\n\n"
                f"💡 **Supported formats:** PDF, TXT, Markdown, Word documents\n"
                f"🚀 **AI-powered:** Semantic search + CLIP image understanding"
                f"{multimodal_note}")
    
    def get_enhanced_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced system statistics"""
        if not self.documents:
            return {
                'documents': 0,
                'total_chunks': 0,
                'images_stored': 0,
                'system_ready': False,
                'multimodal_enabled': self.enable_multimodal,
                'vectorization_status': 'no_documents'
            }
        
        total_chunks = sum(len(doc['chunks']) for doc in self.documents)
        total_size = sum(doc['metadata']['file_size'] for doc in self.documents)
        
        stats = {
            'documents': len(self.documents),
            'total_chunks': total_chunks,
            'images_stored': len(self.image_data),
            'total_size_kb': total_size // 1024,
            'avg_chunks_per_doc': total_chunks / len(self.documents),
            'system_ready': self.is_ready,
            'multimodal_enabled': self.enable_multimodal,
            'vectorization_status': 'ready' if self.is_ready else 'failed',
            'last_updated': self.current_time,
            'user': self.user,
            'processing_stats': self.processing_stats.copy()
        }
        
        # Add vectorization info if available
        if self.is_ready:
            vec_info = self.vectorization_engine.get_vectorization_info()
            stats.update({
                'text_embedding_model': vec_info.get('text_model_name'),
                'clip_model': vec_info.get('clip_model_name'),
                'vector_dimension': vec_info.get('vector_dimension'),
                'clip_dimension': vec_info.get('clip_dimension'),
                'memory_usage_mb': vec_info.get('memory_usage_mb'),
                'image_memory_mb': vec_info.get('image_memory_mb'),
                'faiss_text_index_size': vec_info.get('faiss_index_size'),
                'faiss_image_index_size': vec_info.get('image_index_size')
            })
        
        return stats
    
    # ===== YOUR EXISTING METHODS (MAINTAINED FOR COMPATIBILITY) =====
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic system statistics (compatibility method)"""
        enhanced_stats = self.get_enhanced_stats()
        
        # Return in your original format
        return {
            'documents': enhanced_stats['documents'],
            'total_chunks': enhanced_stats['total_chunks'],
            'system_ready': enhanced_stats['system_ready'],
            'vectorization_status': enhanced_stats['vectorization_status'],
            'last_updated': enhanced_stats['last_updated'],
            'user': enhanced_stats['user']
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get detailed system information (enhanced)"""
        base_info = {
            'agent_name': self.name,
            'user': self.user,
            'papers_folder': self.papers_folder,
            'current_time': self.current_time,
            'system_ready': self.is_ready,
            'documents_loaded': len(self.documents),
            'multimodal_enabled': self.enable_multimodal,
            'images_indexed': len(self.image_data)
        }
        
        if self.is_ready:
            # Add vectorization details
            vec_info = self.vectorization_engine.get_vectorization_info()
            base_info.update(vec_info)
            
            # Add retrieval stats
            retrieval_stats = self.retrieval_engine.get_retrieval_stats()
            base_info.update({'retrieval_' + k: v for k, v in retrieval_stats.items()})
        
        return base_info
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (enhanced)"""
        vec_info = self.vectorization_engine.get_vectorization_info()
        
        return {
            'rag_agent': 'ModernEnhancedRAGAgent',
            'text_processor': 'EnhancedTextProcessor',
            'vectorization': 'FAISS + SentenceTransformers + CLIP',
            'text_embedding_model': vec_info.get('text_model_name', 'all-MiniLM-L6-v2'),
            'clip_model': vec_info.get('clip_model_name', 'None'),
            'vector_dimension': vec_info.get('vector_dimension', 384),
            'clip_dimension': vec_info.get('clip_dimension', 0),
            'search_engine': 'Hybrid FAISS (Text + Images)',
            'multimodal_enabled': self.enable_multimodal,
            'is_ready': self.is_ready,
            'total_documents': len(self.documents),
            'total_chunks': sum(len(doc['chunks']) for doc in self.documents),
            'total_images': len(self.image_data),
            'status': 'multimodal_active' if (self.is_ready and self.enable_multimodal) else 'text_only' if self.is_ready else 'standby'
        }
    
    # ===== PERSISTENCE AND MANAGEMENT =====
    
    def reload_documents(self) -> bool:
        """Reload documents and rebuild indices"""
        print(f"[{self.name}] 🔄 Reloading documents with enhanced processing...")
        
        # Clear current data
        self.documents = []
        self.image_data = []
        self.is_ready = False
        
        # Reinitialize enhanced system
        self._initialize_enhanced_system()
        
        return self.is_ready
    
    def save_index(self, path: Optional[str] = None) -> bool:
        """Save enhanced FAISS indices and metadata"""
        if not self.is_ready:
            print(f"[{self.name}] ⚠️ Cannot save: system not ready")
            return False
        
        save_path = path or self.papers_folder
        return self.vectorization_engine.save_index(save_path)
    
    def load_index(self, path: Optional[str] = None) -> bool:
        """Load saved enhanced indices and metadata"""
        load_path = path or self.papers_folder
        success = self.vectorization_engine.load_index(load_path)
        
        if success:
            self.is_ready = True
            # Update image data from loaded indices
            self.image_data = self.vectorization_engine.image_data
            print(f"[{self.name}] ✅ Enhanced indices loaded successfully")
            print(f"[{self.name}] 📝 Text chunks: {len(self.vectorization_engine.vectorized_chunks)}")
            print(f"[{self.name}] 🖼️ Images: {len(self.image_data)}")
        else:
            print(f"[{self.name}] ❌ Failed to load enhanced indices")
        
        return success
    
    # ===== ENHANCED LLM FORMATTING =====
    
    def format_for_llm(self, query: str, retrieval_results: List[RetrievalResult]) -> str:
        """Enhanced LLM formatting with multimodal context"""
        
        if not retrieval_results:
            return f"I don't have any information about '{query}'. Please ask another question."
        
        # Create enhanced prompt with multimodal awareness
        prompt = "You are a helpful AI assistant with access to both text and visual information. Answer based ONLY on the information provided below.\n\n"
        
        # Special handling for domain-specific content
        if any(term in query.lower() for term in ['parkinson', 'brain', 'neurological']):
            prompt += "IMPORTANT: You have access to medical information including text and images. Provide accurate, helpful information based on the retrieved content.\n\n"
        
        # Separate text and image results
        text_results = [r for r in retrieval_results if r.result_type == "text"]
        image_results = [r for r in retrieval_results if r.result_type == "image"]
        
        prompt += "--- START OF RETRIEVED INFORMATION ---\n\n"
        
        # Add text passages
        if text_results:
            prompt += "TEXT INFORMATION:\n\n"
            for i, result in enumerate(text_results, 1):
                prompt += f"PASSAGE {i}:\n{result.content}\n\n"
        
        # Add image information
        if image_results:
            prompt += "VISUAL INFORMATION:\n\n"
            for i, result in enumerate(image_results, 1):
                prompt += f"IMAGE {i}:\n"
                prompt += f"Source: {result.source}\n"
                if result.image_data and result.image_data.ocr_text:
                    prompt += f"Text in image: {result.image_data.ocr_text}\n"
                if result.image_data and result.image_data.context_text:
                    context = result.image_data.context_text[:200] + "..." if len(result.image_data.context_text) > 200 else result.image_data.context_text
                    prompt += f"Image context: {context}\n"
                prompt += "\n"
        
        prompt += "--- END OF RETRIEVED INFORMATION ---\n\n"
        
        # Enhanced response instructions
        prompt += f"QUERY: {query}\n"
        prompt += "RESPONSE INSTRUCTIONS:\n"
        prompt += "1. Answer ONLY based on the information provided above\n"
        prompt += "2. Use both text and visual information when relevant\n"
        prompt += "3. If referencing images, mention what visual content supports your answer\n"
        prompt += "4. If the information doesn't contain a complete answer, say 'Based on the available information...'\n"
        prompt += "5. Do NOT make up any information that's not in the provided content\n"
        prompt += "6. Be comprehensive but concise in your response\n"
        prompt += "7. Use a professional, helpful tone\n\n"
        prompt += "Your response:"
        
        print(f"[{self.name}] 📝 Created enhanced multimodal prompt for LLM")
        print(f"[{self.name}] 📊 Content: {len(text_results)} text + {len(image_results)} image results")
        
        return prompt

# ===== BACKWARD COMPATIBILITY =====

# Ensure your existing code works unchanged
EnhancedRAGAgent = ModernEnhancedRAGAgent