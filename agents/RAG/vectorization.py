"""
Enhanced Vectorization Engine - CLIP + FAISS + Sentence Transformers
Current Date and Time (UTC): 2025-06-13 09:21:09
Current User's Login: Sagar4276
"""

import numpy as np
import faiss
import pickle
import os
from typing import Dict, List, Tuple, Optional
from sentence_transformers import SentenceTransformer

# Import your existing + new structures
from .data_structures import VectorizedChunk, MultimodalImageData, MultimodalConfig

class ModernVectorizationEngine:
    """Enhanced FAISS-based vectorization with CLIP support"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 cache_dir: str = "knowledge_base", enable_multimodal: bool = True):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.enable_multimodal = enable_multimodal
        self.vector_dimension = 384  # all-MiniLM-L6-v2 dimension
        
        # Core components (your existing)
        self.embedding_model = None
        self.faiss_index = None
        self.vectorized_chunks = []
        self.embeddings = None
        
        # NEW: Multimodal components (Safwan's approach)
        self.clip_model = None
        self.clip_processor = None
        self.image_index = None  # Separate FAISS index for images
        self.image_data = []    # Store image metadata
        self.image_embeddings = None
        self.config = MultimodalConfig()
        
        print(f"[ModernVectorization] 🚀 Initializing for user: Sagar4276")
        print(f"[ModernVectorization] 🎯 Multimodal enabled: {enable_multimodal}")
        
        # Initialize models
        self._load_embedding_model()
        if self.enable_multimodal:
            self._load_clip_model()
        
    def _load_embedding_model(self):
        """Load sentence transformer model (your existing)"""
        try:
            print(f"[ModernVectorization] 🔄 Loading embedding model: {self.model_name}")
            self.embedding_model = SentenceTransformer(self.model_name)
            print(f"[ModernVectorization] ✅ Text embedding model loaded!")
            print(f"[ModernVectorization] 📊 Vector dimension: {self.vector_dimension}")
        except Exception as e:
            print(f"[ModernVectorization] ❌ Failed to load text model: {str(e)}")
            raise e
    
    def _load_clip_model(self):
        """Load CLIP model for image understanding (Safwan's approach)"""
        try:
            print(f"[ModernVectorization] 🔄 Loading CLIP model: {self.config.clip_model}")
            
            # Import CLIP
            import torch
            
            from transformers import CLIPModel, CLIPProcessor

            device = "cuda" if torch.cuda.is_available() else "cpu"

            self.clip_model = CLIPModel.from_pretrained(self.config.clip_model).to(device)
            self.clip_processor = CLIPProcessor.from_pretrained(self.config.clip_model)

            
            self.clip_dimension = 512  # CLIP ViT-B/32 dimension
            
            print(f"[ModernVectorization] ✅ CLIP model loaded!")
            print(f"[ModernVectorization] 🖼️ CLIP dimension: {self.clip_dimension}")
            print(f"[ModernVectorization] 🔧 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
            
        except ImportError:
            print(f"[ModernVectorization] ❌ CLIP not available. Install: pip install clip-by-openai")
            self.enable_multimodal = False
        except Exception as e:
            print(f"[ModernVectorization] ❌ CLIP loading failed: {str(e)}")
            self.enable_multimodal = False
    
    # ===== YOUR EXISTING TEXT VECTORIZATION (ENHANCED) =====
    
    def vectorize_documents(self, documents: List[Dict]) -> bool:
        """Vectorize all documents using modern embeddings (your method enhanced)"""
        try:
            print(f"[ModernVectorization] 🔄 Vectorizing documents with modern embeddings...")
            
            # Prepare chunks and metadata
            all_chunks = []
            chunk_metadata = []
            
            chunk_id = 0
            for doc in documents:
                doc_id = doc['id']
                filename = doc['filename']
                
                for chunk_index, chunk_content in enumerate(doc['chunks']):
                    all_chunks.append(chunk_content)
                    
                    metadata = {
                        'chunk_id': chunk_id,
                        'doc_id': doc_id,
                        'chunk_index': chunk_index,
                        'filename': filename,
                        'content_length': len(chunk_content),
                        'doc_metadata': doc.get('metadata', {})
                    }
                    chunk_metadata.append(metadata)
                    chunk_id += 1
            
            if not all_chunks:
                print(f"[ModernVectorization] ❌ No chunks to vectorize")
                return False
            
            print(f"[ModernVectorization] 🔄 Creating embeddings for {len(all_chunks)} chunks...")
            
            # Create embeddings using sentence transformer
            self.embeddings = self.embedding_model.encode(
                all_chunks,
                batch_size=32,
                show_progress_bar=True,
                convert_to_numpy=True
            )
            
            # Handle invalid values
            invalid_count = np.sum(~np.isfinite(self.embeddings))
            if invalid_count > 0:
                print(f"[ModernVectorization] ⚠️ Warning: {invalid_count} invalid values in embeddings")
                self.embeddings = np.nan_to_num(self.embeddings, nan=0.0, posinf=0.0, neginf=0.0)
            
            print(f"[ModernVectorization] 🔢 Creating text FAISS index...")
            
            # Convert to float32 and normalize
            embeddings_float32 = self.embeddings.astype('float32').copy()
            
            try:
                faiss.normalize_L2(embeddings_float32)
            except Exception as e:
                print(f"[ModernVectorization] ⚠️ Normalization warning: {str(e)}")
                embeddings_float32 = np.nan_to_num(embeddings_float32, nan=0.0, posinf=0.0, neginf=0.0)
                faiss.normalize_L2(embeddings_float32)
            
            # Create FAISS index for text
            self.faiss_index = faiss.IndexFlatIP(self.vector_dimension)
            self.faiss_index.add(embeddings_float32)
            
            # Create vectorized chunks objects
            self.vectorized_chunks = []
            for i, (chunk_content, metadata) in enumerate(zip(all_chunks, chunk_metadata)):
                vectorized_chunk = VectorizedChunk(
                    chunk_id=metadata['chunk_id'],
                    doc_id=metadata['doc_id'],
                    chunk_index=metadata['chunk_index'],
                    content=chunk_content,
                    vector=self.embeddings[i],
                    metadata=metadata
                )
                self.vectorized_chunks.append(vectorized_chunk)
            
            print(f"[ModernVectorization] ✅ Text vectorization complete!")
            print(f"[ModernVectorization] 📊 FAISS text index: {self.faiss_index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            print(f"[ModernVectorization] ❌ Text vectorization failed: {str(e)}")
            return False
    
    # ===== NEW: IMAGE VECTORIZATION (SAFWAN'S APPROACH) =====
    
    def vectorize_images(self, image_list: List[MultimodalImageData]) -> bool:
        """Vectorize images using CLIP (Safwan's approach)"""
        if not self.enable_multimodal or not self.clip_model:
            print(f"[ModernVectorization] ⚠️ CLIP not available for image vectorization")
            return False
        
        if not image_list:
            print(f"[ModernVectorization] ⚠️ No images to vectorize")
            return True  # Not an error
        
        try:
            print(f"[ModernVectorization] 🖼️ Vectorizing {len(image_list)} images with CLIP...")
            
            import torch
            from PIL import Image
            
            # Filter high-quality images (Safwan's approach)
            quality_images = [img for img in image_list 
                            if img.quality_score >= self.config.image_quality_threshold]
            
            if not quality_images:
                print(f"[ModernVectorization] ⚠️ No quality images to vectorize")
                return True
            
            print(f"[ModernVectorization] 🎯 Processing {len(quality_images)} quality images...")
            
            # Create image embeddings
            image_embeddings_list = []
            processed_images = []
            
            for i, img_data in enumerate(quality_images):
                try:
                    # Create dummy PIL image (in real implementation, load actual image)
                    # For demo, we'll create embeddings from the OCR text + context
                    combined_text = f"{img_data.ocr_text} {img_data.context_text}".strip()
                    
                    if combined_text:
                        # Use CLIP text encoder for image context
                        text_tokens = self.clip_model.encode_text(
                            torch.cat([self.clip_model.tokenize(combined_text[:77])])  # CLIP max length
                        )
                        
                        # Convert to numpy and normalize
                        embedding = text_tokens.detach().cpu().numpy().astype('float32')
                        embedding = embedding / np.linalg.norm(embedding)  # Normalize
                        
                        image_embeddings_list.append(embedding.flatten())
                        processed_images.append(img_data)
                        
                        print(f"[ModernVectorization] ✅ Image {i+1}: {img_data.id} vectorized")
                    
                except Exception as e:
                    print(f"[ModernVectorization] ⚠️ Image {i+1} vectorization failed: {str(e)}")
                    continue
            
            if not image_embeddings_list:
                print(f"[ModernVectorization] ❌ No images successfully vectorized")
                return False
            
            # Create image embeddings array
            self.image_embeddings = np.vstack(image_embeddings_list)
            self.image_data = processed_images
            
            # Create separate FAISS index for images
            print(f"[ModernVectorization] 🔢 Creating image FAISS index...")
            self.image_index = faiss.IndexFlatIP(self.clip_dimension)
            self.image_index.add(self.image_embeddings)
            
            print(f"[ModernVectorization] ✅ Image vectorization complete!")
            print(f"[ModernVectorization] 🖼️ FAISS image index: {self.image_index.ntotal} vectors")
            
            return True
            
        except Exception as e:
            print(f"[ModernVectorization] ❌ Image vectorization failed: {str(e)}")
            return False
    
    def search_similar_images(self, query: str, top_k: int = 4) -> List[Tuple[MultimodalImageData, float, float, float]]:
        """Search for similar images using CLIP (Safwan's hybrid approach)"""
        if not self.enable_multimodal or not self.image_index or not self.image_data:
            return []
        
        try:
            import torch
            
            # Create query embedding using CLIP text encoder
            text_tokens = self.clip_model.encode_text(
                torch.cat([self.clip_model.tokenize(query[:77])])
            )
            
            query_embedding = text_tokens.detach().cpu().numpy().astype('float32')
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search using FAISS
            search_k = min(top_k * 2, len(self.image_data))
            similarities, indices = self.image_index.search(query_embedding, search_k)
            
            # Apply Safwan's hybrid scoring
            results = []
            for similarity, index in zip(similarities[0], indices[0]):
                if index < 0 or index >= len(self.image_data):
                    continue
                
                image_data = self.image_data[index]
                
                # Safwan's hybrid scoring components
                clip_similarity = float(similarity)
                
                # Context similarity (simplified)
                context_similarity = self._calculate_context_similarity(query, image_data.context_text)
                
                # Page proximity score (simplified - based on how central the image is)
                proximity_score = 0.5  # Placeholder
                
                # Apply Safwan's hybrid formula
                hybrid_score = (clip_similarity * self.config.hybrid_weights['clip_visual'] +
                              context_similarity * self.config.hybrid_weights['context'] +
                              proximity_score * self.config.hybrid_weights['proximity'])
                
                # Multiply by quality score (Safwan's approach)
                final_score = hybrid_score * image_data.quality_score
                
                # Apply thresholds
                if (clip_similarity >= self.config.clip_similarity_threshold and
                    final_score >= self.config.final_score_threshold):
                    
                    results.append((image_data, clip_similarity, context_similarity, final_score))
            
            # Sort by final score
            results.sort(key=lambda x: x[3], reverse=True)
            
            print(f"[ModernVectorization] 🔍 Found {len(results)} relevant images for: '{query}'")
            
            return results[:top_k]
            
        except Exception as e:
            print(f"[ModernVectorization] ❌ Image search failed: {str(e)}")
            return []
    
    def _calculate_context_similarity(self, query: str, context_text: str) -> float:
        """Calculate context similarity for hybrid scoring"""
        if not context_text:
            return 0.0
        
        try:
            # Simple term overlap for now (can be enhanced with embeddings)
            query_terms = set(query.lower().split())
            context_terms = set(context_text.lower().split())
            
            if not query_terms:
                return 0.0
            
            overlap = len(query_terms.intersection(context_terms))
            similarity = overlap / len(query_terms)
            
            return min(similarity, 1.0)
            
        except Exception:
            return 0.0
    
    # ===== YOUR EXISTING SEARCH (ENHANCED) =====
    
    def search_similar_chunks(self, query: str, top_k: int = 5) -> List[Tuple[VectorizedChunk, float]]:
        """Search for similar chunks using FAISS (your existing method)"""
        if not self.faiss_index or not self.vectorized_chunks:
            print(f"[ModernVectorization] ⚠️ No vectorized data available for search")
            return []
        
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
            
            # Handle potential invalid values in query embedding
            if not np.all(np.isfinite(query_embedding)):
                print(f"[ModernVectorization] ⚠️ Query embedding contains invalid values")
                query_embedding = np.nan_to_num(query_embedding, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Normalize for cosine similarity
            query_vector = query_embedding.astype('float32').copy()
            faiss.normalize_L2(query_vector)
            
            # Search using FAISS
            search_k = min(top_k * 2, len(self.vectorized_chunks))
            if search_k == 0:
                return []
            
            similarities, indices = self.faiss_index.search(query_vector, search_k)
            
            # Debug log
            print(f"[ModernVectorization] 🔍 Text search - Similarity range: {similarities.min():.3f} to {similarities.max():.3f}")
            
            # Prepare results with better filtering
            results = []
            for i, (similarity, index) in enumerate(zip(similarities[0], indices[0])):
                # Skip invalid indices
                if index < 0 or index >= len(self.vectorized_chunks):
                    continue
                    
                # Filter out poor matches
                if similarity < self.config.text_similarity_threshold or not np.isfinite(similarity):
                    continue
                
                chunk = self.vectorized_chunks[index]
                results.append((chunk, float(similarity)))
                
                if len(results) >= top_k:
                    break
            
            print(f"[ModernVectorization] ✅ Found {len(results)} similar text chunks")
            return results
            
        except Exception as e:
            print(f"[ModernVectorization] ❌ Text search failed: {str(e)}")
            return []
    
    # ===== HYBRID SEARCH (SAFWAN'S APPROACH) =====
    
    def hybrid_search(self, query: str, max_text_results: int = 5, 
                     max_image_results: int = 4) -> Dict[str, List]:
        """Perform hybrid text + image search (Safwan's approach)"""
        print(f"[ModernVectorization] 🔍 Performing hybrid search for: '{query}'")
        
        results = {
            'text_results': [],
            'image_results': [],
            'hybrid_score': 0.0
        }
        
        # Search text chunks
        text_results = self.search_similar_chunks(query, max_text_results)
        for chunk, similarity in text_results:
            results['text_results'].append({
                'chunk': chunk,
                'similarity': similarity,
                'content': chunk.content,
                'metadata': chunk.metadata
            })
        
        # Search images if enabled
        if self.enable_multimodal:
            image_results = self.search_similar_images(query, max_image_results)
            for img_data, clip_sim, context_sim, final_score in image_results:
                results['image_results'].append({
                    'image_data': img_data,
                    'clip_similarity': clip_sim,
                    'context_similarity': context_sim,
                    'final_score': final_score,
                    'metadata': {
                        'document_name': img_data.document_name,
                        'page_number': img_data.page_number,
                        'quality_score': img_data.quality_score
                    }
                })
        
        # Calculate combined hybrid score
        text_scores = [r['similarity'] for r in results['text_results']]
        image_scores = [r['final_score'] for r in results['image_results']]
        
        if text_scores or image_scores:
            avg_text_score = np.mean(text_scores) if text_scores else 0.0
            avg_image_score = np.mean(image_scores) if image_scores else 0.0
            results['hybrid_score'] = (avg_text_score + avg_image_score) / 2.0
        
        print(f"[ModernVectorization] ✅ Hybrid search complete:")
        print(f"[ModernVectorization] 📝 Text results: {len(results['text_results'])}")
        print(f"[ModernVectorization] 🖼️ Image results: {len(results['image_results'])}")
        print(f"[ModernVectorization] 🎯 Hybrid score: {results['hybrid_score']:.3f}")
        
        return results
    
    # ===== PERSISTENCE (ENHANCED) =====
    
    def save_index(self, base_path: str = "knowledge_base") -> bool:
        """Save FAISS indices and metadata (enhanced for multimodal)"""
        try:
            os.makedirs(base_path, exist_ok=True)
            
            # Save text FAISS index
            if self.faiss_index:
                faiss_path = os.path.join(base_path, "faiss_text_index.bin")
                faiss.write_index(self.faiss_index, faiss_path)
                print(f"[ModernVectorization] ✅ Text index saved: {faiss_path}")
            
            # Save image FAISS index
            if self.image_index:
                image_faiss_path = os.path.join(base_path, "faiss_image_index.bin")
                faiss.write_index(self.image_index, image_faiss_path)
                print(f"[ModernVectorization] ✅ Image index saved: {image_faiss_path}")
            
            # Save chunk metadata
            if self.vectorized_chunks:
                metadata_path = os.path.join(base_path, "chunk_metadata.pkl")
                with open(metadata_path, 'wb') as f:
                    pickle.dump([chunk.to_dict() for chunk in self.vectorized_chunks], f)
                print(f"[ModernVectorization] ✅ Chunk metadata saved: {metadata_path}")
            
            # Save image metadata
            if self.image_data:
                image_metadata_path = os.path.join(base_path, "image_metadata.pkl")
                with open(image_metadata_path, 'wb') as f:
                    pickle.dump([img.to_dict() for img in self.image_data], f)
                print(f"[ModernVectorization] ✅ Image metadata saved: {image_metadata_path}")
            
            # Save config
            config_path = os.path.join(base_path, "vectorization_config.json")
            config = {
                'text_model_name': self.model_name,
                'clip_model_name': self.config.clip_model if self.enable_multimodal else None,
                'vector_dimension': self.vector_dimension,
                'clip_dimension': getattr(self, 'clip_dimension', 0),
                'total_chunks': len(self.vectorized_chunks),
                'total_images': len(self.image_data),
                'multimodal_enabled': self.enable_multimodal,
                'index_type': 'faiss_flat_ip_multimodal'
            }
            
            import json
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"[ModernVectorization] ✅ Config saved: {config_path}")
            return True
            
        except Exception as e:
            print(f"[ModernVectorization] ❌ Failed to save indices: {str(e)}")
            return False
    
    def load_index(self, base_path: str = "knowledge_base") -> bool:
        """Load FAISS indices and metadata (enhanced for multimodal)"""
        try:
            # Check required files
            faiss_path = os.path.join(base_path, "faiss_text_index.bin")
            metadata_path = os.path.join(base_path, "chunk_metadata.pkl")
            config_path = os.path.join(base_path, "vectorization_config.json")
            
            if not all(os.path.exists(p) for p in [faiss_path, metadata_path, config_path]):
                print(f"[ModernVectorization] ⚠️ Required index files not found in {base_path}")
                return False
            
            # Load text FAISS index
            self.faiss_index = faiss.read_index(faiss_path)
            print(f"[ModernVectorization] ✅ Text index loaded: {self.faiss_index.ntotal} vectors")
            
            # Load text metadata
            with open(metadata_path, 'rb') as f:
                chunk_data = pickle.load(f)
            
            # Reconstruct vectorized chunks
            self.vectorized_chunks = []
            for data in chunk_data:
                chunk = VectorizedChunk(
                    chunk_id=data['chunk_id'],
                    doc_id=data['doc_id'],
                    chunk_index=data['chunk_index'],
                    content=data['content'],
                    vector=np.array(data['vector']),
                    metadata=data['metadata']
                )
                self.vectorized_chunks.append(chunk)
            
            print(f"[ModernVectorization] ✅ Text metadata loaded: {len(self.vectorized_chunks)} chunks")
            
            # Try to load image components if available
            image_faiss_path = os.path.join(base_path, "faiss_image_index.bin")
            image_metadata_path = os.path.join(base_path, "image_metadata.pkl")
            
            if os.path.exists(image_faiss_path) and os.path.exists(image_metadata_path) and self.enable_multimodal:
                try:
                    # Load image index
                    self.image_index = faiss.read_index(image_faiss_path)
                    
                    # Load image metadata
                    with open(image_metadata_path, 'rb') as f:
                        image_data_list = pickle.load(f)
                    
                    # Reconstruct image data
                    self.image_data = []
                    for data in image_data_list:
                        img_data = MultimodalImageData(
                            id=data['id'],
                            document_name=data['document_name'],
                            page_number=data['page_number'],
                            bbox=tuple(data['bbox']),
                            image_hash=data['image_hash'],
                            quality_score=data['quality_score'],
                            ocr_text=data['ocr_text'],
                            ocr_confidence=data['ocr_confidence'],
                            width=data['width'],
                            height=data['height'],
                            file_size=data['file_size'],
                            context_text=data['context_text'],
                            created_at=data['created_at']
                        )
                        if data.get('clip_embedding'):
                            img_data.clip_embedding = np.array(data['clip_embedding'])
                        self.image_data.append(img_data)
                    
                    print(f"[ModernVectorization] ✅ Image index loaded: {self.image_index.ntotal} vectors")
                    print(f"[ModernVectorization] ✅ Image metadata loaded: {len(self.image_data)} images")
                    
                except Exception as e:
                    print(f"[ModernVectorization] ⚠️ Image loading failed: {str(e)}")
            
            return True
            
        except Exception as e:
            print(f"[ModernVectorization] ❌ Failed to load indices: {str(e)}")
            return False
    
    def get_vectorization_info(self) -> Dict:
        """Get comprehensive vectorization status and statistics"""
        if not self.faiss_index:
            return {
                'status': 'not_vectorized',
                'reason': 'FAISS index not initialized'
            }
        
        info = {
            'status': 'vectorized',
            'text_model_name': self.model_name,
            'vector_dimension': self.vector_dimension,
            'total_chunks': len(self.vectorized_chunks),
            'faiss_index_size': self.faiss_index.ntotal,
            'memory_usage_mb': self.embeddings.nbytes / (1024 * 1024) if self.embeddings is not None else 0,
            'index_type': 'FAISS IndexFlatIP (Cosine Similarity)',
            'multimodal_enabled': self.enable_multimodal
        }
        
        # Add multimodal info if available
        if self.enable_multimodal:
            info.update({
                'clip_model_name': self.config.clip_model,
                'clip_dimension': getattr(self, 'clip_dimension', 0),
                'total_images': len(self.image_data),
                'image_index_size': self.image_index.ntotal if self.image_index else 0,
                'image_memory_mb': self.image_embeddings.nbytes / (1024 * 1024) if self.image_embeddings is not None else 0
            })
        
        return info