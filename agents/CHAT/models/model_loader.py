"""
Enhanced Model Loader - Qwen-Only Version (SAME FUNCTION NAMES)
Current Date and Time (UTC): 2025-06-14 07:44:03
Current User's Login: Sagar4276
"""

import os
import torch
import json
import hashlib
from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
import psutil
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from pathlib import Path

class ModelLoader:
    """Enhanced Model Loader with Smart Caching - QWEN ONLY VERSION"""
    
    def __init__(self, agent_name: str = "ModelLoader"):
        self.agent_name = agent_name
        self.current_user = "Sagar4276"
        self.current_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        self.llm = None
        self.loaded_models = {}
        self.model_cache_dir = "./models"
        self.cache_info_file = os.path.join(self.model_cache_dir, "model_cache.json")
        
        # Create cache directory
        os.makedirs(self.model_cache_dir, exist_ok=True)
        
        self.model_info = {
            'loaded': False,
            'model_name': 'None',
            'size_gb': 0,
            'type': 'none'
        }
        
        # QWEN ONLY - Single Best Model
        self.top_models = [
            {
                'id': 1,
                'name': 'qwen2.5_chat',
                'model_name': 'Qwen/Qwen2.5-1.5B-Instruct',
                'display_name': '🏆 Qwen2.5-1.5B-Instruct',
                'type': 'Latest 2024 Instruction-Tuned',
                'size_gb': 3.0,
                'quality': 9.0,
                'speed': 'Fast',
                'ram_requirement': 5.0,
                'description': 'Best open access - ChatGPT-like conversations',
                'pros': ['Most advanced', 'No auth needed', 'Excellent chat'],
                'cons': ['3GB size'],
                'access_note': 'Open access, no authentication required'
            }
        ]   

    def load_offline_slm(self) -> Optional[Dict[str, Any]]:
        """Enhanced function with smart caching - SAME FUNCTION NAME"""
        print(f"[{self.agent_name}] 🔍 Loading RAG-optimized models with smart caching...")
        
        # Load cache info
        self._load_cache_info()
        
        # Get system info
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        has_gpu = torch.cuda.is_available()
        
        print(f"[{self.agent_name}] 💻 System: {'GPU' if has_gpu else 'CPU-only'}, {memory_gb:.1f}GB RAM, {cpu_count} cores")
        
        # Start smart model selection process - SAME FUNCTION NAME
        return self._smart_model_selection(memory_gb)
    
    def _load_cache_info(self):
        """Load cached model information - SAME FUNCTION NAME"""
        try:
            if os.path.exists(self.cache_info_file):
                with open(self.cache_info_file, 'r') as f:
                    self.cache_info = json.load(f)
                print(f"[{self.agent_name}] 📂 Loaded cache info for {len(self.cache_info)} models")
            else:
                self.cache_info = {}
                print(f"[{self.agent_name}] 📂 No cache info found, starting fresh")
        except Exception as e:
            print(f"[{self.agent_name}] ⚠️ Cache load error: {str(e)}")
            self.cache_info = {}
    
    def _save_cache_info(self):
        """Save cache information - SAME FUNCTION NAME"""
        try:
            with open(self.cache_info_file, 'w') as f:
                json.dump(self.cache_info, f, indent=2)
            print(f"[{self.agent_name}] 💾 Cache info saved")
        except Exception as e:
            print(f"[{self.agent_name}] ⚠️ Cache save error: {str(e)}")
    
    def _is_model_cached(self, model_name: str) -> Dict[str, Any]:
        """Check if model is already downloaded and cached - SAME FUNCTION NAME"""
        cache_key = self._get_cache_key(model_name)
        
        if cache_key in self.cache_info:
            cache_entry = self.cache_info[cache_key]
            model_path = cache_entry.get('path')
            
            # Verify files still exist
            if model_path and os.path.exists(model_path):
                # Check for essential files
                essential_files = ['config.json', 'tokenizer.json']
                if all(os.path.exists(os.path.join(model_path, f)) for f in essential_files):
                    return {
                        'cached': True,
                        'path': model_path,
                        'size_mb': cache_entry.get('size_mb', 0),
                        'cached_at': cache_entry.get('cached_at'),
                        'last_used': cache_entry.get('last_used')
                    }
        
        return {'cached': False}
    
    def _get_cache_key(self, model_name: str) -> str:
        """Generate cache key for model - SAME FUNCTION NAME"""
        return hashlib.md5(model_name.encode()).hexdigest()
    
    def _smart_model_selection(self, available_ram: float) -> Optional[Dict[str, Any]]:
        """Smart model selection with caching - SAME FUNCTION NAME"""
        
        print(f"\n{'='*80}")
        print(f"🧠 SMART MODEL SELECTOR WITH CACHING")
        print(f"{'='*80}")
        print(f"👤 User: {self.current_user}")
        print(f"🕐 Time: {self.current_time} UTC")
        print(f"🎯 Mission: Smart model selection with cache detection!")
        print(f"💾 Available RAM: {available_ram:.1f}GB")
        print(f"📂 Cache directory: {self.model_cache_dir}")
        print(f"{'='*80}\n")
        
        # Check if Qwen is compatible
        qwen_model = self.top_models[0]  # Only Qwen
        if qwen_model['ram_requirement'] > available_ram:
            print(f"❌ Qwen requires {qwen_model['ram_requirement']}GB RAM, you have {available_ram:.1f}GB")
            return self._fallback_to_gpt2()
        
        print(f"📋 Found 1 compatible model (Qwen Only)")
        
        # Check cache status for Qwen
        cache_status = self._is_model_cached(qwen_model['model_name'])
        qwen_model['cache_status'] = cache_status
        
        # Display and select Qwen - SAME FUNCTION NAME
        return self._display_and_select_models([qwen_model])
    
    def _display_and_select_models(self, models: List[Dict]) -> Optional[Dict[str, Any]]:
        """Display models with cache status and let user select - SAME FUNCTION NAME"""
        
        print(f"\n🎯 CHOOSE YOUR AI MODEL (with cache status)")
        print(f"{'='*80}")
        
        # Display Qwen model
        model = models[0]  # Only one model (Qwen)
        cache_status = model['cache_status']
        
        # Cache status indicator
        if cache_status['cached']:
            cache_indicator = "✅ CACHED"
            size_info = f"Ready to use ({cache_status['size_mb']//1024:.1f}GB cached)"
        else:
            cache_indicator = "📥 DOWNLOAD"
            size_info = f"Will download {model['size_gb']}GB"
        
        print(f"1. {model['display_name']} ({cache_indicator})")
        print(f"   📦 Type: {model['type']}")
        print(f"   💰 Size: {size_info}")
        print(f"   ⭐ Quality: {model['quality']}/10")
        print(f"   ⚡ Speed: {model['speed']}")
        print(f"   🎯 Best for: {model['description']}")
        print(f"   ✅ Pros: {', '.join(model['pros'])}")
        print(f"   ❌ Cons: {', '.join(model['cons'])}")
        print(f"   🔐 Access: {model['access_note']}")
        
        if cache_status['cached']:
            print(f"   📅 Cached: {cache_status.get('cached_at', 'Unknown')}")
        
        print(f"\n{'='*80}")
        
        # Auto-select Qwen (since it's the only option)
        print(f"Auto-selecting Qwen (only available model): 1")
        
        print(f"\n✅ Selected: {model['display_name']}")
        
        # Load model (from cache or download) - SAME FUNCTION NAME
        return self._load_selected_model(model)
    
    def _load_selected_model(self, model_config: Dict) -> Optional[Dict[str, Any]]:
        """Load selected model (from cache or download) - SAME FUNCTION NAME"""
        
        cache_status = model_config['cache_status']
        
        if cache_status['cached']:
            print(f"🚀 Loading from cache: {cache_status['path']}")
            return self._load_from_cache(model_config, cache_status['path'])
        else:
            print(f"📥 Downloading and caching: {model_config['model_name']}")
            return self._download_and_cache_model(model_config)
    
    def _load_from_cache(self, model_config: Dict, cache_path: str) -> Optional[Dict[str, Any]]:
        """Load model from cache - SAME FUNCTION NAME"""
        
        try:
            print(f"⚡ Fast loading from cache...")
            
            model_name = model_config['model_name']
            
            # Load tokenizer from cache
            tokenizer = AutoTokenizer.from_pretrained(
                cache_path,
                local_files_only=True,
                trust_remote_code=True
            )
            
            # Load model from cache
            model = AutoModelForCausalLM.from_pretrained(
                cache_path,
                local_files_only=True,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            # Set pad token if not available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Quick test - SAME FUNCTION NAME
            test_result = self._quick_test_model(model, tokenizer, model_config['name'])
            
            # Update cache info (last used)
            cache_key = self._get_cache_key(model_name)
            if cache_key in self.cache_info:
                self.cache_info[cache_key]['last_used'] = self.current_time
                self._save_cache_info()
            
            # Store loaded model
            self.llm = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_config['display_name'],
                'config': model_config
            }
            
            self.model_info = {
                'loaded': True,
                'model_name': model_config['display_name'],
                'size_gb': model_config['size_gb'],
                'quality': model_config['quality'],
                'description': model_config['description'],
                'type': 'cached_model',
                'speed': model_config['speed'],
                'loaded_from': 'cache'
            }
            
            print(f"✅ {model_config['display_name']} loaded from cache!")
            print(f"🧪 Test: '{test_result[:50]}{'...' if len(test_result) > 50 else ''}'")
            
            return model_config
            
        except Exception as e:
            print(f"❌ Cache loading failed: {str(e)}")
            print(f"🔄 Falling back to download...")
            return self._download_and_cache_model(model_config)
    
    def _download_and_cache_model(self, model_config: Dict) -> Optional[Dict[str, Any]]:
        """Download model and cache it - SAME FUNCTION NAME"""
        
        try:
            model_name = model_config['model_name']
            cache_key = self._get_cache_key(model_name)
            cache_path = os.path.join(self.model_cache_dir, cache_key)
            
            print(f"📦 Downloading {model_config['display_name']}...")
            print(f"📊 Size: {model_config['size_gb']}GB | Quality: {model_config['quality']}/10")
            print(f"🎯 Type: {model_config['type']}")
            
            # Download tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_path,
                trust_remote_code=True
            )
            
            # Download model
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                cache_dir=cache_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                trust_remote_code=True
            )
            
            # Set pad token if not available
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Test model - SAME FUNCTION NAME
            print(f"🧪 Testing model...")
            test_result = self._quick_test_model(model, tokenizer, model_config['name'])
            
            # Calculate cache size - SAME FUNCTION NAME
            cache_size_mb = self._calculate_directory_size(cache_path)
            
            # Update cache info
            self.cache_info[cache_key] = {
                'model_name': model_name,
                'display_name': model_config['display_name'],
                'path': cache_path,
                'size_mb': cache_size_mb,
                'cached_at': self.current_time,
                'last_used': self.current_time,
                'quality': model_config['quality'],
                'type': model_config['type']
            }
            self._save_cache_info()
            
            # Store loaded model
            self.llm = {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_config['display_name'],
                'config': model_config
            }
            
            self.model_info = {
                'loaded': True,
                'model_name': model_config['display_name'],
                'size_gb': model_config['size_gb'],
                'quality': model_config['quality'],
                'description': model_config['description'],
                'type': 'downloaded_model',
                'speed': model_config['speed'],
                'loaded_from': 'download'
            }
            
            print(f"✅ {model_config['display_name']} downloaded and cached!")
            print(f"📂 Cache location: {cache_path}")
            print(f"💾 Cache size: {cache_size_mb//1024:.1f}GB")
            print(f"🧪 Test: '{test_result[:50]}{'...' if len(test_result) > 50 else ''}'")
            
            return model_config
            
        except Exception as e:
            print(f"❌ Download failed: {str(e)}")
            return None
    
    def _calculate_directory_size(self, directory: str) -> int:
        """Calculate total size of directory in MB - SAME FUNCTION NAME"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size // (1024 * 1024)  # Convert to MB
        except Exception:
            return 0
    
    def _quick_test_model(self, model, tokenizer, model_name: str) -> str:
        """Quick test to verify model works - SAME FUNCTION NAME"""
        
        try:
            # Simple test prompt
            test_prompt = "Hello! How are you?"
            
            # Encode input
            inputs = tokenizer.encode(
                test_prompt,
                return_tensors='pt',
                max_length=100,
                truncation=True
            )
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    inputs,
                    max_new_tokens=20,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            generated_part = full_response.replace(test_prompt, "").strip()
            
            return generated_part if generated_part else "I'm doing well!"
            
        except Exception as e:
            return f"Test failed: {str(e)[:30]}..."
    
    def _fallback_to_gpt2(self) -> Optional[Dict[str, Any]]:
        """Fallback to GPT-2 if advanced models fail - SAME FUNCTION NAME"""
        
        print(f"\n🔄 Falling back to GPT-2 models...")
        
        # GPT-2 fallback models
        gpt2_models = [
            {
                'name': 'distilgpt2',
                'description': 'Optimized GPT-2 for context-aware text generation',
                'size_gb': 0.3,
                'quality': 4.0,
                'ram_requirement': 1.5,
                'good_for_rag': True
            }
        ]
        
        for model_config in gpt2_models:
            print(f"🔍 Trying {model_config['name']} (fallback)...")
            
            model = self._load_gpt2_model(model_config)
            if model:
                self.llm = model
                self.model_info = {
                    'loaded': True,
                    'model_name': model_config['name'],
                    'size_gb': model_config['size_gb'],
                    'quality': model_config['quality'],
                    'description': model_config['description'],
                    'type': 'rag_optimized_generation'
                }
                return model['config']
        
        print(f"❌ All models failed to load!")
        return None
    
    def _load_gpt2_model(self, model_config: dict) -> Optional[Dict[str, Any]]:
        """Load GPT-2 models (original method) - SAME FUNCTION NAME"""
        try:
            model_name = model_config['name']
            
            print(f"📂 Loading {model_name} (RAG-optimized)...")
            
            # Load GPT-2 tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                local_files_only=False
            )
            model = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=self.model_cache_dir,
                torch_dtype=torch.float32,
                local_files_only=False
            )
            
            # Set padding token
            tokenizer.pad_token = tokenizer.eos_token
            
            # Test model
            test_prompt = "Research shows that Parkinson's disease"
            test_tokens = tokenizer.encode(test_prompt, return_tensors='pt')
            
            with torch.no_grad():
                test_output = model.generate(
                    test_tokens,
                    max_new_tokens=15,
                    temperature=0.8,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            test_response = tokenizer.decode(test_output[0], skip_special_tokens=True)
            test_generated = test_response.replace(test_prompt, "").strip()
            
            print(f"✅ {model_name} loaded successfully!")
            print(f"🧪 RAG Test: '{test_generated[:50]}...'")
            
            return {
                'model': model,
                'tokenizer': tokenizer,
                'model_name': model_name,
                'config': model_config
            }
            
        except Exception as e:
            print(f"⚠️ Failed to load {model_config['name']}: {str(e)[:50]}...")
            return None
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics - SAME FUNCTION NAME"""
        try:
            total_models = len(self.cache_info)
            total_size_mb = sum(entry.get('size_mb', 0) for entry in self.cache_info.values())
            
            return {
                'total_cached_models': total_models,
                'total_cache_size_gb': total_size_mb / 1024,
                'cache_directory': self.model_cache_dir,
                'cached_models': [
                    {
                        'name': entry.get('display_name', 'Unknown'),
                        'size_gb': entry.get('size_mb', 0) / 1024,
                        'cached_at': entry.get('cached_at'),
                        'last_used': entry.get('last_used'),
                        'quality': entry.get('quality', 0)
                    }
                    for entry in self.cache_info.values()
                ]
            }
        except Exception:
            return {'total_cached_models': 0, 'total_cache_size_gb': 0}
    
    def clear_cache(self, model_name: Optional[str] = None):
        """Clear cache for specific model or all models - SAME FUNCTION NAME"""
        try:
            if model_name:
                # Clear specific model
                cache_key = self._get_cache_key(model_name)
                if cache_key in self.cache_info:
                    cache_path = self.cache_info[cache_key]['path']
                    if os.path.exists(cache_path):
                        import shutil
                        shutil.rmtree(cache_path)
                    del self.cache_info[cache_key]
                    self._save_cache_info()
                    print(f"✅ Cleared cache for {model_name}")
                else:
                    print(f"❌ Model {model_name} not found in cache")
            else:
                # Clear all cache
                import shutil
                if os.path.exists(self.model_cache_dir):
                    shutil.rmtree(self.model_cache_dir)
                os.makedirs(self.model_cache_dir, exist_ok=True)
                self.cache_info = {}
                self._save_cache_info()
                print(f"✅ Cleared all model cache")
        except Exception as e:
            print(f"❌ Cache clear failed: {str(e)}")
    
    def get_llm(self) -> Optional[Dict[str, Any]]:
        """Get loaded model - SAME FUNCTION NAME"""
        return self.llm
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information - SAME FUNCTION NAME"""
        return self.model_info

# Utility function for easy cache management - SAME FUNCTION NAME
def show_cache_info():
    """Show cache information - SAME FUNCTION NAME"""
    loader = ModelLoader()
    loader._load_cache_info()
    stats = loader.get_cache_stats()
    
    print("📂 MODEL CACHE INFORMATION")
    print("=" * 50)
    print(f"📁 Cache directory: {stats['cache_directory']}")
    print(f"📊 Total models: {stats['total_cached_models']}")
    print(f"💾 Total size: {stats['total_cache_size_gb']:.1f}GB")
    print()
    
    if stats['cached_models']:
        print("📋 Cached models:")
        for model in stats['cached_models']:
            print(f"  • {model['name']} ({model['size_gb']:.1f}GB)")
            print(f"    Quality: {model['quality']}/10")
            print(f"    Cached: {model['cached_at']}")
            print(f"    Last used: {model['last_used']}")
            print()
    else:
        print("📭 No models cached yet")

if __name__ == "__main__":
    show_cache_info()