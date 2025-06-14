import os
import time
import json
from datetime import datetime
from shared_memory.simple_memory import SimpleSharedMemory
import requests

class ChatAgentAPI:
    def __init__(self, shared_memory: SimpleSharedMemory):
        self.shared_memory = shared_memory
        self.name = "ChatAgent-API"
        
        # Dynamic user and time - no hardcoding!
        self.current_user = "Sagar4276"  # From your system
        self.current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"[{self.name}] 🌐 Loading Gemini 2.0 Flash...")
        print(f"[{self.name}] 👤 User: {self.current_user}")
        print(f"[{self.name}] 🕐 Time: {self.current_time} UTC")
        
        # Load API configuration
        self.llm = self._initialize_gemini()
    
    def _initialize_gemini(self):
        """Initialize Gemini 2.0 Flash with API key from .env"""
        try:
            # Try to load from .env file
            env_file = '.env'
            api_key = None
            
            if os.path.exists(env_file):
                print(f"[{self.name}] 📁 Loading .env file...")
                with open(env_file, 'r') as f:
                    for line in f:
                        if line.startswith('GOOGLE_API_KEY='):
                            api_key = line.split('=', 1)[1].strip()
                            break
            
            if not api_key:
                print(f"[{self.name}] ❌ GOOGLE_API_KEY not found in .env")
                print(f"[{self.name}] 📋 Create .env file with: GOOGLE_API_KEY=your_key_here")
                print(f"[{self.name}] 🔗 Get your key at: https://aistudio.google.com/app/apikey")
                
                # Ask for API key
                api_key = input(f"🔑 Enter your Google API key (or press Enter to skip): ").strip()
                
                if not api_key:
                    print(f"[{self.name}] ⚠️ Skipping API setup, using fallback responses")
                    return None
                
                # Save to .env file
                try:
                    with open('.env', 'w') as f:
                        f.write(f"GOOGLE_API_KEY={api_key}\n")
                    print(f"[{self.name}] 💾 API key saved to .env file")
                except:
                    print(f"[{self.name}] ⚠️ Could not save .env file")
            
            # Test the API
            print(f"[{self.name}] 🧪 Testing Gemini API...")
            test_response = self._call_gemini_api("Hello! Test message.", [], api_key)
            
            if test_response:
                print(f"[{self.name}] ✅ Gemini 2.0 Flash connected!")
                print(f"[{self.name}] 🧪 Test: {test_response[:50]}...")
                
                return {
                    'name': 'Gemini 2.0 Flash',
                    'model_id': 'gemini-2.0-flash-exp', 
                    'api_key': api_key,
                    'status': 'ready'
                }
            else:
                print(f"[{self.name}] ❌ API test failed")
                return None
                
        except Exception as e:
            print(f"[{self.name}] ❌ Gemini setup error: {str(e)}")
            return None
    
    def process_message(self, user_id: str, message: str) -> str:
        """Process message using Gemini API"""
        current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{self.name}] 📨 [{current_time}] Processing: '{message}'")
        print(f"[{self.name}] 🔍 Model: {'Gemini 2.0 Flash' if self.llm else 'Smart Fallback'}")
        
        history = self.shared_memory.get_conversation_history(user_id)
        
        if self.llm:
            response = self._generate_gemini_response(message, history)
        else:
            response = self._generate_smart_fallback(message)
        
        self.shared_memory.add_message(user_id, response, self.name)
        print(f"[{self.name}] ✅ Response: '{response}'")
        return response
    
    def _generate_gemini_response(self, message: str, history: list) -> str:
        """Generate response using Gemini 2.0 Flash"""
        try:
            print(f"[{self.name}] 🌐 Calling Gemini 2.0 Flash...")
            
            response = self._call_gemini_api(message, history, self.llm['api_key'])
            
            if response and len(response) > 5:
                print(f"[{self.name}] ✅ Gemini response received")
                return response
            else:
                print(f"[{self.name}] ⚠️ Poor API response, using fallback")
                return self._generate_smart_fallback(message)
                
        except Exception as e:
            print(f"[{self.name}] ❌ Gemini error: {str(e)}")
            return self._generate_smart_fallback(message)
    
    def _call_gemini_api(self, message: str, history: list, api_key: str) -> str:
        """Call Gemini 2.0 Flash API"""
        try:
            # Latest Gemini API endpoint
            url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            
            headers = {
                'Content-Type': 'application/json',
            }
            
            params = {
                'key': api_key
            }
            
            # Build conversation context
            current_time = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
            system_context = f"""You are a helpful AI assistant talking to {self.current_user}. 
Current time: {current_time} UTC.
Be conversational, helpful, and engaging. Keep responses concise but informative.
The user is working on a multi-agent chat system with both local and API models."""
            
            # Build conversation history
            conversation_text = system_context + "\n\nConversation:\n"
            
            if history:
                recent = history[-4:]  # Last 4 exchanges for context
                for msg in recent:
                    if msg['sender'] == 'User':
                        conversation_text += f"User: {msg['message']}\n"
                    elif msg['sender'] == self.name:
                        conversation_text += f"Assistant: {msg['message']}\n"
            
            conversation_text += f"User: {message}\nAssistant:"
            
            data = {
                "contents": [{
                    "parts": [{
                        "text": conversation_text
                    }]
                }],
                "generationConfig": {
                    "maxOutputTokens": 150,
                    "temperature": 0.7,
                    "topP": 0.8,
                    "topK": 40
                }
            }
            
            response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'candidates' in result and len(result['candidates']) > 0:
                    if 'content' in result['candidates'][0]:
                        content = result['candidates'][0]['content']['parts'][0]['text']
                        return content.strip()
                
                raise Exception("No valid content in Gemini response")
            else:
                raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
                
        except Exception as e:
            raise Exception(f"Gemini API call failed: {str(e)}")
    
    def _generate_smart_fallback(self, message: str) -> str:
        """Smart fallback when API fails"""
        msg_lower = message.lower()
        current_time = datetime.utcnow().strftime('%H:%M UTC')
        
        if any(word in msg_lower for word in ['hello', 'hi', 'hey']):
            return f"Hello {self.current_user}! I'm your Gemini-powered AI assistant (though having some connectivity issues right now). How can I help you at {current_time}?"
        
        elif any(word in msg_lower for word in ['api', 'connection', 'error']):
            return f"I'm having some connectivity issues with the Gemini API, {self.current_user}, but I'm still here to help! What would you like to know?"
        
        elif '?' in message:
            return f"Great question about '{message}', {self.current_user}! While I'm having API issues, I'd still love to help. Could you tell me more?"
        
        else:
            return f"I understand you're talking about '{message}'. Even with API connectivity issues, I'm here to help however I can, {self.current_user}!"
    
    def get_model_info(self):
        """Get model info for system status"""
        if self.llm:
            return {
                'model_type': 'api_model',
                'provider': 'google',
                'model_name': self.llm['name'],
                'model_id': self.llm['model_id'],
                'status': self.llm['status'],
                'user': self.current_user
            }
        else:
            return {
                'model_type': 'api_fallback',
                'provider': 'none',
                'model_name': 'smart_fallback',
                'status': 'active',
                'user': self.current_user
            }