"""
Enhanced Chat Agent - LLM Generation Module (FINAL GENERIC VERSION)
Current Date and Time (UTC): 2025-06-13 19:56:46
Current User's Login: Sagar4276
"""

import torch
from typing import Dict, Any, List
import re

class LLMGenerator:
    """FINAL VERSION - Domain-agnostic conversational SLM"""
    
    def __init__(self, model_loader, agent_name: str = "LLMGenerator"):
        self.agent_name = agent_name
        self.model_loader = model_loader
        self.current_user = "Sagar4276"
    
    def generate_rag_response(self, context: str, query: str, retrieval_results: list) -> Dict[str, Any]:
        """Generate conversational responses for ANY domain"""
        llm = self.model_loader.get_llm()
        
        if not llm or not llm.get('model'):
            return self._create_direct_response(query, retrieval_results)
        
        try:
            print(f"[{self.agent_name}] 🤖 Generating conversational response with {llm['model_name']}...")
            
            # Create GENERIC conversational prompt
            prompt = self._create_conversational_prompt(context, query)
            print(f"[{self.agent_name}] 📝 Conversational prompt created")
            
            # Generate with optimized settings
            response = self._generate_conversation(llm, prompt)
            print(f"[{self.agent_name}] 🎯 Generated: '{response[:100]}...'")
            
            # Lenient generic quality check
            if response and self._is_reasonable_response(response, query):
                print(f"[{self.agent_name}] ✅ Response passed quality check")
                return {
                    'success': True,
                    'answer': response,
                    'model_used': llm['model_name'],
                    'generation_info': 'Conversational response generated'
                }
            else:
                print(f"[{self.agent_name}] ⚠️ Using fallback response")
                return self._create_direct_response(query, retrieval_results)
                
        except Exception as e:
            print(f"[{self.agent_name}] ❌ Generation error: {str(e)}")
            return self._create_direct_response(query, retrieval_results)
    
    def _create_conversational_prompt(self, context: str, query: str) -> str:
        """Create domain-agnostic conversational prompt"""
        
        # Extract key information (generic)
        key_info = self._extract_key_info(context)
        
        # Create simple, friendly prompt that works for any domain
        prompt = f"""You are a helpful AI assistant talking to {self.current_user}. Be friendly and helpful.

{self.current_user} asked: "{query}"

Information available:
{key_info}

Respond in a helpful, friendly way. Explain things clearly and naturally.

Response: Hi {self.current_user}! """
        
        return prompt
    
    def _extract_key_info(self, context: str) -> str:
        """Extract key information - works for any domain"""
        
        if not context:
            return "I have information available to help answer your question."
        
        # Clean context
        context_clean = re.sub(r'\n+', ' ', context)
        context_clean = re.sub(r'\s+', ' ', context_clean).strip()
        
        # Keep reasonable length for SLM
        if len(context_clean) > 300:
            sentences = context_clean.split('.')
            key_sentences = []
            char_count = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if char_count + len(sentence) < 300 and len(sentence) > 10:
                    key_sentences.append(sentence)
                    char_count += len(sentence)
                else:
                    break
            
            if key_sentences:
                return '. '.join(key_sentences) + '.'
        
        return context_clean
    
    def _generate_conversation(self, llm: dict, prompt: str) -> str:
        """Generate with SLM-optimized settings"""
        
        try:
            # Encode the prompt
            inputs = llm['tokenizer'].encode(
                prompt,
                return_tensors='pt',
                max_length=400,
                truncation=True
            )
            
            # Generate with optimized settings for small models
            with torch.no_grad():
                outputs = llm['model'].generate(
                    inputs,
                    max_new_tokens=200,   # Good length for conversation
                    min_new_tokens=20,   # Ensure substance
                    temperature=0.7,     # Balanced creativity
                    do_sample=True,
                    top_k=40,           # Focused but creative
                    top_p=0.9,          # Good diversity
                    repetition_penalty=1.1,  # Prevent repetition
                    pad_token_id=llm['tokenizer'].eos_token_id,
                    eos_token_id=llm['tokenizer'].eos_token_id,
                    no_repeat_ngram_size=2
                )
            
            # Decode and extract
            full_response = llm['tokenizer'].decode(outputs[0], skip_special_tokens=True)
            generated_text = full_response.replace(prompt, "").strip()
            
            return self._format_conversational_response(generated_text, prompt)
            
        except Exception as e:
            print(f"[{self.agent_name}] ⚠️ Generation error: {e}")
            return ""
    
    def _format_conversational_response(self, response: str, original_prompt: str) -> str:
        """Format response to be conversational - domain agnostic"""
        
        if not response:
            return ""
        
        # Clean response
        response = response.strip()
        response = re.sub(r'\s+', ' ', response)
        
        # Remove artifacts
        artifacts = ['Response:', 'Hi ' + self.current_user + '!', 'Assistant:', 'AI:']
        for artifact in artifacts:
            response = response.replace(artifact, '').strip()
        
        # Ensure proper capitalization
        if response and response[0].islower():
            response = response[0].upper() + response[1:]
        
        # Add conversational starter if needed
        if response and not response.startswith(('Based on', 'According to', 'The information', 'I can')):
            response = f"Based on the information I found, {response.lower()}"
        
        # Ensure proper ending
        if response and not response.endswith(('.', '!', '?')):
            response += '.'
        
        # Add helpful ending if too short
        if len(response) < 40:
            response += " Would you like me to explain any specific aspect in more detail?"
        
        return response
    
    def _is_reasonable_response(self, response: str, query: str) -> bool:
        """Generic quality check - works for any domain"""
        
        if not response or len(response.strip()) < 8:
            return False
        
        response_lower = response.lower()
        
        # Basic checks
        words = response.split()
        if len(words) < 4:
            return False
        
        # Check for obvious repetition
        if len(set(words)) < len(words) * 0.5:  # More than 50% repeated words
            return False
        
        # Check for obvious nonsense
        nonsense_patterns = [
            'aaaaaaa', 'xxxxxxx', 'error error', 'undefined undefined',
            'null null', 'test test test'
        ]
        
        if any(pattern in response_lower for pattern in nonsense_patterns):
            return False
        
        # Should have some structure
        if not any(char in response for char in '.!?'):
            return False
        
        # Should contain helpful words (domain-agnostic)
        helpful_words = [
            'information', 'help', 'show', 'explain', 'find', 'research',
            'study', 'data', 'based', 'according', 'include', 'cause',
            'result', 'provide', 'available', 'known', 'understand'
        ]
        
        # For any substantive query, should contain some helpful language
        if len(query.split()) > 2:  # Not just single word queries
            if not any(word in response_lower for word in helpful_words):
                # Be more lenient - if it has reasonable length, accept it
                return len(response) > 25
        
        return True
    
    def _create_direct_response(self, query: str, retrieval_results: list) -> Dict[str, Any]:
        """Create direct response - domain agnostic"""
        
        if not retrieval_results:
            response = f"I'd be happy to help you with '{query}'. However, I don't have specific information on this topic in my current knowledge base. Could you try rephrasing your question or asking about a different aspect?"
        else:
            # Use best result
            best_result = retrieval_results[0]
            content = best_result.content
            
            # Extract relevant content
            if len(content) > 400:
                # Try to find most relevant sentences
                sentences = content.split('.')
                query_words = set(query.lower().split())
                relevant_sentences = []
                
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 15:
                        sentence_words = set(sentence.lower().split())
                        if query_words & sentence_words:  # Has overlap
                            relevant_sentences.append(sentence)
                
                if relevant_sentences:
                    content = '. '.join(relevant_sentences[:2]) + '.'
                else:
                    content = content[:400] + '...'
            
            response = f"Based on the information I found:\n\n{content}\n\nWould you like me to explain any specific aspect in more detail?"
        
        return {
            'success': True,
            'answer': response,
            'model_used': 'Direct Response System',
            'generation_info': 'RAG-based direct response'
        }
    
    def generate_general_response(self, message: str, user_id: str, context: str = "") -> Dict[str, Any]:
        """Generate friendly general conversation - domain agnostic"""
        
        message_lower = message.lower().strip()
        
        # Generic friendly responses
        if any(greeting in message_lower for greeting in ['hello', 'hi', 'hey']):
            response = f"Hello {self.current_user}! I'm your AI assistant. I can help you find information and answer questions based on my knowledge base. What would you like to know about?"
        
        elif "how are you" in message_lower:
            response = f"I'm doing great, thanks for asking! I'm here and ready to help you with any questions you might have. What can I help you with today?"
        
        elif any(thanks in message_lower for thanks in ['thank', 'thanks']):
            response = "You're very welcome! I'm happy to help. If you have any other questions or want to explore different topics, just let me know!"
        
        elif "help" in message_lower:
            response = "I'm here to help! I can search through my knowledge base to answer questions on various topics. Just ask me about anything you'd like to know more about."
        
        else:
            # Try to generate with LLM
            llm = self.model_loader.get_llm()
            if llm and llm.get('model'):
                try:
                    prompt = f"User says: {message}\nI respond helpfully: I'm an AI assistant and I"
                    generated = self._generate_conversation(llm, prompt)
                    if generated and len(generated.strip()) > 10:
                        # Clean the generated response
                        generated = generated.replace("I'm an AI assistant and I", "I'm an AI assistant and I").strip()
                        response = generated
                    else:
                        response = "I'm here to help! What would you like to know about?"
                except:
                    response = "I'm here to help! What would you like to know about?"
            else:
                response = "I'm here to help! What would you like to know about?"
        
        return {
            'success': True,
            'answer': response,
            'model_used': 'Conversational System',
            'generation_info': 'Generic conversational response'
        }