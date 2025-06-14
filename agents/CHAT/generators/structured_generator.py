"""
Enhanced Chat Agent - Structured Response Generator (REAL CONTENT FIX)
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-13 06:35:20
Current User's Login: Sagar4276
"""

from typing import List, Dict, Any
import re

class StructuredGenerator:
    """SAME CLASS NAME - Actually uses your real research content"""
    
    def __init__(self, rag_agent, agent_name: str = "StructuredGenerator"):
        self.agent_name = agent_name
        self.rag_agent = rag_agent
        self.current_user = "Sagar4276"
    
    def create_structured_rag_response(self, query: str, retrieval_results: list, current_time: str) -> str:
        """SAME FUNCTION NAME - Use ACTUAL content from your research"""
        
        if not retrieval_results:
            return f"I couldn't find specific information about '{query}' in the research papers. Could you try asking with different keywords?"
        
        # Debug what we actually found
        print(f"[{self.agent_name}] 🔍 Processing '{query}' with {len(retrieval_results)} results")
        
        best_result = retrieval_results[0]
        print(f"[{self.agent_name}] 📄 Source: {best_result.source}")
        print(f"[{self.agent_name}] 🎯 Similarity: {best_result.similarity_score:.3f}")
        print(f"[{self.agent_name}] 📝 Content length: {len(best_result.content)} chars")
        print(f"[{self.agent_name}] 📝 First 200 chars: '{best_result.content[:200]}'")
        
        # Extract and use REAL content
        return self._create_response_from_real_content(query, best_result)
    
    def _create_response_from_real_content(self, query: str, result) -> str:
        """Create response using your actual research content"""
    
        query_lower = query.lower()
        content = result.content
        source_name = result.source.replace('.md', '').replace('_', ' ').replace('.pdf', '')
    
    # Clean and split content into meaningful sentences
        sentences = self._extract_meaningful_sentences(content)
        print(f"[{self.agent_name}] 📊 Extracted {len(sentences)} sentences from content")
    
    # Find most relevant sentences based on query
        relevant_sentences = self._find_relevant_sentences(sentences, query_lower)
        print(f"[{self.agent_name}] ✅ Found {len(relevant_sentences)} relevant sentences")
    
        if not relevant_sentences:
        # If no specific match, use first meaningful sentence
            if sentences:
                return f"Based on {source_name}, {sentences[0]}."
            else:
                return f"I found information in {source_name} but had trouble extracting specific details about '{query}'."
    
    # Create natural response from actual content
        if "parkinson" in query_lower:
            response = f"Based on the research, Parkinson's disease {relevant_sentences[0].lower()}"
        else:
            response = f"According to the research, {relevant_sentences[0]}"
    
    # Fix grammar and add period
        if not response.endswith('.'):
            response += '.'
    
    # Add additional context if available
        if len(relevant_sentences) > 1:
            additional = relevant_sentences[1]
            if not additional.startswith('It ') and not additional.startswith('This '):
                additional = additional.lower()
            response += f" The research also shows that {additional}"
            if not response.endswith('.'):
                response += '.'
    
        return response
    
    def _extract_meaningful_sentences(self, content: str) -> List[str]:
        """Extract meaningful sentences from content"""
        
        # Clean the content
        content = re.sub(r'\n+', ' ', content)  # Replace newlines with spaces
        content = re.sub(r'\s+', ' ', content)  # Multiple spaces to single
        content = content.strip()
        
        # Split into sentences
        sentences = []
        for sentence in content.split('.'):
            sentence = sentence.strip()
            
            # Skip headers, short fragments, bullet points
            if (len(sentence) > 30 and 
                not sentence.startswith('#') and 
                not sentence.startswith('-') and
                not sentence.startswith('*') and
                not sentence.isdigit()):
                
                # Clean up the sentence
                sentence = sentence.replace('**', '').replace('##', '')
                if sentence and sentence[0].islower():
                    sentence = sentence[0].upper() + sentence[1:]
                
                sentences.append(sentence)
        
        return sentences[:10]  # Top 10 sentences
    
    def _find_relevant_sentences(self, sentences: List[str], query: str) -> List[str]:
        """Find sentences most relevant to the query"""
        
        relevant = []
        query_words = [word.lower() for word in query.split() if len(word) > 2]
        
        print(f"[{self.agent_name}] 🔍 Looking for words: {query_words}")
        
        # Score sentences based on relevance
        scored_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            score = 0
            
            # Direct word matches
            for word in query_words:
                if word in sentence_lower:
                    score += 2
                    print(f"[{self.agent_name}] ✅ Found '{word}' in: '{sentence[:80]}...'")
            
            # Related terms for Parkinson's
            if "parkinson" in query.lower():
                related_terms = [
                    'neurological disorder', 'dopamine', 'movement', 'tremor', 
                    'bradykinesia', 'rigidity', 'symptoms', 'progressive'
                ]
                for term in related_terms:
                    if term in sentence_lower:
                        score += 1
            
            # Definitional indicators
            if any(phrase in sentence_lower for phrase in [
                'is a', 'is the', 'occurs when', 'affects', 'causes', 
                'leads to', 'characterized by', 'includes', 'symptoms'
            ]):
                score += 1
            
            if score > 0:
                scored_sentences.append((score, sentence))
        
        # Sort by score and return top sentences
        scored_sentences.sort(reverse=True, key=lambda x: x[0])
        relevant = [sentence for score, sentence in scored_sentences[:3]]
        
        print(f"[{self.agent_name}] 📋 Top relevant sentences:")
        for i, sentence in enumerate(relevant):
            print(f"[{self.agent_name}]   {i+1}. '{sentence[:100]}...'")
        
        return relevant
    
    def create_general_responses(self, user_id: str, message: str, current_time: str, rag_agent) -> str:
        """SAME FUNCTION NAME - Create conversational general responses"""
        
        msg_lower = message.lower().strip()
        
        # Greeting responses
        if any(word in msg_lower for word in ['hello', 'hi', 'hey', 'good morning', 'good afternoon']):
            papers_count = len(getattr(rag_agent, 'documents', []))
            return f"Hello {user_id}! I'm your research assistant. I have access to {papers_count} research papers and I'm ready to help you find information. What would you like to know?"
        
        # How are you responses
        elif any(phrase in msg_lower for phrase in ['how are you', 'how\'re you', 'how do you do']):
            return f"I'm doing great, thank you for asking {user_id}! I'm here and ready to help you with research questions. What can I help you explore today?"
        
        # Thank you responses
        elif any(word in msg_lower for word in ['thank', 'thanks']):
            return f"You're very welcome, {user_id}! I'm happy to help. Feel free to ask me anything else about the research!"
        
        # Goodbye responses
        elif any(word in msg_lower for word in ['bye', 'goodbye', 'see you', 'later']):
            return f"Goodbye {user_id}! It was great helping you with your research. Come back anytime you have questions!"
        
        # Default conversational response
        else:
            papers_count = len(getattr(rag_agent, 'documents', []))
            return f"I understand, {user_id}. I'm here to help you with research questions from my collection of {papers_count} papers. Try asking me about specific topics you're interested in!"