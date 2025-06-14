"""
Response Formatting and Enhancement
Current Date and Time (UTC): 2025-06-14 10:43:05
Current User's Login: Sagar4276

Handles response formatting and enhancement - EASY TO MODIFY FOR LONGER RESPONSES
"""

from typing import Dict, Any
from ..core.message_analysis import MessageAnalysis
from ..core.supervisor_config import SupervisorConfig

class ResponseFormatter:
    """Handles response formatting and enhancement"""
    
    def __init__(self, config: SupervisorConfig):
        self.config = config
    
    def enhance_rag_response(self, response: str, user_id: str, analysis: MessageAnalysis, 
                           processing_time: float, current_time: str) -> str:
        """Enhanced RAG response formatting with configurable length"""
        
        # Clean up malformed responses
        cleaned_response = self._clean_response(response)
        
        # Check if response already has research branding
        if "**Research Analysis:**" in cleaned_response or "**RAG Analysis:**" in cleaned_response:
            enhanced = cleaned_response
            enhanced += f"\n\n---\n"
            enhanced += f"🟢 **Enhanced Processing:** Supervisor v{self.config.version} → RAG System\n"
            enhanced += f"🎯 **Query Classification:** {analysis.type.value.replace('_', ' ').title()}\n"
            enhanced += f"📊 **Confidence:** {analysis.confidence:.1%} | **Processing Time:** {processing_time:.2f}s\n"
            enhanced += f"👤 **User:** {user_id} | 🕐 **Completed:** {current_time} UTC"
        else:
            enhanced = f"🟢 **ENHANCED SUPERVISOR → RAG RESEARCH SYSTEM**\n\n"
            enhanced += f"**Research Analysis for:** {analysis.keywords[0] if analysis.keywords else 'your query'}\n\n"
            enhanced += cleaned_response
            enhanced += f"\n\n---\n"
            enhanced += f"🔄 **Enhanced Route:** Supervisor v{self.config.version} → RAG Agent → Knowledge Analysis\n"
            enhanced += f"🎯 **Query Type:** {analysis.type.value.replace('_', ' ').title()}\n"
            enhanced += f"📊 **Analysis:** Confidence {analysis.confidence:.1%}, Keywords: {', '.join(analysis.keywords[:3])}\n"
            enhanced += f"⚡ **Performance:** {processing_time:.2f}s processing time\n"
            enhanced += f"👤 **User:** {user_id} | 🕐 **Completed:** {current_time} UTC"
        
        return enhanced
    
    def enhance_chat_response(self, response: str, user_id: str, analysis: MessageAnalysis, 
                            processing_time: float) -> str:
        """Enhanced chat response formatting"""
        # Don't modify research responses
        if "Research Analysis:" in response or "RAG Analysis:" in response:
            return response
        
        # Add enhanced branding for general chat
        if not response.startswith("🟢"):
            enhanced = f"🟢 {response}"
            
            # Add processing info for complex queries
            if processing_time > 1.0 or analysis.confidence < 0.7:
                enhanced += f"\n\n💡 *Processed by Enhanced Supervisor in {processing_time:.2f}s*"
            
            return enhanced
        
        return response
    
    def _clean_response(self, response: str) -> str:
        """Clean up malformed responses"""
        if "You are a medical research expert" in response or "RESEARCH FINDINGS:" in response:
            # Extract useful content from malformed response
            lines = response.split('\n')
            useful_lines = []
            for line in lines:
                if line.strip() and not line.startswith("You are") and "RESEARCH FINDINGS:" not in line:
                    useful_lines.append(line.strip())
            
            if useful_lines:
                cleaned_response = '\n'.join(useful_lines[:15])  # Take first 15 useful lines
            else:
                cleaned_response = "Research content found but formatting needs improvement."
        else:
            cleaned_response = response
        
        return cleaned_response
    
    def apply_length_enhancement(self, response: str) -> str:
        """Apply length enhancement based on configuration"""
        # This is where you can add logic to extend responses
        if len(response.split()) < 50:  # If response is short
            # Add more detailed explanation request
            response += "\n\n🔍 **For more detailed information**, please ask specific follow-up questions about any aspect that interests you."
        
        return response