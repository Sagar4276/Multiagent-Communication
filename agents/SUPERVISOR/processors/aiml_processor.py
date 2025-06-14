"""
AI/ML Processor for Enhanced Supervisor Agent
Current Date and Time (UTC): 2025-06-14 18:27:12
Current User's Login: Sagar4276

Handles AI/ML agent routing and medical analysis workflows.
"""

import time
import os
from typing import Dict, Any, Optional
from ..utils.response_formatter import ResponseFormatter
from ..utils.time_utils import TimeUtils
from ..core.message_analysis import MessageAnalysis, MessageType

class AIMLProcessor:
    """Processor for AI/ML agent integration"""
    
    def __init__(self, config, response_formatter: ResponseFormatter, time_utils: TimeUtils):
        self.config = config
        self.response_formatter = response_formatter
        self.time_utils = time_utils
        self.name = "AIMLProcessor"
        
        print(f"🤖 [AIMLProcessor] Initialized AI/ML processing module")
    
    def process_aiml_request(self, user_id: str, message: str, analysis: MessageAnalysis, 
                           current_time: str, shared_memory, aiml_agent) -> str:
        """
        Process AI/ML requests with enhanced routing and context handling
        
        Args:
            user_id: User identifier
            message: User message
            analysis: Message analysis result
            current_time: Current timestamp
            shared_memory: Shared memory instance
            aiml_agent: AI/ML agent instance
            
        Returns:
            Formatted response string
        """
        
        if not aiml_agent:
            return self._handle_aiml_unavailable(user_id, message, analysis)
        
        try:
            print(f"🤖 [AIMLProcessor] Processing {analysis.type.value} request...")
            
            # Prepare context from analysis
            context = self._prepare_aiml_context(user_id, message, analysis, shared_memory)
            
            # Route based on message type
            response = self._route_aiml_request(user_id, message, analysis, context, aiml_agent)
            
            # Format enhanced response
            formatted_response = self._format_aiml_response(
                response, user_id, analysis, current_time
            )
            
            # Store interaction in shared memory
            self._store_aiml_interaction(user_id, message, response, shared_memory)
            
            return formatted_response
            
        except Exception as e:
            return self._handle_aiml_error(user_id, message, analysis, str(e))
    
    def _prepare_aiml_context(self, user_id: str, message: str, analysis: MessageAnalysis, 
                            shared_memory) -> Dict[str, Any]:
        """Prepare context for AI/ML processing"""
        
        context = analysis.context_info.copy() if analysis.context_info else {}
        
        # Add user session context
        context.update({
            'user_id': user_id,
            'message_type': analysis.type.value,
            'priority': analysis.priority.value,
            'requires_rag': analysis.requires_rag,
            'processing_flags': analysis.processing_flags
        })
        
        # Check for image upload context
        if analysis.type in [MessageType.IMAGE_UPLOAD, MessageType.MRI_ANALYSIS, MessageType.MEDICAL_IMAGE_ANALYSIS]:
            context = self._prepare_image_context(message, context)
        
        # Check for report generation context
        elif analysis.type == MessageType.REPORT_GENERATION:
            context = self._prepare_report_context(user_id, message, shared_memory, context)
        
        # Check for patient data context
        elif analysis.type == MessageType.PATIENT_DATA_COLLECTION:
            context = self._prepare_patient_data_context(message, context)
        
        return context
    
    def _prepare_image_context(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for image processing"""
        
        # Extract image path from message if present
        words = message.split()
        image_path = None
        
        # Look for file path patterns
        for word in words:
            if any(ext in word.lower() for ext in ['.jpg', '.jpeg', '.png', '.dcm', '.bmp', '.tiff']):
                image_path = word
                break
        
        # Determine image type
        image_type = 'general'
        if 'mri' in message.lower():
            image_type = 'mri'
        elif any(word in message.lower() for word in ['medical', 'scan', 'brain']):
            image_type = 'medical'
        
        context.update({
            'image_path': image_path or context.get('image_path'),
            'image_type': image_type,
            'analysis_requested': True
        })
        
        return context
    
    def _prepare_report_context(self, user_id: str, message: str, shared_memory, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for report generation"""
        
        # Check for existing analysis data
        analysis_data = shared_memory.get_temp_data(f'mri_analysis_{user_id}')
        
        # Extract patient information from message
        patient_info = self._extract_patient_info(message)
        
        context.update({
            'generate_report': True,
            'analysis_data_available': analysis_data is not None,
            'patient_info': patient_info
        })
        
        return context
    
    def _extract_patient_info(self, message: str) -> Dict[str, str]:
        """Extract patient information from message"""
        
        patient_info = {}
        lines = message.split('\n')
        
        for line in lines:
            line = line.strip()
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'name' in key:
                    patient_info['name'] = value
                elif 'age' in key:
                    patient_info['age'] = value
                elif 'sex' in key or 'gender' in key:
                    patient_info['sex'] = value
                elif 'contact' in key or 'phone' in key:
                    patient_info['contact'] = value
                elif 'email' in key:
                    patient_info['email'] = value
                elif 'doctor' in key:
                    patient_info['doctor'] = value
        
        return patient_info
    
    def _prepare_patient_data_context(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare context for patient data collection"""
        
        context.update({
            'collect_patient_data': True,
            'data_fields_requested': self._identify_data_fields(message)
        })
        
        return context
    
    def _identify_data_fields(self, message: str) -> list:
        """Identify what patient data fields are being requested"""
        
        fields = []
        msg_lower = message.lower()
        
        if 'name' in msg_lower:
            fields.append('name')
        if 'age' in msg_lower:
            fields.append('age')
        if 'sex' in msg_lower or 'gender' in msg_lower:
            fields.append('sex')
        if 'contact' in msg_lower or 'phone' in msg_lower:
            fields.append('contact')
        if 'email' in msg_lower:
            fields.append('email')
        if 'doctor' in msg_lower:
            fields.append('doctor')
        
        return fields if fields else ['all']
    
    def _route_aiml_request(self, user_id: str, message: str, analysis: MessageAnalysis, 
                          context: Dict[str, Any], aiml_agent) -> str:
        """Route request to appropriate AI/ML agent method"""
        
        try:
            # Use the AI/ML agent's main processing method
            response = aiml_agent.process_message(user_id, message, context)
            return response
            
        except Exception as e:
            # If direct processing fails, try specific method based on type
            return self._fallback_aiml_processing(user_id, message, analysis, context, aiml_agent, str(e))
    
    def _fallback_aiml_processing(self, user_id: str, message: str, analysis: MessageAnalysis, 
                                context: Dict[str, Any], aiml_agent, error: str) -> str:
        """Fallback processing for AI/ML requests"""
        
        if analysis.type == MessageType.MRI_ANALYSIS:
            return f"""🧠 **MRI ANALYSIS REQUEST RECEIVED**

📁 **Image Path:** {context.get('image_path', 'Not specified')}
🎯 **Analysis Type:** {analysis.type.value}
⚠️ **Status:** Processing error occurred

❌ **Error:** {error}

💡 **Next Steps:**
1. Verify image file path and format
2. Ensure file is accessible
3. Try re-uploading the image
4. Contact support if issue persists

🔧 **Supported Formats:** .dcm, .jpg, .jpeg, .png, .bmp, .tiff"""
        
        elif analysis.type == MessageType.REPORT_GENERATION:
            return f"""📄 **MEDICAL REPORT GENERATION REQUEST**

👤 **Patient:** {context.get('patient_info', {}).get('name', 'Not specified')}
📊 **Data Available:** {context.get('analysis_data_available', False)}
⚠️ **Status:** Processing error occurred

❌ **Error:** {error}

💡 **Requirements for Report Generation:**
1. Prior MRI analysis completed
2. Patient information collected
3. Classification results available

🔧 **To Generate Report:**
1. Upload and analyze MRI first
2. Provide patient details
3. Request report generation"""
        
        else:
            return f"""🤖 **AI/ML PROCESSING REQUEST**

🎯 **Request Type:** {analysis.type.value}
⚠️ **Status:** Processing error occurred

❌ **Error:** {error}

💡 **Available AI/ML Features:**
• MRI Image Analysis
• Medical Report Generation
• Patient Data Management
• Stage Classification

🔧 **Try:**
• `upload mri [path]` - Analyze MRI scan
• `generate report` - Create medical report
• `check reports [name]` - Search reports"""
    
    def _format_aiml_response(self, response: str, user_id: str, analysis: MessageAnalysis, 
                            current_time: str) -> str:
        """Format AI/ML response with enhanced metadata"""
        
        if not response:
            return "❌ **AI/ML Agent Error:** No response generated"
        
        # Add enhanced footer with AI/ML specific information
        footer = f"""

---
🤖 **Enhanced AI/ML Processing Report**
🔄 **Route:** Supervisor v2.0.0 → AI/ML Agent → Medical Analysis
🎯 **Analysis Type:** {analysis.type.value.replace('_', ' ').title()}
📊 **Routing Confidence:** {analysis.confidence:.1%}
🏥 **Medical System:** {'Active' if analysis.processing_flags.get('medical_related') else 'N/A'}
🖼️ **Image Processing:** {'Enabled' if analysis.processing_flags.get('image_processing') else 'Not Required'}
📄 **Report System:** {'Active' if analysis.processing_flags.get('report_related') else 'Not Required'}
👤 **User:** {user_id} | 🕐 **Completed:** {current_time} UTC"""
        
        # Add specific metadata based on message type
        if analysis.type in [MessageType.MRI_ANALYSIS, MessageType.MEDICAL_IMAGE_ANALYSIS]:
            footer += f"""
🔬 **Medical Imaging:** AI-powered analysis complete
📈 **Classification:** Stage prediction with confidence scoring"""
        
        elif analysis.type == MessageType.REPORT_GENERATION:
            footer += f"""
📋 **Report Generation:** PDF medical report created
📚 **Knowledge Integration:** RAG-enhanced recommendations"""
        
        elif analysis.type == MessageType.REPORT_SEARCH:
            footer += f"""
🔍 **Report Search:** Patient history and previous analyses"""
        
        return response + footer
    
    def _store_aiml_interaction(self, user_id: str, message: str, response: str, shared_memory):
        """Store AI/ML interaction in shared memory"""
        
        try:
            # Store the AI/ML interaction
            shared_memory.add_message(user_id, response, "AIMLAgent")
            
            # Store interaction metadata
            interaction_data = {
                'timestamp': time.time(),
                'user_id': user_id,
                'message': message[:100] + "..." if len(message) > 100 else message,
                'response_length': len(response),
                'interaction_type': 'aiml_processing'
            }
            
            shared_memory.store_temp_data(f'aiml_interaction_{user_id}_{int(time.time())}', interaction_data)
            
        except Exception as e:
            print(f"⚠️ [AIMLProcessor] Failed to store interaction: {str(e)}")
    
    def _handle_aiml_unavailable(self, user_id: str, message: str, analysis: MessageAnalysis) -> str:
        """Handle case when AI/ML agent is not available"""
        
        return f"""❌ **AI/ML AGENT NOT AVAILABLE**

🤖 **Requested Service:** {analysis.type.value.replace('_', ' ').title()}
👤 **User:** {user_id}
📨 **Message:** {message[:50]}{'...' if len(message) > 50 else ''}

🎯 **AI/ML Capabilities (When Available):**
• 🧠 MRI Image Analysis
• 🖼️ Medical Image Processing
• 📊 Parkinson's Stage Classification
• 📄 Medical Report Generation
• 👥 Patient Data Management
• 🔍 Report Search & History

💡 **Alternative Options:**
• Research Questions → Try general medical research queries
• System Status → Type `status` to check system health
• General Chat → Use regular conversation

🔧 **Technical Information:**
• AI/ML Agent Status: Disconnected
• Supervisor Version: 2.0.0
• Routing Confidence: {analysis.confidence:.1%}
• Message Type: {analysis.type.value}

📞 **Support:** Contact system administrator to enable AI/ML features"""
    
    def _handle_aiml_error(self, user_id: str, message: str, analysis: MessageAnalysis, error: str) -> str:
        """Handle AI/ML processing errors"""
        
        return f"""❌ **AI/ML PROCESSING ERROR**

🔴 **Error Details:**
• Message: {error}
• User: {user_id}
• Request Type: {analysis.type.value}
• Time: {time.strftime('%Y-%m-%d %H:%M:%S')} UTC

🎯 **Request Information:**
• Original Message: {message[:100]}{'...' if len(message) > 100 else ''}
• Routing Confidence: {analysis.confidence:.1%}
• Processing Flags: {analysis.processing_flags}

🔧 **Troubleshooting Steps:**
1. Check if AI/ML agent is properly initialized
2. Verify file paths for image uploads
3. Ensure patient data format is correct
4. Try rephrasing your request

💡 **Alternative Actions:**
• System Status: Type `status` for health check
• General Questions: Try research queries instead
• Help: Type `help` for available commands

🆘 **If Error Persists:**
Contact system administrator with error details above."""
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Get processor information"""
        return {
            'name': self.name,
            'version': '1.0.0',
            'capabilities': [
                'MRI Image Analysis Routing',
                'Medical Report Generation Coordination',
                'Patient Data Collection',
                'Medical Image Processing',
                'Report Search and History',
                'Error Handling and Recovery'
            ],
            'supported_message_types': [
                'IMAGE_UPLOAD',
                'MRI_ANALYSIS',
                'MEDICAL_IMAGE_ANALYSIS',
                'REPORT_GENERATION',
                'REPORT_SEARCH',
                'PATIENT_DATA_COLLECTION',
                'MEDICAL_CLASSIFICATION',
                'MEDICAL_CONSULTATION'
            ]
        }