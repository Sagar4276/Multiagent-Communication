"""
AI/ML Agent - Medical Image Analysis & Report Generation
Current Date and Time (UTC): 2025-06-14 11:56:12
Current User's Login: Sagar4276

Integrates with existing CHAT, RAG, and SUPERVISOR agents.
"""

import os
import time
from typing import Dict, Any, Optional
from .processors.image_processor import ImageProcessor
from .processors.stage_classifier import StageClassifier  
from .processors.report_generator import ReportGenerator

class AIMLAgent:
    """
    AI/ML Agent that integrates with your existing 3-agent system
    
    Capabilities:
    - MRI image processing
    - Parkinson's stage classification  
    - Medical report generation
    - Integration with RAG for treatment recommendations
    """
    
    def __init__(self, shared_memory):
        self.shared_memory = shared_memory
        self.name = "AIMLAgent"
        self.version = "1.0.0"
        
        # Initialize processors (following your existing pattern)
        self.image_processor = ImageProcessor(self)
        self.stage_classifier = StageClassifier(self)
        self.report_generator = ReportGenerator(self)
        
        # Integration flags for existing agents
        self.chat_agent = None  # Will be set by supervisor
        self.rag_agent = None   # Will be set by supervisor
        
        print(f"🤖 [AIMLAgent] Initialized v{self.version}")
        print(f"🔗 [AIMLAgent] Ready for integration with CHAT and RAG agents")
    
    def set_agent_connections(self, chat_agent=None, rag_agent=None):
        """Connect to existing CHAT and RAG agents"""
        self.chat_agent = chat_agent
        self.rag_agent = rag_agent
        print(f"🔗 [AIMLAgent] Connected to existing agents")
    
    def process_message(self, user_id: str, message: str, context: Dict[str, Any] = None) -> str:
        """
        Main processing method that matches your existing agent pattern
        
        Args:
            user_id: User identifier
            message: User message or command
            context: Additional context (image paths, etc.)
            
        Returns:
            Response string (matches chat_agent pattern)
        """
        
        try:
            # Check if this is an image upload request
            if context and context.get('image_path'):
                return self._handle_image_analysis(user_id, message, context)
            
            # Check if this is a report generation request
            elif 'generate report' in message.lower() or context and context.get('generate_report'):
                return self._handle_report_generation(user_id, message, context)
            
            # Check for previous reports
            elif 'previous report' in message.lower() or 'check reports' in message.lower():
                return self._handle_previous_reports(user_id, message)
            
            # Default: Pass to chat agent for general AI/ML questions
            else:
                return self._handle_general_aiml_query(user_id, message)
                
        except Exception as e:
            return f"❌ **AI/ML Processing Error**\n\nError: {str(e)}\n\nPlease try again or contact support."
    
    def _handle_image_analysis(self, user_id: str, message: str, context: Dict[str, Any]) -> str:
        """Handle MRI/image analysis workflow"""
        
        image_path = context.get('image_path')
        image_type = context.get('image_type', 'general')  # 'mri' or 'general'
        
        if image_type.lower() == 'mri':
            return self._process_mri_workflow(user_id, image_path)
        else:
            return self._process_general_image(user_id, image_path)
    
    def _process_mri_workflow(self, user_id: str, image_path: str) -> str:
        """Complete MRI analysis workflow"""
        
        response_parts = []
        
        # Step 1: Process MRI image
        response_parts.append("🧠 **MRI ANALYSIS WORKFLOW INITIATED**\n")
        response_parts.append(f"👤 **Patient ID:** {user_id}")
        response_parts.append(f"📁 **Image:** {os.path.basename(image_path)}")
        response_parts.append(f"🕐 **Analysis Time:** {time.strftime('%Y-%m-%d %H:%M:%S')} UTC\n")
        
        # Step 2: Extract features
        response_parts.append("🔍 **STEP 1: IMAGE PROCESSING**")
        features = self.image_processor.extract_mri_features(image_path)
        
        if features['status'] == 'success':
            response_parts.append("✅ MRI image processed successfully")
            response_parts.append(f"📊 Features extracted: {features['feature_count']}")
            response_parts.append(f"🖼️ Image quality: {features['quality_assessment']}\n")
        else:
            return f"❌ **MRI Processing Failed**\n\n{features['error']}"
        
        # Step 3: Stage classification
        response_parts.append("🤖 **STEP 2: AI CLASSIFICATION**")
        classification = self.stage_classifier.classify_stage(features['data'])
        
        if classification['status'] == 'success':
            stage = classification['predicted_stage']
            confidence = classification['confidence']
            
            response_parts.append(f"🎯 **Predicted Stage:** {stage}")
            response_parts.append(f"📈 **Confidence:** {confidence:.1%}")
            response_parts.append(f"📋 **Description:** {classification['stage_description']}\n")
            
            # Step 4: Get RAG recommendations
            response_parts.append("📚 **STEP 3: TREATMENT RECOMMENDATIONS**")
            if self.rag_agent:
                rag_query = f"treatment recommendations for {stage}"
                treatment_info = self.rag_agent.get_relevant_info(rag_query)
                response_parts.append(f"💊 **Treatment Options:** {treatment_info[:200]}...\n")
            
            # Step 5: Store for report generation
            self._store_analysis_results(user_id, {
                'image_path': image_path,
                'features': features,
                'classification': classification,
                'timestamp': time.time()
            })
            
            response_parts.append("📄 **NEXT STEP: REPORT GENERATION**")
            response_parts.append("To generate a medical report, please provide:")
            response_parts.append("• Patient Name")
            response_parts.append("• Age") 
            response_parts.append("• Sex")
            response_parts.append("• Contact Information")
            response_parts.append("\nType: `generate report` when ready.")
            
        else:
            response_parts.append(f"❌ Classification failed: {classification['error']}")
        
        return "\n".join(response_parts)
    
    def _handle_report_generation(self, user_id: str, message: str, context: Dict[str, Any] = None) -> str:
        """Handle medical report generation"""
        
        # Get stored analysis results
        analysis_data = self.shared_memory.get_temp_data(f'mri_analysis_{user_id}')
        
        if not analysis_data:
            return "❌ **No Analysis Data Found**\n\nPlease upload and analyze an MRI image first."
        
        # Check for patient information
        patient_info = context.get('patient_info') if context else None
        
        if not patient_info:
            return """📋 **PATIENT INFORMATION REQUIRED**

Please provide the following information for report generation:

🏷️ **Patient Name:** 
🎂 **Age:** 
⚧️ **Sex:** (Male/Female/Other)
📞 **Contact:** 
📧 **Email:** (Optional)
🏥 **Doctor:** (Optional)

Example format:
Name: John Doe
Age: 65
Sex: Male
Contact: +1-555-0123
Email: john.doe@email.com
Doctor: Dr. Smith"""
        
        # Generate report
        report_result = self.report_generator.generate_pdf_report(
            user_id, analysis_data, patient_info
        )
        
        if report_result['status'] == 'success':
            # Add to knowledge base for future queries
            self._add_report_to_knowledge_base(user_id, report_result)
            
            return f"""✅ **MEDICAL REPORT GENERATED**

📄 **Report ID:** {report_result['report_id']}
👤 **Patient:** {patient_info.get('name', 'Unknown')}
🕐 **Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')} UTC
📁 **File:** {report_result['pdf_path']}

📊 **Report Contents:**
• MRI Analysis Results
• Stage Classification: {analysis_data['classification']['predicted_stage']}
• Treatment Recommendations
• Follow-up Guidelines

📥 **Download:** Report saved to downloads folder
📚 **Knowledge Base:** Report added for future reference

🔍 **Query Previous Reports:** Type `check reports {patient_info.get('name', '')}`"""
        
        else:
            return f"❌ **Report Generation Failed**\n\n{report_result['error']}"
    
    def _handle_previous_reports(self, user_id: str, message: str) -> str:
        """Handle previous report queries"""
        
        # Extract patient name from message
        parts = message.lower().split('check reports')
        patient_name = parts[1].strip() if len(parts) > 1 else None
        
        if not patient_name:
            return """📋 **PREVIOUS REPORTS QUERY**

Please specify patient name:
Example: `check reports John Doe`

Or use: `list all reports` to see all reports"""
        
        # Search for previous reports
        reports = self.report_generator.search_previous_reports(patient_name)
        
        if reports:
            response_parts = [f"📚 **PREVIOUS REPORTS FOR {patient_name.upper()}**\n"]
            
            for i, report in enumerate(reports, 1):
                response_parts.append(f"{i}. **{report['date']}** - Stage: {report['stage']}")
                response_parts.append(f"   📁 File: {report['filename']}")
                response_parts.append(f"   🆔 ID: {report['report_id']}\n")
            
            response_parts.append("❓ **Options:**")
            response_parts.append("• View specific report: `view report [ID]`")
            response_parts.append("• Generate new report: Upload new MRI")
            response_parts.append("• Compare reports: `compare reports [ID1] [ID2]`")
            
            return "\n".join(response_parts)
        else:
            return f"📭 **NO PREVIOUS REPORTS**\n\nNo previous reports found for {patient_name}.\n\nTo create a new report, please upload an MRI image."
    
    def _handle_general_aiml_query(self, user_id: str, message: str) -> str:
        """Handle general AI/ML related questions"""
        
        if self.chat_agent:
            # Use chat agent for general AI/ML questions
            ai_response = self.chat_agent.process_message(user_id, message)
            
            # Add AI/ML agent branding
            return f"🤖 **AI/ML Agent Response**\n\n{ai_response}\n\n💡 **AI/ML Capabilities:**\n• MRI Image Analysis\n• Stage Classification\n• Medical Report Generation\n• Integration with Knowledge Base"
        else:
            return "🤖 **AI/ML Agent**\n\nAI/ML agent is ready for:\n• MRI image analysis\n• Medical report generation\n• Stage classification\n\nPlease upload an MRI image to begin analysis."
    
    def _store_analysis_results(self, user_id: str, data: Dict[str, Any]):
        """Store analysis results for report generation"""
        self.shared_memory.store_temp_data(f'mri_analysis_{user_id}', data)
    
    def _add_report_to_knowledge_base(self, user_id: str, report_result: Dict[str, Any]):
        """Add generated report to knowledge base"""
        try:
            report_summary = {
                'user_id': user_id,
                'report_id': report_result['report_id'],
                'patient_name': report_result.get('patient_name'),
                'generation_date': report_result.get('generation_date'),
                'pdf_path': report_result.get('pdf_path'),
                'classification': report_result.get('classification'),
                'timestamp': time.time()
            }
            
            self.shared_memory.store_user_data(f'medical_report_{user_id}', report_summary)
            print(f"📚 [AIMLAgent] Report added to knowledge base")
            
        except Exception as e:
            print(f"⚠️ [AIMLAgent] Failed to add report to KB: {str(e)}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information (matches existing agent pattern)"""
        return {
            'agent_name': self.name,
            'version': self.version,
            'model_name': 'AI/ML Medical Analysis System',
            'capabilities': [
                'MRI Image Processing',
                'Parkinson\'s Stage Classification',
                'Medical Report Generation',
                'Knowledge Base Integration'
            ],
            'status': 'ready',
            'integrated_with': ['CHAT', 'RAG', 'SUPERVISOR']
        }