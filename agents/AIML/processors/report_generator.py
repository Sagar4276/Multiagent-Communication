"""
Report Generation Module for AI/ML Agent
Current Date and Time (UTC): 2025-06-14 12:12:10
Current User's Login: Sagar4276

Handles PDF medical report generation with template integration.
"""

import os
import time
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from reportlab.lib.pagesizes import A4, letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.colors import black, blue, red, green, orange
from reportlab.lib import colors
from PIL import Image
import uuid

class ReportGenerator:
    """Medical report generator with PDF output"""
    
    def __init__(self, aiml_agent):
        self.aiml_agent = aiml_agent
        self.reports_dir = "generated_reports"
        self.templates_dir = "agents/AIML/templates"
        self.knowledge_base_dir = "knowledge_base/reports"
        
        # Create directories if they don't exist
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.templates_dir, exist_ok=True)
        os.makedirs(self.knowledge_base_dir, exist_ok=True)
        
        # Report styles
        self.styles = getSampleStyleSheet()
        self._create_custom_styles()
        
        print(f"📊 [ReportGenerator] Initialized medical report generator")
        print(f"📁 [ReportGenerator] Reports directory: {self.reports_dir}")
    
    def generate_medical_report(self, user_id: str, analysis_data: Dict[str, Any], patient_info: Dict[str, Any]) -> Dict[str, Any]:
        print(f"📊 [ReportGenerator] Generating medical report for {patient_info.get('name', 'Unknown')}")
        try:
            # Generate unique report ID
            report_id = self._generate_report_id()
            
            # Prepare report data
            report_data = self._prepare_report_data(user_id, analysis_data, patient_info, report_id)
            
            # Generate PDF report
            pdf_path = self._generate_pdf_report(report_data)
            
            # Save report metadata
            metadata_path = self._save_report_metadata(report_data, pdf_path)
            
            # Add to knowledge base
            self._add_to_knowledge_base(report_data, pdf_path)
            
            return {
                'status': 'success',
                'report_id': report_id,
                'pdf_path': pdf_path,
                'metadata_path': metadata_path,
                'patient_name': patient_info.get('name'),
                'generation_date': report_data['generation_date'],
                'classification': analysis_data.get('classification', {}),
                'file_size_mb': os.path.getsize(pdf_path) / (1024 * 1024),
                'message': f"Medical report generated successfully for {patient_info.get('name')}"
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Report generation failed: {str(e)}"
            }
    
    def _prepare_report_data(self, user_id: str, analysis_data: Dict[str, Any], patient_info: Dict[str, Any], report_id: str) -> Dict[str, Any]:
        """Prepare comprehensive report data"""
        
        classification = analysis_data.get('classification', {})
        features = analysis_data.get('features', {})
        
        # Get RAG recommendations if available
        rag_recommendations = self._get_rag_recommendations(classification.get('predicted_stage', 0))
        
        report_data = {
            'report_id': report_id,
            'generation_date': datetime.now().strftime('%Y-%m-%d'),
            'generation_time': datetime.now().strftime('%H:%M:%S UTC'),
            'user_id': user_id,
            
            # Patient Information
            'patient': {
                'name': patient_info.get('name', 'Unknown'),
                'age': patient_info.get('age', 'Unknown'),
                'sex': patient_info.get('sex', 'Unknown'),
                'contact': patient_info.get('contact', 'Unknown'),
                'email': patient_info.get('email', 'Not provided'),
                'doctor': patient_info.get('doctor', 'Not specified'),
                'medical_id': patient_info.get('medical_id', 'Not provided')
            },
            
            # Analysis Results
            'analysis': {
                'mri_analysis_date': datetime.fromtimestamp(analysis_data.get('timestamp', time.time())).strftime('%Y-%m-%d %H:%M:%S'),
                'image_path': analysis_data.get('image_path', 'Unknown'),
                'predicted_stage': classification.get('predicted_stage', 0),
                'stage_name': classification.get('stage_name', 'Unknown'),
                'stage_description': classification.get('stage_description', 'No description available'),
                'confidence': classification.get('confidence', 0.0),
                'confidence_percentage': classification.get('confidence_percentage', '0%'),
                'severity': classification.get('severity', 'Unknown'),
                'motor_symptoms': classification.get('motor_symptoms', 'Not specified'),
                'prognosis': classification.get('prognosis', 'Not available')
            },
            
            # Detailed Analysis
            'detailed_analysis': classification.get('detailed_analysis', {}),
            
            # Recommendations
            'recommendations': classification.get('recommendations', {}),
            'rag_recommendations': rag_recommendations,
            
            # System Information
            'system_info': {
                'ai_system': 'Enhanced Multi-Agent Medical Analysis System',
                'version': '2.0.0',
                'classification_model': 'Parkinson\'s Stage Classifier v1.0',
                'analysis_type': 'MRI-based Parkinson\'s Disease Stage Classification',
                'disclaimer': 'This report is generated by AI analysis and should be reviewed by qualified medical professionals.'
            }
        }
        
        return report_data
    
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> str:
        """Generate PDF report using ReportLab"""
        
        # Create filename
        patient_name = report_data['patient']['name'].replace(' ', '_')
        filename = f"Medical_Report_{patient_name}_{report_data['report_id']}.pdf"
        pdf_path = os.path.join(self.reports_dir, filename)
        
        # Create PDF document
        doc = SimpleDocTemplate(pdf_path, pagesize=A4, topMargin=0.5*inch)
        story = []
        
        # Build report content
        story.extend(self._build_header(report_data))
        story.extend(self._build_patient_info(report_data))
        story.extend(self._build_analysis_results(report_data))
        story.extend(self._build_detailed_findings(report_data))
        story.extend(self._build_recommendations(report_data))
        story.extend(self._build_footer(report_data))
        
        # Generate PDF
        doc.build(story)
        
        print(f"📄 [ReportGenerator] PDF generated: {pdf_path}")
        return pdf_path
    
    def _build_header(self, report_data: Dict[str, Any]) -> List[Any]:
        """Build report header"""
        
        content = []
        
        # Title
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=self.styles['Title'],
            fontSize=18,
            textColor=blue,
            spaceAfter=20,
            alignment=1  # Center
        )
        
        content.append(Paragraph("MEDICAL ANALYSIS REPORT", title_style))
        content.append(Paragraph("Parkinson's Disease Stage Classification", self.styles['Heading2']))
        content.append(Spacer(1, 20))
        
        # Report info table
        report_info = [
            ['Report ID:', report_data['report_id']],
            ['Generation Date:', f"{report_data['generation_date']} {report_data['generation_time']}"],
            ['Analysis System:', report_data['system_info']['ai_system']],
            ['Model Version:', report_data['system_info']['classification_model']]
        ]
        
        table = Table(report_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _build_patient_info(self, report_data: Dict[str, Any]) -> List[Any]:
        """Build patient information section"""
        
        content = []
        
        # Section header
        content.append(Paragraph("PATIENT INFORMATION", self.styles['Heading2']))
        content.append(Spacer(1, 10))
        
        # Patient data table
        patient = report_data['patient']
        patient_info = [
            ['Name:', patient['name']],
            ['Age:', str(patient['age'])],
            ['Sex:', patient['sex']],
            ['Contact:', patient['contact']],
            ['Email:', patient['email']],
            ['Referring Doctor:', patient['doctor']],
            ['Medical ID:', patient['medical_id']]
        ]
        
        table = Table(patient_info, colWidths=[2*inch, 4*inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightblue),
            ('VALIGN', (0, 0), (-1, -1), 'TOP')
        ]))
        
        content.append(table)
        content.append(Spacer(1, 20))
        
        return content
    
    def _build_analysis_results(self, report_data: Dict[str, Any]) -> List[Any]:
        """Build analysis results section"""
        
        content = []
        analysis = report_data['analysis']
        
        # Section header
        content.append(Paragraph("ANALYSIS RESULTS", self.styles['Heading2']))
        content.append(Spacer(1, 10))
        
        # MRI Analysis Info
        content.append(Paragraph("MRI Analysis Information", self.styles['Heading3']))
        mri_info = [
            ['Analysis Date:', analysis['mri_analysis_date']],
            ['Image Source:', os.path.basename(analysis['image_path'])],
            ['Analysis Type:', report_data['system_info']['analysis_type']]
        ]
        
        table1 = Table(mri_info, colWidths=[2*inch, 4*inch])
        table1.setStyle(self._get_standard_table_style())
        content.append(table1)
        content.append(Spacer(1, 15))
        
        # Classification Results
        content.append(Paragraph("Classification Results", self.styles['Heading3']))
        
        # Determine color based on stage
        stage = analysis['predicted_stage']
        if stage == 0:
            stage_color = green
        elif stage <= 2:
            stage_color = orange
        else:
            stage_color = red
        
        # Create colored stage result
        stage_text = f"<font color='#{stage_color.hexval()}'><b>{analysis['stage_name']}</b></font>"
        
        classification_info = [
            ['Predicted Stage:', Paragraph(stage_text, self.styles['Normal'])],
            ['Confidence Level:', f"{analysis['confidence_percentage']} ({analysis['confidence']:.3f})"],
            ['Severity:', analysis['severity']],
            ['Description:', analysis['stage_description']]
        ]
        
        table2 = Table(classification_info, colWidths=[2*inch, 4*inch])
        table2.setStyle(self._get_standard_table_style())
        content.append(table2)
        content.append(Spacer(1, 15))
        
        # Motor Symptoms and Prognosis
        content.append(Paragraph("Clinical Information", self.styles['Heading3']))
        clinical_info = [
            ['Motor Symptoms:', analysis['motor_symptoms']],
            ['Prognosis:', analysis['prognosis']]
        ]
        
        table3 = Table(clinical_info, colWidths=[2*inch, 4*inch])
        table3.setStyle(self._get_standard_table_style())
        content.append(table3)
        content.append(Spacer(1, 20))
        
        return content
    
    def _build_detailed_findings(self, report_data: Dict[str, Any]) -> List[Any]:
        """Build detailed findings section"""
        
        content = []
        detailed = report_data.get('detailed_analysis', {})
        
        # Section header
        content.append(Paragraph("DETAILED FINDINGS", self.styles['Heading2']))
        content.append(Spacer(1, 10))
        
        # Feature Analysis
        feature_analysis = detailed.get('feature_analysis', {})
        if feature_analysis:
            content.append(Paragraph("Image Feature Analysis", self.styles['Heading3']))
            
            for feature, description in feature_analysis.items():
                content.append(Paragraph(f"<b>{feature.title()}:</b> {description}", self.styles['Normal']))
                content.append(Spacer(1, 5))
        
        content.append(Spacer(1, 10))
        
        # Risk Factors
        risk_factors = detailed.get('risk_factors', [])
        if risk_factors:
            content.append(Paragraph("Identified Risk Factors", self.styles['Heading3']))
            
            for i, factor in enumerate(risk_factors, 1):
                content.append(Paragraph(f"{i}. {factor}", self.styles['Normal']))
                content.append(Spacer(1, 3))
        
        content.append(Spacer(1, 10))
        
        # Confidence Assessment
        confidence_assessment = detailed.get('confidence_assessment', '')
        if confidence_assessment:
            content.append(Paragraph("Confidence Assessment", self.styles['Heading3']))
            content.append(Paragraph(confidence_assessment, self.styles['Normal']))
        
        content.append(Spacer(1, 15))
        
        # Alternative Possibilities
        alternatives = detailed.get('alternative_possibilities', [])
        if alternatives:
            content.append(Paragraph("Alternative Diagnostic Considerations", self.styles['Heading3']))
            
            for alt in alternatives:
                content.append(Paragraph(f"• {alt}", self.styles['Normal']))
                content.append(Spacer(1, 3))
        
        content.append(Spacer(1, 20))
        
        return content
    
    def _build_recommendations(self, report_data: Dict[str, Any]) -> List[Any]:
        """Build recommendations section"""
        
        content = []
        recommendations = report_data.get('recommendations', {})
        rag_recommendations = report_data.get('rag_recommendations', {})
        
        # Section header
        content.append(Paragraph("RECOMMENDATIONS", self.styles['Heading2']))
        content.append(Spacer(1, 10))
        
        # AI/ML Recommendations
        content.append(Paragraph("AI Analysis Recommendations", self.styles['Heading3']))
        
        for category, items in recommendations.items():
            if items:
                category_title = category.replace('_', ' ').title()
                content.append(Paragraph(f"<b>{category_title}:</b>", self.styles['Normal']))
                
                for item in items:
                    content.append(Paragraph(f"• {item}", self.styles['Normal']))
                    content.append(Spacer(1, 3))
                
                content.append(Spacer(1, 8))
        
        # RAG Knowledge Base Recommendations
        if rag_recommendations.get('treatment_info'):
            content.append(Paragraph("Knowledge Base Treatment Information", self.styles['Heading3']))
            content.append(Paragraph(rag_recommendations['treatment_info'], self.styles['Normal']))
            content.append(Spacer(1, 10))
        
        if rag_recommendations.get('prevention_info'):
            content.append(Paragraph("Prevention Guidelines", self.styles['Heading3']))
            content.append(Paragraph(rag_recommendations['prevention_info'], self.styles['Normal']))
            content.append(Spacer(1, 15))
        
        return content
    
    def _build_footer(self, report_data: Dict[str, Any]) -> List[Any]:
        """Build report footer"""
        
        content = []
        
        # Disclaimer
        content.append(Paragraph("IMPORTANT DISCLAIMER", self.styles['Heading3']))
        disclaimer_text = report_data['system_info']['disclaimer']
        
        disclaimer_style = ParagraphStyle(
            'Disclaimer',
            parent=self.styles['Normal'],
            fontSize=10,
            textColor=red,
            borderWidth=1,
            borderColor=red,
            borderPadding=10
        )
        
        content.append(Paragraph(disclaimer_text, disclaimer_style))
        content.append(Spacer(1, 15))
        
        # Report Generation Info
        content.append(Paragraph("Report Generation Information", self.styles['Heading4']))
        
        generation_info = [
            ['Generated by:', report_data['system_info']['ai_system']],
            ['System Version:', report_data['system_info']['version']],
            ['Generation Date:', f"{report_data['generation_date']} {report_data['generation_time']}"],
            ['Report ID:', report_data['report_id']]
        ]
        
        table = Table(generation_info, colWidths=[2*inch, 4*inch])
        table.setStyle(self._get_standard_table_style())
        content.append(table)
        
        return content
    
    def _get_rag_recommendations(self, stage: int) -> Dict[str, str]:
        """Get recommendations from RAG knowledge base"""
        
        try:
            # Get RAG agent from the AI/ML agent's connections
            if hasattr(self.aiml_agent, 'rag_agent') and self.aiml_agent.rag_agent:
                
                # Query for treatment information
                treatment_query = f"treatment options for parkinson's stage {stage}"
                treatment_info = self.aiml_agent.rag_agent.get_relevant_info(treatment_query)
                
                # Query for prevention information
                prevention_query = f"prevention methods parkinson's disease stage {stage}"
                prevention_info = self.aiml_agent.rag_agent.get_relevant_info(prevention_query)
                
                return {
                    'treatment_info': treatment_info[:500] + "..." if len(treatment_info) > 500 else treatment_info,
                    'prevention_info': prevention_info[:500] + "..." if len(prevention_info) > 500 else prevention_info
                }
            
            else:
                return {
                    'treatment_info': "RAG knowledge base not available for detailed treatment recommendations.",
                    'prevention_info': "Please consult with healthcare provider for personalized prevention strategies."
                }
                
        except Exception as e:
            print(f"⚠️ [ReportGenerator] RAG recommendations failed: {str(e)}")
            return {
                'treatment_info': "Unable to retrieve treatment recommendations from knowledge base.",
                'prevention_info': "Consult healthcare provider for prevention guidance."
            }
    
    def _save_report_metadata(self, report_data: Dict[str, Any], pdf_path: str) -> str:
        """Save report metadata as JSON"""
        
        metadata_filename = f"metadata_{report_data['report_id']}.json"
        metadata_path = os.path.join(self.reports_dir, metadata_filename)
        
        metadata = {
            'report_id': report_data['report_id'],
            'patient_name': report_data['patient']['name'],
            'generation_date': report_data['generation_date'],
            'pdf_path': pdf_path,
            'classification_stage': report_data['analysis']['predicted_stage'],
            'confidence': report_data['analysis']['confidence'],
            'user_id': report_data['user_id']
        }
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata_path
    
    def _add_to_knowledge_base(self, report_data: Dict[str, Any], pdf_path: str):
        """Add report to knowledge base for future queries"""
        
        try:
            # Create knowledge base entry
            kb_entry = {
                'type': 'medical_report',
                'patient_name': report_data['patient']['name'],
                'report_id': report_data['report_id'],
                'generation_date': report_data['generation_date'],
                'stage': report_data['analysis']['predicted_stage'],
                'stage_name': report_data['analysis']['stage_name'],
                'confidence': report_data['analysis']['confidence'],
                'pdf_path': pdf_path,
                'searchable_content': self._create_searchable_content(report_data)
            }
            
            # Save to knowledge base directory
            kb_filename = f"report_{report_data['report_id']}.json"
            kb_path = os.path.join(self.knowledge_base_dir, kb_filename)
            
            with open(kb_path, 'w') as f:
                json.dump(kb_entry, f, indent=2)
            
            print(f"📚 [ReportGenerator] Added report to knowledge base: {kb_path}")
            
        except Exception as e:
            print(f"⚠️ [ReportGenerator] Failed to add to knowledge base: {str(e)}")
    
    def _create_searchable_content(self, report_data: Dict[str, Any]) -> str:
        """Create searchable text content from report"""
        
        content_parts = [
            f"Patient: {report_data['patient']['name']}",
            f"Stage: {report_data['analysis']['stage_name']}",
            f"Confidence: {report_data['analysis']['confidence_percentage']}",
            f"Description: {report_data['analysis']['stage_description']}",
            f"Motor symptoms: {report_data['analysis']['motor_symptoms']}",
            f"Prognosis: {report_data['analysis']['prognosis']}"
        ]
        
        return " | ".join(content_parts)
    
    def search_previous_reports(self, patient_name: str) -> List[Dict[str, Any]]:
        """Search for previous reports for a patient"""
        
        try:
            reports = []
            
            # Search in knowledge base directory
            for filename in os.listdir(self.knowledge_base_dir):
                if filename.startswith('report_') and filename.endswith('.json'):
                    filepath = os.path.join(self.knowledge_base_dir, filename)
                    
                    with open(filepath, 'r') as f:
                        report_data = json.load(f)
                    
                    if report_data.get('patient_name', '').lower() == patient_name.lower():
                        reports.append({
                            'report_id': report_data['report_id'],
                            'date': report_data['generation_date'],
                            'stage': report_data['stage_name'],
                            'confidence': f"{report_data['confidence']:.1%}",
                            'filename': os.path.basename(report_data['pdf_path'])
                        })
            
            # Sort by date (newest first)
            reports.sort(key=lambda x: x['date'], reverse=True)
            
            return reports
            
        except Exception as e:
            print(f"❌ [ReportGenerator] Search failed: {str(e)}")
            return []
    
    def _generate_report_id(self) -> str:
        """Generate unique report ID"""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_id = str(uuid.uuid4())[:8]
        return f"RPT_{timestamp}_{unique_id}"
    
    def _create_custom_styles(self):
        """Create custom paragraph styles"""
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading1',
            parent=self.styles['Heading1'],
            fontSize=16,
            textColor=blue,
            spaceAfter=15
        ))
        
        self.styles.add(ParagraphStyle(
            name='CustomHeading2',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=blue,
            spaceAfter=12
        ))
    
    def _get_standard_table_style(self) -> TableStyle:
        """Get standard table style"""
        
        return TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey),
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('WORDWRAP', (0, 0), (-1, -1), True)
        ])
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information"""
        return {
            'name': 'ReportGenerator',
            'version': '1.0.0',
            'output_format': 'PDF',
            'template_support': True,
            'knowledge_base_integration': True,
            'reports_directory': self.reports_dir
        }