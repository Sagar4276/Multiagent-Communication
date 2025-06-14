"""
Enhanced Streamlit Application with Image Upload
Current Date and Time (UTC): 2025-06-14 12:12:10
Current User's Login: Sagar4276

Streamlit interface supporting:
- MRI image upload and analysis
- General medical image upload
- Patient data collection
- Report generation and download
"""

import streamlit as st
import os
import time
import tempfile
from PIL import Image
import base64
from typing import Dict, Any
import json

# Import your existing agents
from shared_memory.simple_memory import SimpleSharedMemory
from agents.CHAT.chat_agent import ChatAgent
from agents.AIML.aiml_agent import AIMLAgent
from agents.SUPERVISOR.supervisor_agent import EnhancedSupervisorAgent

class EnhancedStreamlitApp:
    """Enhanced Streamlit application with image upload capabilities"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.initialize_agents()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Enhanced Medical Analysis System",
            page_icon="🧠",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 2rem;
        }
        .upload-section {
            background: #f0f2f6;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .analysis-section {
            background: #e8f4fd;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1rem;
        }
        .patient-form {
            background: #f9f9f9;
            padding: 1.5rem;
            border-radius: 10px;
            border: 2px solid #ddd;
        }
        .success-box {
            background: #d4edda;
            color: #155724;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #c3e6cb;
        }
        .error-box {
            background: #f8d7da;
            color: #721c24;
            padding: 1rem;
            border-radius: 5px;
            border: 1px solid #f5c6cb;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state"""
        
        if 'initialized' not in st.session_state:
            st.session_state.initialized = False
            st.session_state.agents_loaded = False
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.session_state.analysis_results = None
            st.session_state.patient_data = {}
            st.session_state.report_generated = False
            st.session_state.current_user = "Sagar4276"
    
    def initialize_agents(self):
        """Initialize all agents"""
        
        if not st.session_state.agents_loaded:
            try:
                with st.spinner("🚀 Initializing Enhanced Medical Analysis System..."):
                    # Initialize shared memory
                    self.shared_memory = SimpleSharedMemory()
                    
                    # Initialize agents
                    self.chat_agent = ChatAgent(self.shared_memory)
                    self.aiml_agent = AIMLAgent(self.shared_memory)
                    self.supervisor = EnhancedSupervisorAgent()
                    
                    # Connect agents
                    self.aiml_agent.set_agent_connections(
                        chat_agent=self.chat_agent,
                        rag_agent=getattr(self.chat_agent, 'rag_agent', None)
                    )
                    
                    st.session_state.agents_loaded = True
                    st.success("✅ All agents initialized successfully!")
                    
            except Exception as e:
                st.error(f"❌ Failed to initialize agents: {str(e)}")
                st.stop()
    
    def run(self):
        """Main application runner"""
        
        # Header
        st.markdown("""
        <div class="main-header">
            <h1>🧠 Enhanced Medical Analysis System</h1>
            <p>AI-Powered Parkinson's Disease Analysis with MRI Processing</p>
            <p><strong>Current User:</strong> Sagar4276 | <strong>System Version:</strong> 2.0.0</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Sidebar navigation
        self.render_sidebar()
        
        # Main content based on selected page
        page = st.session_state.get('current_page', 'image_analysis')
        
        if page == 'image_analysis':
            self.render_image_analysis_page()
        elif page == 'chat':
            self.render_chat_page()
        elif page == 'reports':
            self.render_reports_page()
        elif page == 'system_status':
            self.render_system_status_page()
    
    def render_sidebar(self):
        """Render sidebar navigation"""
        
        st.sidebar.markdown("## 🧭 Navigation")
        
        pages = {
            'image_analysis': '🖼️ Image Analysis',
            'chat': '💬 Chat & Knowledge',
            'reports': '📊 Reports & History',
            'system_status': '⚙️ System Status'
        }
        
        for page_key, page_name in pages.items():
            if st.sidebar.button(page_name, key=f"nav_{page_key}"):
                st.session_state.current_page = page_key
                st.rerun()
        
        st.sidebar.markdown("---")
        
        # System info
        st.sidebar.markdown("### 📊 System Info")
        st.sidebar.info(f"""
        **User:** {st.session_state.current_user}
        **Agents Status:** {'✅ Active' if st.session_state.agents_loaded else '❌ Loading'}
        **Session:** {len(st.session_state.chat_history)} messages
        """)
        
        # Quick actions
        st.sidebar.markdown("### ⚡ Quick Actions")
        if st.sidebar.button("🔄 Reset Session"):
            self.reset_session()
        
        if st.sidebar.button("📥 Download Sample"):
            self.download_sample_data()
    
    def render_image_analysis_page(self):
        """Render image analysis page"""
        
        st.markdown("## 🖼️ Medical Image Analysis")
        
        # Image upload section
        with st.container():
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📤 Upload Medical Image")
                
                # Image type selection
                image_type = st.selectbox(
                    "Select Image Type:",
                    ["MRI Scan", "Other Medical Image"],
                    help="Choose MRI for Parkinson's analysis or Other for general analysis"
                )
                
                # File uploader
                uploaded_file = st.file_uploader(
                    "Choose an image file",
                    type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'dcm'],
                    help="Supported formats: JPG, PNG, BMP, TIFF, DICOM"
                )
                
                if uploaded_file is not None:
                    # Save uploaded file
                    image_path = self.save_uploaded_file(uploaded_file)
                    st.session_state.uploaded_image = image_path
                    st.session_state.image_type = "mri" if "MRI" in image_type else "general"
                    
                    st.success(f"✅ Image uploaded: {uploaded_file.name}")
            
            with col2:
                if st.session_state.uploaded_image:
                    st.markdown("### 🖼️ Uploaded Image")
                    
                    # Display image
                    image = Image.open(st.session_state.uploaded_image)
                    st.image(image, caption="Uploaded Medical Image", use_column_width=True)
                    
                    # Image info
                    st.info(f"""
                    **Type:** {st.session_state.get('image_type', 'unknown').upper()}
                    **Size:** {image.size}
                    **Mode:** {image.mode}
                    """)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis section
        if st.session_state.uploaded_image:
            with st.container():
                st.markdown('<div class="analysis-section">', unsafe_allow_html=True)
                st.markdown("### 🧠 AI Analysis")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🚀 Start Analysis", type="primary"):
                        self.perform_image_analysis()
                
                with col2:
                    if st.session_state.analysis_results:
                        if st.button("📋 Collect Patient Data"):
                            st.session_state.show_patient_form = True
                
                with col3:
                    if st.session_state.get('patient_data_collected'):
                        if st.button("📄 Generate Report"):
                            self.generate_medical_report()
                
                st.markdown('</div>', unsafe_allow_html=True)
        
        # Display analysis results
        if st.session_state.analysis_results:
            self.display_analysis_results()
        
        # Patient data collection form
        if st.session_state.get('show_patient_form'):
            self.render_patient_form()
        
        # Report generation status
        if st.session_state.get('report_generated'):
            self.display_report_status()
    
    def perform_image_analysis(self):
        """Perform image analysis using AI/ML agent"""
        
        try:
            with st.spinner("🧠 Analyzing image... This may take a moment..."):
                
                # Prepare context for AI/ML agent
                context = {
                    'image_path': st.session_state.uploaded_image,
                    'image_type': st.session_state.image_type
                }
                
                # Process through AI/ML agent
                result = self.aiml_agent.process_message(
                    st.session_state.current_user,
                    f"Analyze {st.session_state.image_type} image",
                    context
                )
                
                st.session_state.analysis_results = result
                st.success("✅ Analysis completed!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
    
    def display_analysis_results(self):
        """Display analysis results"""
        
        st.markdown("### 📊 Analysis Results")
        
        # Parse results from AI/ML agent response
        results_text = st.session_state.analysis_results
        
        if "STEP 2: AI CLASSIFICATION" in results_text:
            # Extract classification information
            lines = results_text.split('\n')
            
            stage_info = {}
            for line in lines:
                if "Predicted Stage:" in line:
                    stage_info['stage'] = line.split(':')[1].strip()
                elif "Confidence:" in line:
                    stage_info['confidence'] = line.split(':')[1].strip()
                elif "Description:" in line:
                    stage_info['description'] = line.split(':')[1].strip()
            
            # Display in organized format
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🎯 Classification Result")
                if stage_info.get('stage'):
                    if 'Stage 0' in stage_info['stage'] or 'No Parkinson' in stage_info['stage']:
                        st.success(f"**Stage:** {stage_info['stage']}")
                    elif 'Stage 1' in stage_info['stage'] or 'Stage 2' in stage_info['stage']:
                        st.warning(f"**Stage:** {stage_info['stage']}")
                    else:
                        st.error(f"**Stage:** {stage_info['stage']}")
                
                if stage_info.get('confidence'):
                    st.metric("Confidence", stage_info['confidence'])
            
            with col2:
                st.markdown("#### 📋 Clinical Information")
                if stage_info.get('description'):
                    st.info(stage_info['description'])
        
        # Full results in expandable section
        with st.expander("📄 Full Analysis Report"):
            st.text(results_text)
    
    def render_patient_form(self):
        """Render patient data collection form"""
        
        st.markdown('<div class="patient-form">', unsafe_allow_html=True)
        st.markdown("### 👤 Patient Information")
        st.markdown("Please provide patient details for report generation:")
        
        with st.form("patient_data_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Patient Name *", placeholder="John Doe")
                age = st.number_input("Age *", min_value=1, max_value=120, value=65)
                sex = st.selectbox("Sex *", ["Male", "Female", "Other"])
                contact = st.text_input("Contact Number *", placeholder="+1-555-0123")
            
            with col2:
                email = st.text_input("Email", placeholder="john.doe@email.com")
                doctor = st.text_input("Referring Doctor", placeholder="Dr. Smith")
                medical_id = st.text_input("Medical ID", placeholder="MED-12345")
                notes = st.text_area("Additional Notes", placeholder="Any additional information...")
            
            submitted = st.form_submit_button("💾 Save Patient Data", type="primary")
            
            if submitted:
                if name and age and sex and contact:
                    st.session_state.patient_data = {
                        'name': name,
                        'age': age,
                        'sex': sex,
                        'contact': contact,
                        'email': email or 'Not provided',
                        'doctor': doctor or 'Not specified',
                        'medical_id': medical_id or 'Not provided',
                        'notes': notes or 'None'
                    }
                    st.session_state.patient_data_collected = True
                    st.session_state.show_patient_form = False
                    st.success("✅ Patient data saved successfully!")
                    st.rerun()
                else:
                    st.error("❌ Please fill in all required fields (*)")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def generate_medical_report(self):
        """Generate medical report"""
        
        try:
            with st.spinner("📄 Generating medical report..."):
                
                # Prepare context for report generation
                context = {
                    'patient_info': st.session_state.patient_data,
                    'generate_report': True
                }
                
                # Generate report through AI/ML agent
                report_result = self.aiml_agent.process_message(
                    st.session_state.current_user,
                    "generate report",
                    context
                )
                
                st.session_state.report_result = report_result
                st.session_state.report_generated = True
                st.success("✅ Medical report generated successfully!")
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Report generation failed: {str(e)}")
    
    def display_report_status(self):
        """Display report generation status"""
        
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.markdown("### ✅ Report Generated Successfully!")
        
        result_text = st.session_state.get('report_result', '')
        
        # Extract report information
        if 'Report ID:' in result_text:
            lines = result_text.split('\n')
            for line in lines:
                if 'Report ID:' in line:
                    st.markdown(f"**{line}**")
                elif 'Patient:' in line:
                    st.markdown(f"**{line}**")
                elif 'Generated:' in line:
                    st.markdown(f"**{line}**")
                elif 'File:' in line:
                    st.markdown(f"**{line}**")
        
        # Download button (simulated)
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download PDF"):
                st.info("PDF download functionality - integration with actual file path needed")
        
        with col2:
            if st.button("📧 Email Report"):
                st.info("Email functionality - SMTP integration needed")
        
        with col3:
            if st.button("🔄 Generate New Report"):
                self.reset_analysis()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    def render_chat_page(self):
        """Render chat interface"""
        
        st.markdown("## 💬 Chat & Knowledge Assistant")
        
        # Chat history
        st.markdown("### 📜 Conversation History")
        
        chat_container = st.container()
        
        with chat_container:
            for i, message in enumerate(st.session_state.chat_history):
                if message['role'] == 'user':
                    st.markdown(f"**👤 You:** {message['content']}")
                else:
                    st.markdown(f"**🤖 Assistant:** {message['content']}")
                st.markdown("---")
        
        # Chat input
        st.markdown("### 💬 Send Message")
        
        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area(
                "Type your message:",
                placeholder="Ask about Parkinson's disease, treatments, or system status...",
                height=100
            )
            
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                send_button = st.form_submit_button("📤 Send", type="primary")
            
            with col2:
                clear_button = st.form_submit_button("🗑️ Clear Chat")
            
            if send_button and user_input:
                self.process_chat_message(user_input)
            
            if clear_button:
                st.session_state.chat_history = []
                st.rerun()
    
    def process_chat_message(self, message: str):
        """Process chat message through supervisor"""
        
        try:
            with st.spinner("🤖 Processing your message..."):
                
                # Add user message to history
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': message,
                    'timestamp': time.time()
                })
                
                # Process through supervisor
                response = self.supervisor.handle_user_input(
                    st.session_state.current_user,
                    message
                )
                
                # Add response to history
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response,
                    'timestamp': time.time()
                })
                
                st.rerun()
                
        except Exception as e:
            st.error(f"❌ Failed to process message: {str(e)}")
    
    def render_reports_page(self):
        """Render reports and history page"""
        
        st.markdown("## 📊 Reports & Patient History")
        
        # Search section
        st.markdown("### 🔍 Search Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            search_name = st.text_input("Patient Name", placeholder="Enter patient name...")
        
        with col2:
            if st.button("🔍 Search Reports"):
                if search_name:
                    self.search_patient_reports(search_name)
        
        # Display search results
        if st.session_state.get('search_results'):
            self.display_search_results()
        
        # Recent reports section
        st.markdown("### 📋 Recent Reports")
        self.display_recent_reports()
    
    def render_system_status_page(self):
        """Render system status page"""
        
        st.markdown("## ⚙️ System Status & Diagnostics")
        
        # System overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("System Status", "✅ Operational")
        
        with col2:
            st.metric("Active Agents", "4/4")
        
        with col3:
            st.metric("Session Messages", len(st.session_state.chat_history))
        
        # Detailed status
        if st.button("🔄 Refresh Status"):
            self.get_system_status()
        
        # Display system information
        if st.session_state.get('system_status'):
            st.markdown("### 📊 Detailed System Information")
            st.text(st.session_state.system_status)
    
    def save_uploaded_file(self, uploaded_file) -> str:
        """Save uploaded file to temporary location"""
        
        # Create uploads directory
        uploads_dir = "uploads"
        os.makedirs(uploads_dir, exist_ok=True)
        
        # Save file
        file_path = os.path.join(uploads_dir, uploaded_file.name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    
    def search_patient_reports(self, patient_name: str):
        """Search for patient reports"""
        
        try:
            # Use AI/ML agent to search reports
            response = self.aiml_agent.process_message(
                st.session_state.current_user,
                f"check reports {patient_name}"
            )
            
            st.session_state.search_results = response
            
        except Exception as e:
            st.error(f"❌ Search failed: {str(e)}")
    
    def display_search_results(self):
        """Display search results"""
        
        st.markdown("#### 🔍 Search Results")
        st.text(st.session_state.search_results)
    
    def display_recent_reports(self):
        """Display recent reports"""
        
        st.markdown("Recent reports functionality - integration with report storage needed")
    
    def get_system_status(self):
        """Get system status"""
        
        try:
            status = self.supervisor.handle_user_input(
                st.session_state.current_user,
                "system status"
            )
            
            st.session_state.system_status = status
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Failed to get status: {str(e)}")
    
    def reset_session(self):
        """Reset session state"""
        
        st.session_state.chat_history = []
        st.session_state.uploaded_image = None
        st.session_state.analysis_results = None
        st.session_state.patient_data = {}
        st.session_state.report_generated = False
        st.session_state.show_patient_form = False
        st.success("🔄 Session reset successfully!")
        st.rerun()
    
    def reset_analysis(self):
        """Reset analysis state"""
        
        st.session_state.uploaded_image = None
        st.session_state.analysis_results = None
        st.session_state.patient_data = {}
        st.session_state.report_generated = False
        st.session_state.show_patient_form = False
        st.rerun()
    
    def download_sample_data(self):
        """Download sample data"""
        
        st.info("Sample data download - integration with sample files needed")

# Main execution
if __name__ == "__main__":
    app = EnhancedStreamlitApp()
    app.run()