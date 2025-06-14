"""
Enhanced Multi-Agent Chat System with AI/ML Integration
Current Date and Time (UTC): 2025-06-14 20:16:52
Current User's Login: Sagar4276

Added Features:
- AI/ML Agent for image analysis
- MRI upload and processing
- Medical report generation
- Parkinson's stage classification
"""

import time
import os
import argparse
import sys
from datetime import datetime, timezone
from agents.SUPERVISOR.supervisor_agent import EnhancedSupervisorAgent, create_medical_config

class EnhancedChatApp:
    def __init__(self):
        print("🤖 Initializing Enhanced Multi-Agent Medical System...")
        current_user = input("Enter your username: ").strip() or "Sagar4276"
        
        # Dynamic user and time - FIXED deprecation
        self.current_user = current_user if current_user else "Sagar4276"
        self.current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
        
        print(f"👤 User: {self.current_user}")
        print(f"🕐 Current Time (UTC): {self.current_time}")
        
        # Choose AI model type
        self.model_choice = self._choose_model_type()
        
        # Initialize supervisor with chosen model (with AI/ML integration)
        self.supervisor = self._initialize_supervisor()
        
        # Show system status
        self._display_initialization_status()
    
    def _choose_model_type(self):
        """Let user choose between local SLM and API models"""
        print("\n" + "="*60)
        print("🎯 CHOOSE YOUR AI MODEL")
        print("="*60)
        print("1. 🤖 Local SLM + AI/ML Medical")
        print("   📦 Model: DistilGPT2 (Conversational)")
        print("   🏥 Medical: AI/ML Image Analysis & Reports")
        print("   💰 Cost: Free")
        print("   ⚡ Speed: Fast")
        print("   🔧 Runs: On your computer")
        print()
        print("2. 🌐 Gemini 2.0 Flash (API) + AI/ML Medical")
        print("   📦 Model: Google's latest Gemini")
        print("   🏥 Medical: AI/ML Image Analysis & Reports")
        print("   💰 Cost: Pay per use") 
        print("   ⚡ Speed: Very fast")
        print("   🔧 Runs: Google's servers")
        print("   🔑 Requires: API key")
        print("="*60)
        
        while True:
            choice = input("Select option (1 or 2): ").strip()
            
            if choice == "1":
                print("✅ Selected: Local SLM + AI/ML Medical System")
                return "local"
            elif choice == "2":
                print("✅ Selected: Gemini 2.0 Flash API + AI/ML Medical System")
                return "api"
            else:
                print("❌ Please enter 1 or 2")
    
    def _initialize_supervisor(self):
        """🆕 Initialize supervisor with AI/ML integration"""
        print("🔧 Initializing Enhanced Supervisor with AI/ML Medical System...")
        
        try:
            # Use medical configuration for AI/ML support
            supervisor = EnhancedSupervisorAgent(create_medical_config())
            
            # Update supervisor with current user and time
            supervisor.update_current_time(self.current_time)
            supervisor.current_user = self.current_user
            
            return supervisor
            
        except Exception as e:
            print(f"❌ Supervisor initialization failed: {str(e)}")
            print("🔄 Falling back to basic supervisor...")
            
            # Fallback to basic supervisor
            supervisor = EnhancedSupervisorAgent()
            supervisor.update_current_time(self.current_time)
            supervisor.current_user = self.current_user
            return supervisor
    
    def _display_initialization_status(self):
        """🆕 Display comprehensive initialization status"""
        try:
            # Get model info
            model_info = self.supervisor.chat_agent.get_model_info()
            
            # Get AI/ML status
            aiml_status = self.supervisor.get_aiml_status()
            
            print(f"\n{'='*70}")
            print(f"🎉 ENHANCED MEDICAL SYSTEM INITIALIZED")
            print(f"{'='*70}")
            print(f"🧠 Chat Agent: ✅ {model_info.get('model_name', 'Ready')}")
            print(f"📚 RAG Agent: ✅ Knowledge Base Loaded")
            print(f"🎯 Supervisor: ✅ Enhanced Routing Active")
            print(f"🤖 AI/ML Agent: {'✅ Medical Analysis Ready' if aiml_status['available'] else '❌ Not Available'}")
            
            if aiml_status['available']:
                capabilities = aiml_status.get('capabilities', [])
                print(f"🏥 Medical Features: {len(capabilities)} capabilities")
                print(f"   • MRI Image Analysis")
                print(f"   • Medical Report Generation") 
                print(f"   • Patient Data Management")
                print(f"   • Stage Classification")
            else:
                print(f"⚠️ Medical AI: {aiml_status.get('reason', 'Not loaded')}")
                print(f"💡 Image analysis and reports disabled")
            
            print(f"{'='*70}")
            
        except Exception as e:
            print(f"✅ System ready! (Note: Status display error: {str(e)[:30]}...)")
    
    def run_chat(self):
        """🆕 Run interactive chat with enhanced AI/ML capabilities"""
        print("\n" + "="*70)
        print("🏥 ENHANCED MEDICAL MULTI-AGENT CHAT SYSTEM")
        print("="*70)
        print("💬 Chat Commands:")
        print("- Type your message to chat")
        print("- Type 'history' to see conversation history")
        print("- Type 'status' to see system status")
        print("- Type 'time' to see current time")
        print()
        print("🤖 AI/ML Medical Commands:")
        print("- Type 'upload mri [path]' to analyze MRI image")
        print("- Type 'upload image [path]' to analyze general image")
        print("- Type 'generate report' for medical report")
        print("- Type 'check reports [name]' to search patient reports")
        print("- Type 'patient data' to collect patient information")
        print()
        print("🔧 System Commands:")
        print("- Type 'web' to launch Streamlit interface")
        print("- Type 'aiml status' to check AI/ML system")
        print("- Type 'exit' to quit")
        print("="*70)
        
        while True:
            try:
                # FIXED deprecation
                current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                
                user_input = input(f"\n💬 [{self.current_user}] [{current_time}]: ").strip()
                
                if user_input.lower() == 'exit':
                    print("👋 Goodbye!")
                    break
                
                elif user_input.lower() == 'history':
                    self._show_history()
                    continue
                
                elif user_input.lower() == 'status':
                    self._show_status()
                    continue
                
                elif user_input.lower() == 'aiml status':
                    self._show_aiml_status()
                    continue
                
                elif user_input.lower() == 'time':
                    self._show_current_time()
                    continue
                
                elif user_input.lower() == 'web':
                    self._launch_streamlit()
                    continue
                
                elif user_input.lower().startswith('upload'):
                    self._handle_image_upload(user_input)
                    continue
                
                elif user_input.lower().startswith('generate report'):
                    self._handle_report_generation()
                    continue
                
                elif user_input.lower().startswith('check reports'):
                    self._handle_report_search(user_input)
                    continue
                
                elif user_input.lower() == 'patient data':
                    self._handle_patient_data_collection()
                    continue
                
                elif not user_input:
                    continue
                
                # 🆕 Process through enhanced supervisor with AI/ML support
                print(f"\n🔄 Processing at {current_time}...")
                
                # Determine if this might need special context
                context = self._determine_context(user_input)
                
                # Use the enhanced supervisor's handle_user_input_with_timestamp method
                response = self.supervisor.handle_user_input_with_timestamp(
                    self.current_user, 
                    user_input, 
                    current_time,
                    context
                )
                
                # FIXED deprecation
                response_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n🤖 Bot [{response_time}]: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                error_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n❌ Error at {error_time}: {str(e)}")
    
    def _determine_context(self, user_input: str) -> dict:
        """🆕 Determine if the input needs special context for AI/ML processing"""
        
        context = {}
        user_lower = user_input.lower()
        
        # Check for medical terms that might benefit from AI/ML
        medical_terms = ['parkinson', 'mri', 'brain', 'medical', 'disease', 'treatment', 'diagnosis']
        if any(term in user_lower for term in medical_terms):
            context['medical_context'] = True
        
        # Check for image-related terms
        if any(term in user_lower for term in ['image', 'scan', 'picture', 'photo']):
            context['image_related'] = True
        
        # Check for report-related terms
        if any(term in user_lower for term in ['report', 'generate', 'create', 'pdf']):
            context['report_related'] = True
        
        return context if context else None
    
    def _handle_image_upload(self, user_input: str):
        """🆕 Handle image upload commands through supervisor"""
        
        parts = user_input.split()
        
        if len(parts) < 3:
            print("❌ Usage: upload [mri|image] [path/to/image]")
            print("Example: upload mri brain_scan.jpg")
            return
        
        upload_type = parts[1].lower()
        image_path = " ".join(parts[2:])  # Handle paths with spaces
        
        # Validate image type
        if upload_type not in ['mri', 'image']:
            print("❌ Image type must be 'mri' or 'image'")
            return
        
        # Check if file exists
        if not os.path.exists(image_path):
            print(f"❌ Image file not found: {image_path}")
            print("💡 Tip: Use relative or absolute path to your image")
            return
        
        try:
            print(f"🖼️ Processing {upload_type.upper()} image: {os.path.basename(image_path)}")
            print("⏳ This may take a moment...")
            
            # Use supervisor's image upload method
            result = self.supervisor.handle_image_upload(
                self.current_user,
                image_path,
                'mri' if upload_type == 'mri' else 'general'
            )
            
            # Display results
            print("\n" + "="*80)
            print("📊 AI/ML ANALYSIS RESULTS")
            print("="*80)
            print(result)
            print("="*80)
            
        except Exception as e:
            print(f"❌ Image analysis failed: {str(e)}")
    
    def _handle_report_generation(self):
        """🆕 Handle medical report generation through supervisor"""
        
        print("📄 Medical Report Generation")
        print("-" * 40)
        
        # Collect patient information
        print("Please provide patient information:")
        
        try:
            name = input("Patient Name: ").strip()
            if not name:
                print("❌ Patient name is required")
                return
            
            age = input("Age: ").strip()
            sex = input("Sex (Male/Female/Other): ").strip()
            contact = input("Contact Number: ").strip()
            email = input("Email (optional): ").strip()
            doctor = input("Referring Doctor (optional): ").strip()
            
            # Prepare patient info
            patient_info = {
                'name': name,
                'age': age,
                'sex': sex,
                'contact': contact,
                'email': email or 'Not provided',
                'doctor': doctor or 'Not specified'
            }
            
            print("\n📄 Generating medical report...")
            
            # Use supervisor's report generation method
            result = self.supervisor.handle_report_generation(self.current_user, patient_info)
            
            print("\n" + "="*60)
            print("📊 REPORT GENERATION RESULT")
            print("="*60)
            print(result)
            print("="*60)
            
        except KeyboardInterrupt:
            print("\n🔄 Report generation cancelled.")
        except Exception as e:
            print(f"❌ Report generation failed: {str(e)}")
    
    def _handle_report_search(self, user_input: str):
        """Handle report search commands through supervisor"""
        
        parts = user_input.split()
        
        if len(parts) < 3:
            print("❌ Usage: check reports [patient_name]")
            print("Example: check reports John Doe")
            return
        
        patient_name = " ".join(parts[2:])
        
        try:
            print(f"🔍 Searching reports for: {patient_name}")
            
            # Process through supervisor
            result = self.supervisor.handle_user_input(
                self.current_user,
                f"check reports {patient_name}"
            )
            
            print("\n" + "="*60)
            print("📚 REPORT SEARCH RESULTS")
            print("="*60)
            print(result)
            print("="*60)
            
        except Exception as e:
            print(f"❌ Report search failed: {str(e)}")
    
    def _handle_patient_data_collection(self):
        """🆕 Handle patient data collection"""
        
        print("👥 Patient Data Collection")
        print("-" * 30)
        
        try:
            # Process through supervisor
            result = self.supervisor.handle_user_input(
                self.current_user,
                "collect patient data"
            )
            
            print("\n" + "="*50)
            print("📋 PATIENT DATA COLLECTION")
            print("="*50)
            print(result)
            print("="*50)
            
        except Exception as e:
            print(f"❌ Patient data collection failed: {str(e)}")
    
    def _show_aiml_status(self):
        """🆕 Show detailed AI/ML system status"""
        
        try:
            aiml_status = self.supervisor.get_aiml_status()
            
            print(f"\n🤖 AI/ML MEDICAL SYSTEM STATUS")
            print("=" * 50)
            print(f"🏥 Available: {'✅ Yes' if aiml_status['available'] else '❌ No'}")
            print(f"⚡ Status: {aiml_status['status']}")
            
            if aiml_status['available']:
                print(f"📦 Model: {aiml_status.get('model_name', 'AI/ML Medical System')}")
                print(f"🔢 Version: {aiml_status.get('version', '1.0.0')}")
                
                capabilities = aiml_status.get('capabilities', [])
                print(f"\n🎯 Capabilities ({len(capabilities)} features):")
                for cap in capabilities:
                    print(f"   • {cap}")
                
                integrated_with = aiml_status.get('integrated_with', [])
                if integrated_with:
                    print(f"\n🔗 Integrated With:")
                    for integration in integrated_with:
                        print(f"   • {integration}")
            else:
                print(f"❌ Reason: {aiml_status.get('reason', 'Unknown')}")
                if 'error' in aiml_status:
                    print(f"🔴 Error: {aiml_status['error']}")
            
            print("=" * 50)
            
        except Exception as e:
            print(f"\n🤖 AI/ML Status: ❌ Error retrieving status")
            print(f"🔴 Error: {str(e)}")
    
    def _launch_streamlit(self):
        """Launch Streamlit web interface"""
        
        print("🌐 Launching Streamlit Web Interface...")
        print("💡 This will open in your default web browser")
        print("🔄 Press Ctrl+C in the terminal to stop the web server")
        
        try:
            import subprocess
            import sys
            
            # Check if streamlit is available
            try:
                import streamlit
                print("✅ Streamlit found, launching web interface...")
            except ImportError:
                print("❌ Streamlit not installed. Install with: pip install streamlit")
                return
            
            # Launch streamlit app
            if os.path.exists("streamlit_app.py"):
                subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
            else:
                print("❌ streamlit_app.py not found in current directory")
                print("💡 Make sure streamlit_app.py exists in the same folder as main.py")
            
        except KeyboardInterrupt:
            print("\n🔄 Streamlit interface closed.")
        except Exception as e:
            print(f"❌ Failed to launch Streamlit: {str(e)}")
    
    def _show_history(self):
        """Show conversation history with dynamic formatting"""
        try:
            history = self.supervisor.get_conversation_history(self.current_user)
            
            print(f"\n📋 Conversation History for {self.current_user}")
            print("-" * 50)
            
            if not history:
                print("No conversation history yet.")
            else:
                recent_history = history[-10:] if len(history) > 10 else history
                
                if len(history) > 10:
                    print(f"... (showing last 10 of {len(history)} messages)")
                
                for msg in recent_history:
                    if isinstance(msg['timestamp'], str):
                        timestamp = msg['timestamp']
                    else:
                        timestamp = msg['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                    
                    print(f"[{timestamp}] {msg['sender']}: {msg['message']}")
            
            print("-" * 50)
        except Exception as e:
            print(f"\n📋 Conversation History")
            print("-" * 30)
            print("History temporarily unavailable.")
            print(f"Error: {str(e)[:50]}...")
            print("-" * 30)
    
    def _show_status(self):
        """🆕 Show enhanced system status through supervisor"""
        
        try:
            # Use supervisor's built-in status command
            status_response = self.supervisor.handle_user_input(self.current_user, "status")
            
            print("\n" + "="*70)
            print("📊 ENHANCED MEDICAL SYSTEM STATUS")
            print("="*70)
            print(status_response)
            print("="*70)
            
        except Exception as e:
            print(f"\n📊 BASIC SYSTEM STATUS")
            print("=" * 40)
            current_time = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            print(f"👤 User: {self.current_user}")
            print(f"🕐 Time: {current_time} UTC")
            print(f"🤖 System: {'✅ Running' if self.supervisor else '❌ Not loaded'}")
            
            try:
                aiml_status = self.supervisor.get_aiml_status()
                print(f"🔬 AI/ML: {'✅ Ready' if aiml_status['available'] else '❌ Not available'}")
            except:
                print(f"🔬 AI/ML: ❓ Status unknown")
            
            print("=" * 40)
            print(f"⚠️ Status Error: {str(e)[:50]}...")
    
    def _show_current_time(self):
        """Show current time in multiple formats"""
        now = datetime.now(timezone.utc)
        
        print(f"\n🕐 Current Time Information")
        print("-" * 30)
        print(f"UTC: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"ISO: {now.isoformat()}")
        print(f"Timestamp: {int(now.timestamp())}")
        print("-" * 30)

def run_streamlit_app():
    """Run the Streamlit web application"""
    
    print("🚀 Starting Enhanced Streamlit Medical Analysis System...")
    print(f"🕐 Current Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print(f"👤 Current User: Sagar4276")
    print("🌐 Opening web interface...")
    
    try:
        import subprocess
        import sys
        
        if os.path.exists("streamlit_app.py"):
            subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])
        else:
            print("❌ streamlit_app.py not found")
            print("💡 Make sure streamlit_app.py exists in the current directory")
    except Exception as e:
        print(f"❌ Failed to launch Streamlit: {str(e)}")

def upload_image_cli(image_path: str, image_type: str = "mri"):
    """🆕 Command line image upload through supervisor"""
    
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    print(f"🖼️ Processing {image_type.upper()} image: {os.path.basename(image_path)}")
    
    try:
        # Initialize supervisor with medical configuration
        supervisor = EnhancedSupervisorAgent(create_medical_config())
        
        # Process image through supervisor
        result = supervisor.handle_image_upload("Sagar4276", image_path, image_type)
        
        print("\n" + "="*80)
        print("📊 ANALYSIS RESULTS")
        print("="*80)
        print(result)
        print("="*80)
        
    except Exception as e:
        print(f"❌ Analysis failed: {str(e)}")

def main():
    """Main application entry point with argument parsing"""
    
    parser = argparse.ArgumentParser(description="Enhanced Medical Analysis System")
    
    parser.add_argument(
        '--mode', 
        choices=['web', 'cli', 'image'], 
        default='cli',
        help='Application mode: web (Streamlit), cli (command line), or image (image analysis)'
    )
    
    parser.add_argument(
        '--image', 
        type=str,
        help='Path to image file for analysis (use with --mode image)'
    )
    
    parser.add_argument(
        '--type', 
        choices=['mri', 'general'], 
        default='mri',
        help='Image type: mri or general (use with --mode image)'
    )
    
    args = parser.parse_args()
    
    print(f"""
🧠 Enhanced Medical Analysis System v2.0.0-AIML
Current Date and Time (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
Current User's Login: Sagar4276

Mode: {args.mode.upper()}
Features: Chat, RAG, AI/ML Image Analysis, Medical Report Generation
    """)
    
    if args.mode == 'web':
        run_streamlit_app()
    
    elif args.mode == 'cli':
        # Run the enhanced chat application
        try:
            app = EnhancedChatApp()
            app.run_chat()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            print("Please check your setup and try again.")
    
    elif args.mode == 'image':
        if not args.image:
            print("❌ Error: --image argument required for image mode")
            print("Example: python main.py --mode image --image path/to/mri.jpg --type mri")
            sys.exit(1)
        
        upload_image_cli(args.image, args.type)
    
    else:
        print("❌ Invalid mode. Use --help for options.")
        sys.exit(1)

# Run the application - backwards compatibility
if __name__ == "__main__":
    # Check if arguments are provided
    if len(sys.argv) > 1:
        main()  # Use argument parsing
    else:
        # Default behavior - run enhanced chat app
        try:
            app = EnhancedChatApp()
            app.run_chat()
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            print("Please check your setup and try again.")