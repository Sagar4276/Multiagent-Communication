"""
Enhanced Message Analysis and Routing Logic with AI/ML Support
Current Date and Time (UTC): 2025-06-14 12:26:59
Current User's Login: Sagar4276

UPDATED: Added AI/ML agent routing capabilities while preserving existing functionality.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from enum import Enum

class MessageType(Enum):
    """Enhanced message type classification with AI/ML support"""
    # Existing message types
    SYSTEM_COMMAND = "system_command"
    RESEARCH_QUERY = "research_query"
    ACADEMIC_QUESTION = "academic_question"
    TECHNICAL_QUERY = "technical_query"
    KNOWLEDGE_SEARCH = "knowledge_search"
    COMPARATIVE_ANALYSIS = "comparative_analysis"
    DEFINITION_REQUEST = "definition_request"
    GENERAL_CONVERSATION = "general_conversation"
    QUESTION = "question"
    ERROR_RECOVERY = "error_recovery"
    HISTORY_REQUEST = "history_request"
    
    # 🆕 NEW AI/ML message types
    IMAGE_UPLOAD = "image_upload"
    MRI_ANALYSIS = "mri_analysis"
    MEDICAL_IMAGE_ANALYSIS = "medical_image_analysis"
    REPORT_GENERATION = "report_generation"
    REPORT_SEARCH = "report_search"
    PATIENT_DATA_COLLECTION = "patient_data_collection"
    MEDICAL_CLASSIFICATION = "medical_classification"
    STAGE_ANALYSIS = "stage_analysis"
    MEDICAL_CONSULTATION = "medical_consultation"

class Priority(Enum):
    """Message priority levels"""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class MessageAnalysis:
    """Structured message analysis result"""
    type: MessageType
    priority: Priority
    route_target: str
    confidence: float
    keywords: List[str]
    requires_rag: bool
    estimated_complexity: str
    processing_flags: Dict[str, bool]
    context_info: Dict[str, Any] = None  # 🆕 NEW for image paths, etc.

class MessageAnalyzer:
    """Handles message analysis and routing decisions with AI/ML support"""
    
    def __init__(self):
        # Existing system command keywords
        self.system_keywords = [
            'system', 'status', 'supervisor', 'help', 'commands', 'capabilities', 
            'health', 'metrics', 'performance', 'history', 'show history', 
            'conversation history', 'sessions', 'show papers', 'list papers',
            'show documents', 'list documents', 'diagnostics'
        ]
        
        # Enhanced research keywords (keeping all your existing ones)
        self.research_keywords = [
            # General research terms
            'research', 'study', 'paper', 'analysis', 'findings', 'explain', 'what is', 
            'how does', 'compare', 'define', 'describe', 'summarize', 'overview',
            
            # Medical terms
            'parkinsons', 'parkinson', 'alzheimer', 'dementia', 'cancer', 'diabetes',
            'medical', 'disease', 'diagnosis', 'treatment', 'therapy', 'clinical',
            'patient', 'healthcare', 'medicine', 'pharmaceutical', 'drug',
            'prevention', 'symptoms', 'cure', 'medication',
            
            # AI/ML terms
            'machine learning', 'artificial intelligence', 'ai', 'ml', 'deep learning',
            'neural network', 'algorithm', 'model', 'training', 'testing', 'validation',
            'classification', 'regression', 'clustering', 'supervised', 'unsupervised',
            'reinforcement learning', 'feature', 'dataset', 'prediction', 'accuracy',
            'precision', 'recall', 'f1 score', 'cross validation', 'overfitting',
            
            # Technical terms
            'computer vision', 'nlp', 'natural language processing', 'transformer',
            'bert', 'gpt', 'cnn', 'rnn', 'lstm', 'attention', 'embedding',
            'optimization', 'gradient descent', 'backpropagation', 'loss function',
            
            # General academic terms
            'methodology', 'experiment', 'hypothesis', 'literature', 'framework',
            'evaluation', 'performance', 'benchmark', 'comparison', 'results'
        ]
        
        self.academic_keywords = [
            'academic', 'journal', 'publication', 'citation', 'peer review', 
            'conference', 'proceedings', 'bibliography', 'abstract', 'conclusion'
        ]
        
        # 🆕 NEW AI/ML specific keywords
        self.image_keywords = [
            'upload', 'image', 'analyze image', 'process image', 'upload image',
            'medical image', 'scan', 'x-ray', 'ct scan', 'dicom'
        ]
        
        self.mri_keywords = [
            'mri', 'magnetic resonance', 'brain scan', 'mri scan', 'upload mri',
            'brain mri', 'neuroimaging', 'brain imaging', 'mri analysis'
        ]
        
        self.report_keywords = [
            'generate report', 'create report', 'medical report', 'report generation',
            'patient report', 'pdf report', 'download report', 'make report'
        ]
        
        self.report_search_keywords = [
            'check reports', 'search reports', 'find reports', 'previous reports',
            'patient history', 'view reports', 'list reports', 'show reports'
        ]
        
        self.patient_data_keywords = [
            'patient data', 'patient information', 'patient details', 'collect data',
            'patient name', 'patient age', 'patient sex', 'contact information'
        ]
        
        self.medical_classification_keywords = [
            'stage classification', 'parkinson stage', 'disease stage', 'classify',
            'stage analysis', 'medical classification', 'diagnosis stage'
        ]
        
        self.medical_consultation_keywords = [
            'medical consultation', 'medical advice', 'clinical advice', 'medical help',
            'doctor consultation', 'medical opinion', 'health advice'
        ]
    
    def analyze_message(self, message: str, context: Dict[str, Any] = None) -> MessageAnalysis:
        """
        🆕 Enhanced message analysis with AI/ML support and context awareness
        
        Args:
            message: User message
            context: Additional context (image_path, image_type, etc.)
            
        Returns:
            MessageAnalysis with routing information
        """
        msg_lower = message.lower().strip()
        keywords = []
        confidence = 0.0
        context_info = context or {}
        
        # 🆕 PRIORITY 1: Check for image upload context (highest priority)
        if context and context.get('image_path'):
            image_type = context.get('image_type', 'general')
            
            if image_type == 'mri' or any(kw in msg_lower for kw in self.mri_keywords):
                message_type = MessageType.MRI_ANALYSIS
                route_target = "aiml_system"
                priority = Priority.CRITICAL  # Medical images are critical
                confidence = 0.98
                keywords = ['mri', 'image_upload'] + [kw for kw in self.mri_keywords if kw in msg_lower]
                requires_rag = True  # May need medical knowledge
                complexity = "high"
                
            else:
                message_type = MessageType.MEDICAL_IMAGE_ANALYSIS
                route_target = "aiml_system"
                priority = Priority.HIGH
                confidence = 0.95
                keywords = ['medical_image', 'image_upload'] + [kw for kw in self.image_keywords if kw in msg_lower]
                requires_rag = False
                complexity = "medium"
        
        # 🆕 PRIORITY 2: Report generation requests
        elif any(keyword in msg_lower for keyword in self.report_keywords):
            message_type = MessageType.REPORT_GENERATION
            route_target = "aiml_system"
            priority = Priority.HIGH
            confidence = 0.92
            keywords = [kw for kw in self.report_keywords if kw in msg_lower]
            requires_rag = True  # Needs treatment recommendations
            complexity = "high"
        
        # 🆕 PRIORITY 3: Report search requests
        elif any(keyword in msg_lower for keyword in self.report_search_keywords):
            message_type = MessageType.REPORT_SEARCH
            route_target = "aiml_system"
            priority = Priority.MEDIUM
            confidence = 0.88
            keywords = [kw for kw in self.report_search_keywords if kw in msg_lower]
            requires_rag = False
            complexity = "medium"
        
        # 🆕 PRIORITY 4: Image upload commands (without context)
        elif any(keyword in msg_lower for keyword in self.image_keywords + self.mri_keywords):
            if any(kw in msg_lower for kw in self.mri_keywords):
                message_type = MessageType.MRI_ANALYSIS
                priority = Priority.CRITICAL
            else:
                message_type = MessageType.IMAGE_UPLOAD
                priority = Priority.HIGH
            
            route_target = "aiml_system"
            confidence = 0.85
            keywords = [kw for kw in (self.image_keywords + self.mri_keywords) if kw in msg_lower]
            requires_rag = False
            complexity = "medium"
        
        # 🆕 PRIORITY 5: Medical classification requests
        elif any(keyword in msg_lower for keyword in self.medical_classification_keywords):
            message_type = MessageType.MEDICAL_CLASSIFICATION
            route_target = "aiml_system"
            priority = Priority.HIGH
            confidence = 0.80
            keywords = [kw for kw in self.medical_classification_keywords if kw in msg_lower]
            requires_rag = True  # Needs medical knowledge
            complexity = "high"
        
        # 🆕 PRIORITY 6: Patient data collection
        elif any(keyword in msg_lower for keyword in self.patient_data_keywords):
            message_type = MessageType.PATIENT_DATA_COLLECTION
            route_target = "aiml_system"
            priority = Priority.MEDIUM
            confidence = 0.75
            keywords = [kw for kw in self.patient_data_keywords if kw in msg_lower]
            requires_rag = False
            complexity = "low"
        
        # 🆕 PRIORITY 7: Medical consultation requests
        elif any(keyword in msg_lower for keyword in self.medical_consultation_keywords):
            message_type = MessageType.MEDICAL_CONSULTATION
            route_target = "aiml_system"  # Route to AI/ML for medical advice
            priority = Priority.HIGH
            confidence = 0.82
            keywords = [kw for kw in self.medical_consultation_keywords if kw in msg_lower]
            requires_rag = True  # Needs medical knowledge
            complexity = "high"
        
        # EXISTING LOGIC: System command detection
        elif any(keyword in msg_lower for keyword in self.system_keywords):
            message_type = MessageType.SYSTEM_COMMAND
            route_target = "supervisor"
            priority = Priority.HIGH
            confidence = 0.95
            keywords = [kw for kw in self.system_keywords if kw in msg_lower]
            requires_rag = False
            complexity = "low"
        
        # EXISTING LOGIC: Research query detection - enhanced for medical
        elif (any(keyword in msg_lower for keyword in self.research_keywords) or 
              (len(message.split()) > 1 and not any(word in msg_lower for word in ['hi', 'hello', 'hey', 'thanks', 'bye']))):
            
            # Check if medical research should go to AI/ML
            medical_terms = ['parkinsons', 'parkinson', 'medical', 'disease', 'treatment', 'clinical', 'patient']
            if any(term in msg_lower for term in medical_terms):
                message_type = MessageType.RESEARCH_QUERY
                route_target = "rag_system"  # Keep medical research in RAG for now
                priority = Priority.HIGH
                confidence = 0.9
                requires_rag = True
                complexity = "medium"
            else:
                message_type = MessageType.RESEARCH_QUERY
                route_target = "rag_system"
                priority = Priority.HIGH
                confidence = 0.9
                requires_rag = True
                complexity = "medium"
            
            keywords = [kw for kw in self.research_keywords if kw in msg_lower]
            if not keywords:
                keywords = ['research_query']
        
        # EXISTING LOGIC: Academic question detection
        elif any(keyword in msg_lower for keyword in self.academic_keywords):
            message_type = MessageType.ACADEMIC_QUESTION
            route_target = "rag_system"
            priority = Priority.HIGH
            confidence = 0.85
            keywords = [kw for kw in self.academic_keywords if kw in msg_lower]
            requires_rag = True
            complexity = "high"
        
        # EXISTING LOGIC: Question detection
        elif '?' in message or len(message.split()) > 2:
            message_type = MessageType.QUESTION
            route_target = "rag_system"
            priority = Priority.MEDIUM
            confidence = 0.8
            keywords = ['question']
            requires_rag = True
            complexity = "medium"
        
        # EXISTING LOGIC: General conversation
        else:
            message_type = MessageType.GENERAL_CONVERSATION
            route_target = "chat_system"
            priority = Priority.LOW
            confidence = 0.5
            keywords = ['conversation']
            requires_rag = False
            complexity = "low"
        
        # 🆕 Enhanced processing flags
        processing_flags = {
            'requires_rag': requires_rag,
            'high_priority': priority in [Priority.CRITICAL, Priority.HIGH],
            'complex_query': complexity == "high",
            'system_command': message_type == MessageType.SYSTEM_COMMAND,
            'medical_related': route_target == "aiml_system",  # 🆕 NEW
            'image_processing': message_type in [MessageType.IMAGE_UPLOAD, MessageType.MRI_ANALYSIS, MessageType.MEDICAL_IMAGE_ANALYSIS],  # 🆕 NEW
            'report_related': message_type in [MessageType.REPORT_GENERATION, MessageType.REPORT_SEARCH],  # 🆕 NEW
            'requires_patient_data': message_type in [MessageType.REPORT_GENERATION, MessageType.PATIENT_DATA_COLLECTION]  # 🆕 NEW
        }
        
        return MessageAnalysis(
            type=message_type,
            priority=priority,
            route_target=route_target,
            confidence=confidence,
            keywords=keywords,
            requires_rag=requires_rag,
            estimated_complexity=complexity,
            processing_flags=processing_flags,
            context_info=context_info  # 🆕 NEW
        )
    
    def get_routing_summary(self) -> Dict[str, List[str]]:
        """🆕 Get summary of all routing keywords for debugging"""
        return {
            'aiml_system': {
                'image_keywords': self.image_keywords,
                'mri_keywords': self.mri_keywords,
                'report_keywords': self.report_keywords,
                'report_search_keywords': self.report_search_keywords,
                'patient_data_keywords': self.patient_data_keywords,
                'medical_classification_keywords': self.medical_classification_keywords,
                'medical_consultation_keywords': self.medical_consultation_keywords
            },
            'rag_system': self.research_keywords + self.academic_keywords,
            'chat_system': ['general conversation', 'greetings'],
            'supervisor': self.system_keywords
        }
    
    def get_aiml_capabilities(self) -> List[str]:
        """🆕 Get list of AI/ML capabilities for status reporting"""
        return [
            "MRI Image Analysis",
            "Medical Image Processing", 
            "Parkinson's Stage Classification",
            "Medical Report Generation",
            "Patient Data Collection",
            "Report Search and History",
            "Medical Consultation Support",
            "PDF Report Creation"
        ]
    
    def is_medical_emergency(self, message: str) -> bool:
        """🆕 Detect potential medical emergency keywords"""
        emergency_keywords = [
            'emergency', 'urgent', 'critical', 'severe pain', 'chest pain',
            'difficulty breathing', 'stroke', 'heart attack', 'seizure',
            'unconscious', 'bleeding', 'accident', 'trauma'
        ]
        
        msg_lower = message.lower()
        return any(keyword in msg_lower for keyword in emergency_keywords)
    
    def get_confidence_explanation(self, analysis: MessageAnalysis) -> str:
        """🆕 Get human-readable explanation of confidence score"""
        confidence = analysis.confidence
        
        if confidence >= 0.9:
            return "Very High - Clear indicators present"
        elif confidence >= 0.8:
            return "High - Strong pattern match"
        elif confidence >= 0.7:
            return "Good - Reasonable indicators"
        elif confidence >= 0.6:
            return "Moderate - Some uncertainty"
        else:
            return "Low - Ambiguous request"