"""
Enhanced Chat Agent - Query Analysis Module (COMPLETE FIX)
Current Date and Time (UTC - YYYY-MM-DD HH:MM:SS formatted): 2025-06-13 06:55:50
Current User's Login: Sagar4276
"""

import re
from typing import Dict, Any
from dataclasses import dataclass

@dataclass
class QueryAnalysis:
    """Query analysis result"""
    is_research_query: bool
    query_type: str
    confidence: float
    domain: str
    keywords: list
    requires_rag: bool

class QueryAnalyzer:
    """SAME CLASS NAME - Analyzes user queries to determine routing and processing strategy"""
    
    def __init__(self, agent_name: str = "QueryAnalyzer"):
        self.agent_name = agent_name
        self.current_user = "Sagar4276"
        
        # Enhanced research indicators (comprehensive list)
        self.research_indicators = [
            # Direct research terms
            'paper', 'research', 'study', 'studies', 'analysis', 'findings', 'results',
            'method', 'methodology', 'approach', 'algorithm', 'model', 'experiment', 
            'data', 'dataset', 'conclusion', 'abstract', 'literature', 'review',
            
            # Action words for research
            'explain', 'summarize', 'compare', 'discuss', 'analyze', 'describe',
            'tell me about', 'what is', 'how does', 'find', 'search', 'look up',
            'define', 'definition', 'overview', 'survey', 'examine',
            
            # Medical/Scientific terms
            'parkinson', 'disease', 'medical', 'treatment', 'symptoms', 'diagnosis',
            'therapy', 'clinical', 'patient', 'brain', 'neural', 'neuron', 'cognitive',
            'pharmaceutical', 'drug', 'medication', 'syndrome', 'disorder',
            
            # Technical domains
            'machine learning', 'deep learning', 'neural network', 'artificial intelligence',
            'computer vision', 'natural language', 'reinforcement learning', 'supervised',
            'unsupervised', 'classification', 'regression', 'clustering', 'optimization',
            'transformer', 'attention', 'convolutional', 'recurrent', 'lstm', 'gru',
            'bert', 'gpt', 'nlp', 'cv', 'ai', 'ml', 'dl', 'cnn', 'rnn',
            
            # Scientific terms
            'hypothesis', 'theory', 'evidence', 'correlation', 'causation', 'statistical',
            'significance', 'methodology', 'systematic', 'meta-analysis', 'randomized'
        ]
        
        # Enhanced research patterns
        self.research_patterns = [
            r'\bwhat (is|are)\b',
            r'\bhow (does|do|can|to)\b',
            r'\bwhy (is|are|does|do)\b',
            r'\bcompare\b.*\band\b',
            r'\bdifference between\b',
            r'\badvantages of\b',
            r'\bdisadvantages of\b',
            r'\bbenefits of\b',
            r'\bimpact of\b',
            r'\beffects of\b',
            r'\bapplications of\b'
        ]
    
    def analyze_query(self, message: str) -> QueryAnalysis:
        """SAME FUNCTION NAME - Enhanced query analysis for better routing"""
        msg_lower = message.lower().strip()
        
        # Enhanced research detection
        is_research = self._is_research_query(msg_lower)
        
        # Smart domain identification
        domain = self._identify_domain(msg_lower)
        
        # Enhanced keyword extraction
        keywords = self._extract_keywords(msg_lower)
        
        # Improved query type classification
        query_type = self._classify_query_type(msg_lower, is_research)
        
        # Enhanced confidence calculation
        confidence = self._calculate_confidence(msg_lower, is_research, keywords)
        
        return QueryAnalysis(
            is_research_query=is_research,
            query_type=query_type,
            confidence=confidence,
            domain=domain,
            keywords=keywords,
            requires_rag=is_research
        )
    
    def _is_research_query(self, msg_lower: str) -> bool:
        """SAME FUNCTION NAME - More aggressive research query detection (FIXED)"""
        
        # Immediate research indicators (expanded)
        immediate_research_terms = [
            'stages', 'stage', 'preventive', 'measures', 'prevention', 'treatment', 
            'treatments', 'therapy', 'therapies', 'symptoms', 'symptom', 'causes', 
            'cause', 'effects', 'effect', 'management', 'diagnosis', 'prognosis',
            'medication', 'medications', 'drug', 'drugs', 'clinical', 'medical',
            'research', 'study', 'studies', 'findings', 'results', 'evidence',
            'parkinson', 'parkinsons', 'disease', 'disorder', 'condition',
            'neurological', 'neurodegenerative', 'progressive', 'dopamine',
            'brain', 'neurons', 'motor', 'non-motor', 'rigidity', 'tremor',
            'bradykinesia', 'postural', 'gait', 'cognitive', 'dementia'
        ]
        
        # Check for immediate research indicators
        if any(term in msg_lower for term in immediate_research_terms):
            return True
        
        # Check for existing research indicators
        if any(term in msg_lower for term in self.research_indicators):
            return True
        
        # Check for pattern matching
        for pattern in self.research_patterns:
            if re.search(pattern, msg_lower):
                return True
        
        # Enhanced heuristics
        # Single-word medical/scientific terms
        single_word_research = [
            'genetics', 'inheritance', 'hereditary', 'familial', 'sporadic',
            'idiopathic', 'onset', 'progression', 'deterioration', 'decline',
            'impairment', 'disability', 'functional', 'mobility', 'balance',
            'coordination', 'speech', 'swallowing', 'sleep', 'depression',
            'anxiety', 'hallucinations', 'delusions', 'psychosis', 'cognition'
        ]
        
        if any(term in msg_lower for term in single_word_research):
            return True
        
        # Question words + length heuristic (more aggressive)
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who']
        if any(word in msg_lower for word in question_words) and len(msg_lower.split()) >= 1:
            return True
        
        # Medical abbreviations and acronyms
        medical_acronyms = ['pd', 'dbs', 'mao-b', 'comt', 'ldopa', 'adbs']
        if any(acronym in msg_lower for acronym in medical_acronyms):
            return True
        
        # Technical terms suggest research intent (more aggressive)
        technical_indicators = [
            'model', 'system', 'approach', 'technique', 'process', 'method',
            'analysis', 'evaluation', 'assessment', 'monitoring', 'tracking',
            'classification', 'diagnosis', 'screening', 'testing', 'examination'
        ]
        if any(term in msg_lower for term in technical_indicators):
            return True
        
        # If it's not clearly conversational, assume it's research
        conversational_only = [
            'hello', 'hi', 'hey', 'thanks', 'thank you', 'bye', 'goodbye',
            'how are you', 'good morning', 'good afternoon', 'good evening'
        ]
        
        if not any(phrase in msg_lower for phrase in conversational_only):
            # If it has more than 2 words and isn't clearly conversational, treat as research
            if len(msg_lower.split()) >= 2:
                return True
        
        return False
    
    def _identify_domain(self, msg_lower: str) -> str:
        """SAME FUNCTION NAME - Enhanced domain identification"""
        # Medical domain (enhanced)
        medical_terms = ['parkinson', 'disease', 'medical', 'treatment', 'symptoms', 
                        'diagnosis', 'therapy', 'clinical', 'patient', 'brain', 
                        'pharmaceutical', 'drug', 'medication', 'health', 'healthcare',
                        'neurological', 'cognitive', 'motor', 'dopamine']
        if any(term in msg_lower for term in medical_terms):
            return "medical"
        
        # AI/ML domain (enhanced)
        ai_terms = ['machine learning', 'deep learning', 'neural network', 
                   'artificial intelligence', 'ai', 'ml', 'dl', 'algorithm',
                   'model training', 'classification', 'regression', 'clustering',
                   'supervised', 'unsupervised', 'reinforcement']
        if any(term in msg_lower for term in ai_terms):
            return "artificial_intelligence"
        
        # Computer Science domain (enhanced)
        cs_terms = ['computer', 'programming', 'software', 'algorithm', 'data structure',
                   'computational', 'computing', 'system', 'architecture', 'framework']
        if any(term in msg_lower for term in cs_terms):
            return "computer_science"
        
        # Business/Economics domain (new)
        business_terms = ['business', 'market', 'economic', 'finance', 'strategy', 
                         'management', 'organization', 'enterprise']
        if any(term in msg_lower for term in business_terms):
            return "business"
        
        # General science (enhanced)
        science_terms = ['research', 'study', 'experiment', 'hypothesis', 'theory',
                        'scientific', 'empirical', 'evidence', 'methodology']
        if any(term in msg_lower for term in science_terms):
            return "science"
        
        return "general"
    
    def _extract_keywords(self, msg_lower: str) -> list:
        """SAME FUNCTION NAME - Enhanced keyword extraction"""
        # Extract words and handle compound terms
        words = re.findall(r'\b[a-zA-Z]+\b', msg_lower)
        
        # Enhanced stop words list
        stop_words = {
            'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
            'has', 'have', 'this', 'that', 'with', 'from', 'they', 'been', 'were',
            'was', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'does', 'did', 'do', 'be', 'being', 'is', 'am', 'are'
        }
        
        # Filter meaningful words (enhanced)
        keywords = []
        for word in words:
            if (len(word) > 3 and 
                word not in stop_words and 
                not word.isdigit() and
                word.isalpha()):
                keywords.append(word)
        
        # Look for compound terms from research indicators
        compound_terms = []
        for term in self.research_indicators:
            if ' ' in term and term in msg_lower:
                compound_terms.append(term.replace(' ', '_'))
        
        # Combine and deduplicate
        all_keywords = list(set(keywords + compound_terms))
        
        return all_keywords[:12]  # Return top 12 keywords
    
    def _classify_query_type(self, msg_lower: str, is_research: bool) -> str:
        """SAME FUNCTION NAME - Enhanced query type classification"""
        if not is_research:
            if any(cmd in msg_lower for cmd in ['show papers', 'list papers', 'status']):
                return "system_command"
            elif any(word in msg_lower for word in ['hello', 'hi', 'hey', 'greetings']):
                return "greeting"
            elif any(phrase in msg_lower for phrase in ['help', 'commands', 'capabilities']):
                return "help_request"
            elif any(phrase in msg_lower for phrase in ['how are you', 'how\'re you']):
                return "status_inquiry"
            else:
                return "general_conversation"
        
        # Enhanced research query types
        if any(pattern in msg_lower for pattern in ['what is', 'what are', 'define', 'definition']):
            return "definition_request"
        elif any(pattern in msg_lower for pattern in ['compare', 'difference', 'versus', 'vs']):
            return "comparison_request"
        elif any(pattern in msg_lower for pattern in ['explain', 'how does', 'how do', 'describe']):
            return "explanation_request"
        elif any(pattern in msg_lower for pattern in ['find', 'search', 'look up', 'locate']):
            return "search_request"
        elif any(pattern in msg_lower for pattern in ['why', 'reason', 'cause']):
            return "causation_request"
        elif any(pattern in msg_lower for pattern in ['when', 'time', 'period']):
            return "temporal_request"
        elif any(pattern in msg_lower for pattern in ['where', 'location', 'place']):
            return "spatial_request"
        else:
            return "research_query"
    
    def _calculate_confidence(self, msg_lower: str, is_research: bool, keywords: list) -> float:
        """SAME FUNCTION NAME - Enhanced confidence calculation"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence for research queries
        if is_research:
            # Count research indicators (enhanced)
            indicator_count = sum(1 for term in self.research_indicators if term in msg_lower)
            confidence += min(indicator_count * 0.08, 0.4)
            
            # Pattern matches (enhanced)
            pattern_matches = sum(1 for pattern in self.research_patterns 
                                if re.search(pattern, msg_lower))
            confidence += min(pattern_matches * 0.12, 0.3)
            
            # Question structure bonus
            if any(word in msg_lower for word in ['what', 'how', 'why', 'when', 'where']):
                confidence += 0.1
            
            # Technical domain bonus
            if any(domain_term in msg_lower for domain_term in 
                   ['machine learning', 'neural', 'algorithm', 'medical', 'clinical']):
                confidence += 0.15
        
        # Keyword relevance (enhanced)
        if len(keywords) > 5:
            confidence += 0.15
        elif len(keywords) > 3:
            confidence += 0.1
        
        # Query length consideration
        word_count = len(msg_lower.split())
        if word_count > 8:  # Longer queries suggest more specific intent
            confidence += 0.1
        elif word_count < 3:  # Very short queries are less confident
            confidence -= 0.1
        
        return min(max(confidence, 0.0), 1.0)  # Clamp between 0 and 1
    
    def is_system_command(self, message: str) -> bool:
        """SAME FUNCTION NAME - Enhanced system command detection"""
        msg_lower = message.lower().strip()
        
        # Enhanced system commands list
        system_commands = [
            'show papers', 'list papers', 'paper list', 'papers',
            'rag status', 'search status', 'system status', 'vectorization status',
            'vectorization', 'vector info', 'help', 'commands', 'capabilities',
            'ai info', 'model info', 'system info', 'stats', 'statistics'
        ]
        
        # Exact matches and partial matches
        for cmd in system_commands:
            if cmd in msg_lower:
                return True
        
        # Pattern-based command detection
        command_patterns = [
            r'^(show|list|display)\s+(papers?|documents?|files?)$',
            r'^(rag|system|vectorization)\s+(status|info|stats)$',
            r'^(help|commands?|capabilities)$'
        ]
        
        for pattern in command_patterns:
            if re.match(pattern, msg_lower):
                return True
        
        return False
    
    def get_command_type(self, message: str) -> str:
        """SAME FUNCTION NAME - Enhanced command type detection"""
        msg_lower = message.lower().strip()
        
        if any(cmd in msg_lower for cmd in ['show papers', 'list papers', 'papers']):
            return "show_papers"
        elif any(cmd in msg_lower for cmd in ['rag status', 'system status']):
            return "system_status"
        elif any(cmd in msg_lower for cmd in ['vectorization', 'vector info']):
            return "vectorization_info"
        elif any(cmd in msg_lower for cmd in ['ai info', 'model info']):
            return "ai_info"
        elif any(cmd in msg_lower for cmd in ['help', 'commands', 'capabilities']):
            return "help"
        elif any(cmd in msg_lower for cmd in ['stats', 'statistics']):
            return "statistics"
        else:
            return "unknown_command"
    
    def debug_analysis(self, message: str) -> Dict[str, Any]:
        """NEW function - Debug query analysis for troubleshooting"""
        analysis = self.analyze_query(message)
        
        debug_info = {
            'original_query': message,
            'normalized_query': message.lower().strip(),
            'is_research_query': analysis.is_research_query,
            'query_type': analysis.query_type,
            'confidence': analysis.confidence,
            'domain': analysis.domain,
            'keywords': analysis.keywords,
            'requires_rag': analysis.requires_rag,
            'word_count': len(message.split()),
            'detected_indicators': []
        }
        
        # Check which indicators were found
        msg_lower = message.lower().strip()
        for term in self.research_indicators:
            if term in msg_lower:
                debug_info['detected_indicators'].append(term)
        
        return debug_info
    
    # NEW enhanced functions (optional to use)
    def get_query_complexity(self, message: str) -> str:
        """NEW function - Determine query complexity level"""
        msg_lower = message.lower().strip()
        word_count = len(msg_lower.split())
        
        # Count technical terms
        technical_count = sum(1 for term in self.research_indicators if term in msg_lower)
        
        if word_count > 15 or technical_count > 3:
            return "complex"
        elif word_count > 8 or technical_count > 1:
            return "moderate"
        else:
            return "simple"
    
    def get_intent_strength(self, analysis: QueryAnalysis) -> str:
        """NEW function - Determine intent strength"""
        if analysis.confidence >= 0.8:
            return "strong"
        elif analysis.confidence >= 0.6:
            return "moderate"
        else:
            return "weak"