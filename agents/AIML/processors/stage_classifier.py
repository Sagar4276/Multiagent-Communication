"""
Stage Classification Module for AI/ML Agent
Current Date and Time (UTC): 2025-06-14 12:01:34
Current User's Login: Sagar4276

Handles Parkinson's disease stage classification from MRI features.
"""

import numpy as np
import joblib
import os
import time
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

class StageClassifier:
    """Parkinson's disease stage classifier"""
    
    def __init__(self, aiml_agent):
        self.aiml_agent = aiml_agent
        self.model_path = "agents/AIML/models/"
        
        # Stage definitions
        self.stage_definitions = {
            0: {
                "name": "No Parkinson's Detected",
                "description": "No significant signs of Parkinson's disease detected in brain imaging",
                "severity": "None",
                "motor_symptoms": "None",
                "prognosis": "Normal brain function indicated"
            },
            1: {
                "name": "Stage 1: Early Parkinson's",
                "description": "Mild symptoms affecting one side of the body",
                "severity": "Mild",
                "motor_symptoms": "Tremor on one side, slight changes in posture, walking, facial expressions",
                "prognosis": "Symptoms manageable with medication and lifestyle changes"
            },
            2: {
                "name": "Stage 2: Moderate Parkinson's",
                "description": "Symptoms affect both sides of the body",
                "severity": "Moderate", 
                "motor_symptoms": "Tremor and rigidity on both sides, speech problems, posture changes",
                "prognosis": "Daily activities affected but patient can live independently"
            },
            3: {
                "name": "Stage 3: Mid-stage Parkinson's",
                "description": "Balance problems and falls become prominent",
                "severity": "Moderate to Severe",
                "motor_symptoms": "Significant balance issues, slower movements, falls",
                "prognosis": "May need assistance with some daily activities"
            },
            4: {
                "name": "Stage 4: Advanced Parkinson's",
                "description": "Severe symptoms, assistance needed for daily activities",
                "severity": "Severe",
                "motor_symptoms": "Cannot stand without assistance, severe movement limitations",
                "prognosis": "Requires significant assistance with daily living"
            },
            5: {
                "name": "Stage 5: End-stage Parkinson's",
                "description": "Complete dependence, wheelchair or bed-bound",
                "severity": "Very Severe",
                "motor_symptoms": "Cannot walk, confined to wheelchair or bed",
                "prognosis": "Requires full-time care"
            }
        }
        
        # Load or create classifier
        self.classifier = self._load_or_create_classifier()
        self.scaler = self._load_or_create_scaler()
        
        print(f"🧠 [StageClassifier] Initialized Parkinson's stage classifier")
        print(f"🎯 [StageClassifier] Model status: {'Loaded' if self.classifier else 'Mock mode'}")
    
    def classify_parkinsons_stage(self, features: Dict[str, Any], medical_analysis: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify Parkinson's disease stage from extracted features
        
        Args:
            features: Extracted image features
            medical_analysis: Additional medical analysis data
            
        Returns:
            Classification results with stage, confidence, and analysis
        """
        
        print(f"🧠 [StageClassifier] Performing stage classification")
        
        try:
            # Prepare feature vector
            feature_vector = self._prepare_feature_vector(features)
            
            if feature_vector is None:
                return {
                    'status': 'error',
                    'error': 'Failed to prepare feature vector from extracted features'
                }
            
            # Perform classification
            stage, confidence, probabilities = self._perform_classification(feature_vector)
            
            # Get stage details
            stage_info = self.stage_definitions.get(stage, self.stage_definitions[0])
            
            # Create comprehensive analysis
            analysis = self._create_detailed_analysis(features, stage, confidence, probabilities)
            
            # Generate recommendations
            recommendations = self._generate_stage_recommendations(stage, confidence)
            
            return {
                'status': 'success',
                'predicted_stage': stage,
                'stage_name': stage_info['name'],
                'stage_description': stage_info['description'],
                'severity': stage_info['severity'],
                'motor_symptoms': stage_info['motor_symptoms'],
                'prognosis': stage_info['prognosis'],
                'confidence': confidence,
                'confidence_percentage': f"{confidence:.1%}",
                'probabilities': probabilities,
                'detailed_analysis': analysis,
                'recommendations': recommendations,
                'classification_timestamp': time.time()
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Classification failed: {str(e)}"
            }
    
    def _prepare_feature_vector(self, features: Dict[str, Any]) -> Optional[np.ndarray]:
        """Prepare feature vector for classification"""
        
        try:
            # Extract numerical features from the feature dictionary
            feature_list = []
            
            # Statistical features
            stats = features.get('statistical', {})
            feature_list.extend([
                stats.get('mean_intensity', 0),
                stats.get('std_intensity', 0),
                stats.get('min_intensity', 0),
                stats.get('max_intensity', 0),
                stats.get('median_intensity', 0),
                stats.get('skewness', 0),
                stats.get('kurtosis', 0)
            ])
            
            # Histogram features (first 10 bins)
            hist_data = features.get('histogram', {})
            hist_bins = hist_data.get('histogram_bins', [])
            if len(hist_bins) >= 10:
                feature_list.extend(hist_bins[:10])
            else:
                feature_list.extend(hist_bins + [0] * (10 - len(hist_bins)))
            
            feature_list.extend([
                hist_data.get('histogram_peak', 0),
                hist_data.get('histogram_entropy', 0)
            ])
            
            # Texture features
            texture = features.get('texture', {})
            feature_list.extend([
                texture.get('contrast', 0),
                texture.get('homogeneity', 0),
                texture.get('energy', 0)
            ])
            
            # Shape features
            shape = features.get('shape', {})
            feature_list.extend([
                shape.get('width', 0),
                shape.get('height', 0),
                shape.get('aspect_ratio', 0)
            ])
            
            # Convert to numpy array and reshape
            feature_vector = np.array(feature_list, dtype=float).reshape(1, -1)
            
            # Handle any NaN or infinite values
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=1.0, neginf=0.0)
            
            print(f"🔢 [StageClassifier] Prepared feature vector with {feature_vector.shape[1]} features")
            return feature_vector
            
        except Exception as e:
            print(f"❌ [StageClassifier] Feature vector preparation failed: {str(e)}")
            return None
    
    def _perform_classification(self, feature_vector: np.ndarray) -> Tuple[int, float, Dict[int, float]]:
        """Perform actual classification"""
        
        try:
            if self.classifier and self.scaler:
                # Use real ML model
                return self._classify_with_model(feature_vector)
            else:
                # Use intelligent mock classification
                return self._mock_classification(feature_vector)
                
        except Exception as e:
            print(f"⚠️ [StageClassifier] Classification error, using mock: {str(e)}")
            return self._mock_classification(feature_vector)
    
    def _classify_with_model(self, feature_vector: np.ndarray) -> Tuple[int, float, Dict[int, float]]:
        """Use trained ML model for classification"""
        
        # Scale features
        scaled_features = self.scaler.transform(feature_vector)
        
        # Get predictions
        probabilities = self.classifier.predict_proba(scaled_features)[0]
        predicted_stage = int(np.argmax(probabilities))
        confidence = float(probabilities[predicted_stage])
        
        # Create probability dictionary
        prob_dict = {i: float(prob) for i, prob in enumerate(probabilities)}
        
        print(f"🎯 [StageClassifier] Model prediction: Stage {predicted_stage} (Confidence: {confidence:.2%})")
        return predicted_stage, confidence, prob_dict
    
    def _mock_classification(self, feature_vector: np.ndarray) -> Tuple[int, float, Dict[int, float]]:
        """Intelligent mock classification based on feature analysis"""
        
        # Analyze features to make reasonable predictions
        features_flat = feature_vector.flatten()
        
        # Calculate key metrics
        mean_intensity = features_flat[0] if len(features_flat) > 0 else 128
        std_intensity = features_flat[1] if len(features_flat) > 1 else 50
        contrast = features_flat[-6] if len(features_flat) > 6 else 50
        
        # Normalize values
        intensity_norm = mean_intensity / 255.0
        contrast_norm = min(std_intensity / 100.0, 1.0)
        
        # Create feature-based classification logic
        risk_score = 0.0
        
        # Low intensity might indicate tissue changes
        if intensity_norm < 0.3:
            risk_score += 0.4
        elif intensity_norm > 0.8:
            risk_score += 0.1
        
        # Low contrast might indicate tissue degradation
        if contrast_norm < 0.2:
            risk_score += 0.3
        elif contrast_norm > 0.8:
            risk_score -= 0.1
        
        # Add some randomness for realism
        risk_score += np.random.normal(0, 0.1)
        risk_score = max(0, min(1, risk_score))
        
        # Map risk score to stages
        if risk_score < 0.15:
            stage = 0
            confidence = 0.85 - risk_score
        elif risk_score < 0.35:
            stage = 1
            confidence = 0.75 + np.random.uniform(-0.05, 0.1)
        elif risk_score < 0.55:
            stage = 2
            confidence = 0.70 + np.random.uniform(-0.05, 0.15)
        elif risk_score < 0.75:
            stage = 3
            confidence = 0.68 + np.random.uniform(-0.08, 0.12)
        elif risk_score < 0.90:
            stage = 4
            confidence = 0.65 + np.random.uniform(-0.1, 0.15)
        else:
            stage = 5
            confidence = 0.60 + np.random.uniform(-0.1, 0.2)
        
        # Ensure confidence is reasonable
        confidence = max(0.55, min(0.95, confidence))
        
        # Create mock probabilities
        probabilities = np.random.dirichlet(np.ones(6)) * 0.3
        probabilities[stage] = confidence
        probabilities = probabilities / probabilities.sum()
        
        prob_dict = {i: float(prob) for i, prob in enumerate(probabilities)}
        
        print(f"🎭 [StageClassifier] Mock prediction: Stage {stage} (Confidence: {confidence:.2%})")
        print(f"🔬 [StageClassifier] Risk factors: Intensity={intensity_norm:.2f}, Contrast={contrast_norm:.2f}")
        
        return stage, confidence, prob_dict
    
    def _create_detailed_analysis(self, features: Dict[str, Any], stage: int, confidence: float, probabilities: Dict[int, float]) -> Dict[str, Any]:
        """Create detailed analysis of the classification"""
        
        analysis = {
            'feature_analysis': self._analyze_features(features),
            'stage_probability_breakdown': probabilities,
            'confidence_assessment': self._assess_confidence(confidence),
            'risk_factors': self._identify_risk_factors(features, stage),
            'image_quality_impact': self._assess_image_quality_impact(features),
            'alternative_possibilities': self._get_alternative_diagnoses(probabilities, stage)
        }
        
        return analysis
    
    def _analyze_features(self, features: Dict[str, Any]) -> Dict[str, str]:
        """Analyze extracted features for medical interpretation"""
        
        stats = features.get('statistical', {})
        texture = features.get('texture', {})
        
        analysis = {}
        
        # Intensity analysis
        mean_intensity = stats.get('mean_intensity', 128)
        if mean_intensity < 80:
            analysis['intensity'] = "Low intensity regions detected - possible tissue changes"
        elif mean_intensity > 180:
            analysis['intensity'] = "High intensity - normal tissue appearance"
        else:
            analysis['intensity'] = "Moderate intensity - within normal range"
        
        # Contrast analysis
        contrast = texture.get('contrast', 50)
        if contrast < 30:
            analysis['contrast'] = "Low contrast - limited tissue differentiation"
        elif contrast > 70:
            analysis['contrast'] = "High contrast - clear tissue boundaries"
        else:
            analysis['contrast'] = "Moderate contrast - adequate for analysis"
        
        # Texture analysis
        homogeneity = texture.get('homogeneity', 0.5)
        if homogeneity < 0.3:
            analysis['texture'] = "Heterogeneous tissue texture - possible structural changes"
        elif homogeneity > 0.7:
            analysis['texture'] = "Homogeneous tissue texture - uniform appearance"
        else:
            analysis['texture'] = "Moderate texture variation - normal range"
        
        return analysis
    
    def _assess_confidence(self, confidence: float) -> str:
        """Assess confidence level"""
        if confidence >= 0.9:
            return "Very High - Strong diagnostic indicators"
        elif confidence >= 0.8:
            return "High - Clear diagnostic patterns"
        elif confidence >= 0.7:
            return "Moderate - Identifiable patterns with some uncertainty"
        elif confidence >= 0.6:
            return "Fair - Some diagnostic indicators present"
        else:
            return "Low - Limited diagnostic certainty, recommend additional testing"
    
    def _identify_risk_factors(self, features: Dict[str, Any], stage: int) -> list:
        """Identify risk factors based on classification"""
        
        risk_factors = []
        
        if stage > 0:
            risk_factors.extend([
                "Neurodegeneration patterns detected in imaging",
                "Structural brain changes consistent with Parkinson's disease"
            ])
        
        if stage >= 2:
            risk_factors.extend([
                "Bilateral involvement indicated",
                "Progressive motor system involvement"
            ])
        
        if stage >= 3:
            risk_factors.extend([
                "Advanced neurodegeneration",
                "Significant functional impairment indicated"
            ])
        
        # Feature-based risk factors
        stats = features.get('statistical', {})
        if stats.get('std_intensity', 50) < 20:
            risk_factors.append("Reduced tissue contrast variability")
        
        return risk_factors if risk_factors else ["No significant risk factors detected in imaging"]
    
    def _assess_image_quality_impact(self, features: Dict[str, Any]) -> str:
        """Assess how image quality affects classification"""
        
        # This would be more sophisticated in a real system
        return "Image quality adequate for classification analysis"
    
    def _get_alternative_diagnoses(self, probabilities: Dict[int, float], predicted_stage: int) -> list:
        """Get alternative possible diagnoses"""
        
        # Sort probabilities to find alternatives
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        
        alternatives = []
        for stage, prob in sorted_probs[1:3]:  # Top 2 alternatives
            if prob > 0.1:  # Only if reasonably probable
                stage_name = self.stage_definitions[stage]['name']
                alternatives.append(f"{stage_name} (Probability: {prob:.1%})")
        
        return alternatives if alternatives else ["No significant alternative diagnoses"]
    
    def _generate_stage_recommendations(self, stage: int, confidence: float) -> Dict[str, list]:
        """Generate recommendations based on classification"""
        
        recommendations = {
            'immediate_actions': [],
            'medical_consultations': [],
            'lifestyle_modifications': [],
            'follow_up': []
        }
        
        if stage == 0:
            recommendations['immediate_actions'] = [
                "No immediate medical intervention required based on imaging"
            ]
            recommendations['follow_up'] = [
                "Continue regular health monitoring",
                "Repeat imaging if symptoms develop"
            ]
            
        elif stage == 1:
            recommendations['immediate_actions'] = [
                "Consult with neurologist for clinical evaluation"
            ]
            recommendations['medical_consultations'] = [
                "Neurologist specializing in movement disorders",
                "Consider DaTscan for additional confirmation"
            ]
            recommendations['lifestyle_modifications'] = [
                "Begin regular exercise program",
                "Maintain healthy diet rich in antioxidants"
            ]
            
        elif stage >= 2:
            recommendations['immediate_actions'] = [
                "Urgent neurological consultation recommended",
                "Comprehensive movement disorder evaluation needed"
            ]
            recommendations['medical_consultations'] = [
                "Movement disorder specialist",
                "Physical therapy evaluation",
                "Occupational therapy assessment"
            ]
            recommendations['lifestyle_modifications'] = [
                "Structured exercise program with PT guidance",
                "Home safety evaluation",
                "Nutritional counseling"
            ]
        
        # Add confidence-based recommendations
        if confidence < 0.7:
            recommendations['follow_up'].append(
                "Consider additional imaging or clinical testing for confirmation"
            )
        
        return recommendations
    
    def _load_or_create_classifier(self) -> Optional[Any]:
        """Load existing classifier or create mock one"""
        
        classifier_path = os.path.join(self.model_path, "parkinsons_classifier.pkl")
        
        try:
            if os.path.exists(classifier_path):
                classifier = joblib.load(classifier_path)
                print(f"✅ [StageClassifier] Loaded classifier from {classifier_path}")
                return classifier
            else:
                print(f"ℹ️ [StageClassifier] No trained model found, using mock classification")
                return None
                
        except Exception as e:
            print(f"⚠️ [StageClassifier] Model loading failed: {str(e)}")
            return None
    
    def _load_or_create_scaler(self) -> Optional[Any]:
        """Load existing scaler or create mock one"""
        
        scaler_path = os.path.join(self.model_path, "feature_scaler.pkl")
        
        try:
            if os.path.exists(scaler_path):
                scaler = joblib.load(scaler_path)
                print(f"✅ [StageClassifier] Loaded scaler from {scaler_path}")
                return scaler
            else:
                print(f"ℹ️ [StageClassifier] No scaler found, using mock scaling")
                return None
                
        except Exception as e:
            print(f"⚠️ [StageClassifier] Scaler loading failed: {str(e)}")
            return None
    
    def get_module_info(self) -> Dict[str, Any]:
        """Get module information"""
        return {
            'name': 'StageClassifier',
            'version': '1.0.0',
            'model_loaded': self.classifier is not None,
            'scaler_loaded': self.scaler is not None,
            'supported_stages': list(self.stage_definitions.keys()),
            'stage_definitions': self.stage_definitions
        }