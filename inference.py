import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import joblib

class RobustATSInference:
    """Enhanced ATS inference using trained model with handcrafted features"""
    
    def __init__(self):
        self.transformer_model = None
        self.ml_model = None
        self.scaler = None
        self.feature_names = []
        self.model_metrics = {}
        
        try:
            # Load transformer
            self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Transformer model loaded")
            
            # Load ML model
            self.ml_model, self.scaler, self.model_metrics = self._load_ml_model()
            
            if self.ml_model is not None:
                print("✅ Trained ML Model loaded successfully")
                if self.model_metrics:
                    print(f"   Model R²: {self.model_metrics.get('r2', 0):.3f}")
                    print(f"   Model MAE: {self.model_metrics.get('mae', 0):.2f}")
            else:
                print("⚠️ Trained model not found. Please run train_model.py first.")
                
        except Exception as e:
            print(f"⚠️ Init error: {e}")

    def _load_ml_model(self):
        """Load ML model from disk"""
        try:
            model_path = 'models/optimized_ats_model.pkl'
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    data = joblib.load(f)
                
                if isinstance(data, dict) and 'model' in data:
                    return data['model'], data.get('scaler'), data.get('metrics', {})
            return None, None, {}
        except Exception as e:
            print(f"⚠️ ML model load failed: {e}")
            return None, None, {}

    def extract_additional_features(self, resume_text, jd_text):
        """Extract handcrafted features matching training pipeline"""
        features = []
        
        # Text length features
        features.append(len(resume_text.split()))
        features.append(len(jd_text.split()))
        features.append(len(resume_text.split()) / (len(jd_text.split()) + 1))
        
        # Keyword overlap features
        resume_words = set(resume_text.lower().split())
        jd_words = set(jd_text.lower().split())
        overlap = len(resume_words.intersection(jd_words))
        features.append(overlap)
        features.append(overlap / (len(jd_words) + 1))
        
        # Section presence features
        sections = ['education', 'experience', 'skills', 'work', 'projects', 'certifications']
        for section in sections:
            features.append(1 if section in resume_text.lower() else 0)
        
        # Action verbs count
        action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed', 
                        'built', 'improved', 'increased', 'reduced', 'achieved']
        action_verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
        features.append(action_verb_count)
        
        # Quantifiable achievements
        numbers_count = len(re.findall(r'\d+%|\$\d+|\d+\+', resume_text))
        features.append(numbers_count)
        
        # Technical skills
        tech_skills = ['python', 'java', 'sql', 'javascript', 'machine learning', 'aws', 
                       'docker', 'kubernetes', 'react', 'angular', 'django', 'flask']
        tech_skills_count = sum(1 for skill in tech_skills if skill in resume_text.lower())
        features.append(tech_skills_count)
        
        # Contact info presence
        has_email = 1 if '@' in resume_text else 0
        has_phone = 1 if re.search(r'\d{10}', resume_text) else 0
        features.append(has_email)
        features.append(has_phone)
        
        return np.array(features)

    def extract_features(self, resume_text, jd_text):
        """Extract all features compatible with the training pipeline"""
        try:
            if not resume_text or not jd_text:
                return None, None, None
            
            # Get embeddings
            resume_emb = self.transformer_model.encode([resume_text])[0]
            jd_emb = self.transformer_model.encode([jd_text])[0]
            
            # Calculate similarity
            similarity = cosine_similarity([resume_emb], [jd_emb])[0][0]
            
            # Extract handcrafted features
            additional_feats = self.extract_additional_features(resume_text, jd_text)
            
            # Combine features: [JD_Embedding, Resume_Embedding, Cosine_Similarity, Handcrafted_Features]
            features_vector = np.hstack([jd_emb, resume_emb, [similarity], additional_feats])
            
            return features_vector, resume_emb, jd_emb
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            return None, None, None

    def predict_ats_score(self, features_vector):
        """Predict score using trained ML model"""
        try:
            if features_vector is None:
                return 0.0
                
            # Use ML model if available
            if self.ml_model is not None and self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform([features_vector])
                    score = self.ml_model.predict(features_scaled)[0]
                    # Ensure score is in valid range
                    return float(max(min(score, 100), 0))
                except Exception as e:
                    print(f"⚠️ ML prediction failed: {e}")
                    return 0.0
            
            # Fallback if model is missing
            print("⚠️ Model not loaded - cannot predict")
            return 0.0
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return 0.0

    def get_ats_category(self, score):
        """Categorize ATS score"""
        if score >= 80:
            return "Excellent Match"
        elif score >= 65:
            return "Good Match"
        elif score >= 50:
            return "Potential Fit"
        elif score >= 35:
            return "Weak Match"
        else:
            return "Poor Fit"

    def calculate_similarity_percentage(self, resume_emb, jd_emb):
        """Calculate semantic similarity percentage"""
        try:
            if resume_emb is None or jd_emb is None:
                return 0.0
            sim = cosine_similarity([resume_emb], [jd_emb])[0][0] * 100
            return float(max(min(sim, 100), 0))
        except:
            return 0.0

    def get_feature_importance(self):
        """Get feature importance (proxy for visualization)"""
        return {
            'Semantic Similarity': 0.35,
            'Contextual Match': 0.25,
            'Skill Alignment': 0.20,
            'Content Quality': 0.12,
            'Structure': 0.08
        }

    def get_model_status(self):
        """Get current model status"""
        status = {
            'transformer': 'Loaded' if self.transformer_model else 'Error',
            'ml_model': 'Trained' if self.ml_model else 'Not Found'
        }
        if self.model_metrics:
            status['metrics'] = self.model_metrics
        return status