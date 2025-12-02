import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
import re
import joblib

class RobustATSInference:
    """Enhanced ATS inference using train_ats_score.py trained model"""
    
    def __init__(self):
        self.transformer_model = None
        self.ml_model = None
        self.scaler = None
        self.selector = None
        self.pca_resume = None
        self.pca_jd = None
        self.pca_transcript = None
        self.model_metrics = {}
        
        try:
            # Load transformer
            self.transformer_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✅ Transformer model loaded")
            
            # Load ML model
            self.ml_model, self.scaler, self.selector, self.pca_resume, self.pca_jd, self.pca_transcript, self.model_metrics = self._load_ml_model()
            
            if self.ml_model is not None:
                print("✅ ATS Score Model loaded successfully")
                if self.model_metrics:
                    print(f"   Model Test R²: {self.model_metrics.get('test_r2', 0):.3f}")
                    print(f"   Model MAE: {self.model_metrics.get('mae', 0):.2f}")
                    print(f"   Overfit Gap: {self.model_metrics.get('overfit_gap', 0):.4f}")
            else:
                print("⚠️ ATS Score model not found. Please run train_ats_score.py first.")
                
        except Exception as e:
            print(f"⚠️ Init error: {e}")

    def _load_ml_model(self):
        """Load ML model from models/ats_score_model.pkl"""
        try:
            model_path = 'models/ats_score_model.pkl'
            if os.path.exists(model_path):
                data = joblib.load(model_path)
                
                if isinstance(data, dict):
                    return (
                        data.get('model'), 
                        data.get('scaler'), 
                        data.get('selector'),
                        data.get('pca_resume'),
                        data.get('pca_jd'),
                        data.get('pca_transcript'),
                        data.get('metrics', {})
                    )
            return None, None, None, None, None, None, {}
        except Exception as e:
            print(f"⚠️ ATS model load failed: {e}")
            return None, None, None, None, None, None, {}

    def extract_resume_jd_features(self, resume_text, jd_text):
        """Extract handcrafted features - MUST MATCH train_ats_score.py exactly"""
        features = []
        
        if not resume_text or not jd_text or not isinstance(resume_text, str) or not isinstance(jd_text, str):
            return np.zeros(25)
        
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        
        resume_words = resume_lower.split()
        jd_words = jd_lower.split()
        
        # Feature 1: Length ratio
        features.append(np.log1p(len(resume_words)) / np.log1p(len(jd_words)) if len(jd_words) > 0 else 0)
        
        # Features 2-3: Word overlap
        resume_set = set(resume_words)
        jd_set = set(jd_words)
        overlap = len(resume_set.intersection(jd_set))
        features.append(min(overlap / max(len(resume_set), 1), 1.0))
        features.append(min(overlap / max(len(jd_set), 1), 1.0))
        
        # Feature 4: Action verbs
        action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed',
                        'built', 'improved', 'increased', 'reduced', 'achieved', 'delivered',
                        'launched', 'optimized', 'executed', 'coordinated', 'established']
        action_count = sum(1 for verb in action_verbs if verb in resume_lower)
        features.append(min(action_count / 10.0, 1.0))
        
        # Feature 5: Tech skills match
        tech_skills = ['python', 'java', 'sql', 'javascript', 'machine learning', 'aws', 
                       'docker', 'kubernetes', 'react', 'angular', 'django', 'flask', 
                       'tensorflow', 'pytorch', 'data science', 'agile', 'cloud', 'api',
                       'git', 'linux', 'azure', 'gcp', 'mongodb', 'postgresql']
        resume_tech = [s for s in tech_skills if s in resume_lower]
        jd_tech = [s for s in tech_skills if s in jd_lower]
        tech_match = len(set(resume_tech).intersection(set(jd_tech))) / max(len(jd_tech), 1) if jd_tech else 0
        features.append(tech_match)
        
        # Features 6-8: Section presence
        for section in ['experience', 'education', 'skills']:
            features.append(1 if section in resume_lower else 0)
        
        # Feature 9: Education score
        edu_score = 0
        if any(term in resume_lower for term in ['phd', 'doctorate']):
            edu_score = 1.0
        elif any(term in resume_lower for term in ['master', 'mba', 'ms', 'ma', 'm.tech']):
            edu_score = 0.75
        elif any(term in resume_lower for term in ['bachelor', 'bs', 'ba', 'btech', 'b.tech', 'be', 'b.e']):
            edu_score = 0.5
        elif any(term in resume_lower for term in ['diploma', 'associate']):
            edu_score = 0.25
        features.append(edu_score)
        
        # Feature 10: Numbers/quantification
        numbers_count = len([w for w in resume_words if any(c.isdigit() for c in w)])
        features.append(min(numbers_count / max(len(resume_words) / 50, 1), 1.0))
        
        # Feature 11: Contact info
        features.append(1 if '@' in resume_lower or 'email' in resume_lower else 0)
        
        # Feature 12: Professional keywords
        pro_keywords = ['professional', 'certified', 'expert', 'senior', 'lead', 'principal', 'architect']
        pro_count = sum(1 for kw in pro_keywords if kw in resume_lower)
        features.append(min(pro_count / 5.0, 1.0))
        
        # Feature 13: Year mentions
        year_mentions = len([w for w in resume_words if w.isdigit() and 1 <= int(w) <= 30])
        features.append(min(year_mentions / 5.0, 1.0))
        
        # Feature 14: Completeness
        completeness = sum([
            1 if 'experience' in resume_lower else 0,
            1 if 'education' in resume_lower else 0,
            1 if 'skills' in resume_lower else 0,
            1 if '@' in resume_lower else 0,
            1 if any(c.isdigit() for c in resume_lower) else 0
        ]) / 5.0
        features.append(completeness)
        
        # Feature 15: Soft skills
        soft_skills = ['communication', 'leadership', 'teamwork', 'problem solving', 
                       'analytical', 'creative', 'adaptable', 'collaborative']
        resume_soft = [s for s in soft_skills if s in resume_lower]
        jd_soft = [s for s in soft_skills if s in jd_lower]
        soft_match = len(set(resume_soft).intersection(set(jd_soft))) / max(len(jd_soft), 1) if jd_soft else 0
        features.append(soft_match)
        
        # Feature 16: Vocabulary diversity
        features.append(len(resume_set) / max(len(resume_words), 1))
        
        # Feature 17: Average word length
        avg_word_len = np.mean([len(w) for w in resume_words if w.isalpha()]) if resume_words else 0
        features.append(min(avg_word_len / 10.0, 1.0))
        
        # Feature 18: Certifications
        cert_keywords = ['certified', 'certification', 'certificate', 'license', 'accredited']
        cert_count = sum(1 for kw in cert_keywords if kw in resume_lower)
        features.append(min(cert_count / 3.0, 1.0))
        
        # Feature 19: Projects
        project_count = resume_lower.count('project')
        features.append(min(project_count / 5.0, 1.0))
        
        # Feature 20: Achievement indicators
        achievement_patterns = ['%', 'increased', 'decreased', 'improved', 'reduced', 'grew', 'saved']
        achievement_score = sum(1 for p in achievement_patterns if p in resume_lower)
        features.append(min(achievement_score / 5.0, 1.0))
        
        # Feature 21: Domain keywords density
        jd_important_words = [w for w in jd_words if len(w) > 5 and w not in 
                              ['experience', 'skills', 'education', 'required', 'preferred']][:20]
        domain_match = sum(1 for w in jd_important_words if w in resume_lower) / max(len(jd_important_words), 1)
        features.append(domain_match)
        
        # Feature 22: Resume structure
        structure_keywords = ['summary', 'objective', 'experience', 'education', 'skills', 
                             'projects', 'certifications', 'achievements', 'awards']
        structure_score = sum(1 for kw in structure_keywords if kw in resume_lower)
        features.append(min(structure_score / 6.0, 1.0))
        
        # Feature 23: Bigram match
        jd_bigrams = set([' '.join(jd_words[i:i+2]) for i in range(len(jd_words)-1)])
        resume_bigrams = set([' '.join(resume_words[i:i+2]) for i in range(len(resume_words)-1)])
        bigram_match = len(jd_bigrams.intersection(resume_bigrams)) / max(len(jd_bigrams), 1)
        features.append(min(bigram_match, 1.0))
        
        # Feature 24: Organization mentions
        org_keywords = ['company', 'corporation', 'inc', 'ltd', 'llc', 'organization', 'university']
        org_count = sum(1 for kw in org_keywords if kw in resume_lower)
        features.append(min(org_count / 3.0, 1.0))
        
        # Feature 25: URL/Link presence
        url_indicators = ['http', 'www', 'github', 'linkedin', 'portfolio']
        url_score = sum(1 for ind in url_indicators if ind in resume_lower)
        features.append(min(url_score / 3.0, 1.0))
        
        return np.array(features)

    def create_interaction_features(self, similarity_features, handcrafted_features):
        """Create interaction features matching train_ats_score.py"""
        interactions = []
        
        resume_jd = similarity_features[0]
        resume_trans = similarity_features[1]
        trans_jd = similarity_features[2]
        
        # Two-way interactions
        interactions.append(resume_jd * resume_trans)
        interactions.append(resume_jd * trans_jd)
        interactions.append(resume_trans * trans_jd)
        
        # Three-way interaction
        interactions.append(resume_jd * resume_trans * trans_jd)
        
        # Squared terms
        interactions.append(resume_jd ** 2)
        interactions.append(trans_jd ** 2)
        
        # Key handcrafted interactions
        tech_skills = handcrafted_features[4]
        edu_level = handcrafted_features[8]
        
        interactions.append(tech_skills * resume_jd)
        interactions.append(edu_level * resume_jd)
        
        return np.array(interactions)

    def extract_features(self, resume_text, jd_text, transcript_text=""):
        """Extract all features compatible with train_ats_score.py"""
        try:
            if not resume_text or not jd_text:
                return None, None, None
            
            # Generate embeddings
            resume_emb = self.transformer_model.encode([resume_text])[0]
            jd_emb = self.transformer_model.encode([jd_text])[0]
            
            # Use resume as transcript if not provided
            if not transcript_text:
                transcript_emb = resume_emb
            else:
                transcript_emb = self.transformer_model.encode([transcript_text])[0]
            
            # Apply PCA if available
            if self.pca_resume is not None and self.pca_jd is not None and self.pca_transcript is not None:
                resume_reduced = self.pca_resume.transform([resume_emb])[0]
                jd_reduced = self.pca_jd.transform([jd_emb])[0]
                transcript_reduced = self.pca_transcript.transform([transcript_emb])[0]
            else:
                # Use first 20 dims if PCA not available
                resume_reduced = resume_emb[:20]
                jd_reduced = jd_emb[:20]
                transcript_reduced = transcript_emb[:20]
            
            # Calculate similarities
            resume_jd_sim = cosine_similarity([resume_emb], [jd_emb])[0][0]
            resume_transcript_sim = cosine_similarity([resume_emb], [transcript_emb])[0][0]
            transcript_jd_sim = cosine_similarity([transcript_emb], [jd_emb])[0][0]
            
            similarity_features = np.array([resume_jd_sim, resume_transcript_sim, transcript_jd_sim])
            
            # Extract handcrafted features
            handcrafted_features = self.extract_resume_jd_features(resume_text, jd_text)
            
            # Create interaction features
            interaction_features = self.create_interaction_features(similarity_features, handcrafted_features)
            
            # Combine all: [PCA_resume, PCA_jd, PCA_transcript, similarities, handcrafted, interactions]
            X_combined = np.hstack([
                resume_reduced,
                jd_reduced,
                transcript_reduced,
                similarity_features,
                handcrafted_features,
                interaction_features
            ])
            
            # Apply feature selection if available
            if self.selector is not None:
                X_selected = self.selector.transform([X_combined])[0]
            else:
                X_selected = X_combined
            
            return X_selected, resume_emb, jd_emb
            
        except Exception as e:
            print(f"❌ Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return None, None, None

    def predict_ats_score(self, features_vector):
        """Predict ATS score using trained model"""
        try:
            if features_vector is None:
                return 0.0
                
            if self.ml_model is not None and self.scaler is not None:
                try:
                    features_scaled = self.scaler.transform([features_vector])
                    score = self.ml_model.predict(features_scaled)[0]
                    return float(max(min(score, 100), 0))
                except Exception as e:
                    print(f"⚠️ Prediction failed: {e}")
                    return 0.0
            
            print("⚠️ Model not loaded")
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

    def get_model_status(self):
        """Get current model status"""
        status = {
            'transformer': 'Loaded' if self.transformer_model else 'Error',
            'ml_model': 'Trained' if self.ml_model else 'Not Found',
            'pca': 'Loaded' if self.pca_resume and self.pca_jd and self.pca_transcript else 'Not Found',
            'selector': 'Loaded' if self.selector else 'Not Found'
        }
        if self.model_metrics:
            status['metrics'] = self.model_metrics
        return status