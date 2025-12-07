import streamlit as st
# CRITICAL: st.set_page_config MUST be the first Streamlit command
st.set_page_config(
    page_title="Advanced ATS System",
    page_icon="📊",
    layout="wide"
)

import pandas as pd
import os
import re
import time
import uuid
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Import custom modules
try:
    from utils import FileParser, ResumeValidator, DatabaseManager
    from inference import RobustATSInference
    from job_recommender import EnhancedJobRecommender
    from gemini_integration import get_gemini_client
except ImportError as e:
    st.error(f"❌ Critical Import Error: {e}")
    st.error("Please ensure all required modules are present")
    st.stop()

# Custom CSS
st.markdown("""
<style>
.portal-header {
    background: linear-gradient(45deg, #1f77b4, #ff7f0e);
    padding: 2rem;
    border-radius: 10px;
    color: white;
    margin-bottom: 2rem;
}
.score-card {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
    text-align: center;
}
.poor-fit { background-color: #f8d7da; color: #721c24; }
.weak-match { background-color: #fff3cd; color: #856404; }
.good-match { background-color: #d4edda; color: #155724; }
.excellent-match { background-color: #d1ecf1; color: #0c5460; }
.potential-fit { background-color: #fff3cd; color: #856404; }
.chat-message {
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
.user-message { background-color: #e3f2fd; border-left: 4px solid #2196f3; }
.assistant-message { background-color: #f3e5f5; border-left: 4px solid #9c27b0; }
.chat-container {
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 1rem;
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
}
.job-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
}
.real-job { border-left: 4px solid #28a745; }
.candidate-card {
    border: 1px solid #ddd;
    border-radius: 10px;
    padding: 1rem;
    margin: 0.5rem 0;
    transition: all 0.3s ease;
}
.candidate-card:hover {
    box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    transform: translateY(-2px);
}
.shortlisted {
    border-left: 4px solid #28a745;
    background-color: #f8fff9;
}
.rejected {
    border-left: 4px solid #dc3545;
    background-color: #fff8f8;
}
.pending {
    border-left: 4px solid #ffc107;
}
.entry-level { border-left: 4px solid #17a2b8; background-color: #f0f8ff; }
.mid-level { border-left: 4px solid #ffc107; background-color: #fffbf0; }
.senior-level { border-left: 4px solid #28a745; background-color: #f0fff4; }
</style>
""", unsafe_allow_html=True)

class AdvancedATSApp:
    def __init__(self):
        try:
            self.file_parser = FileParser()
            self.resume_validator = ResumeValidator()
            self.db_manager = DatabaseManager()
            self.ats_inference = RobustATSInference()
            self.job_recommender = EnhancedJobRecommender()
            self.gemini = get_gemini_client()
            
            # Load classifier and clustering models
            self.load_additional_models()
        except Exception as e:
            st.error(f"❌ Initialization Error: {e}")
            st.stop()
        
        self._init_session_state()
    
    def load_additional_models(self):
        """Load resume classifier and clustering models - FIXED VERSION"""
        import pickle

        # Load resume category classifier - FIXED FILE NAMES
        try:
            # Load the model and vectorizer separately
            with open('resume_classifier_model.pkl', 'rb') as f:
                self.category_model = pickle.load(f)
            with open('tfidf_vectorizer.pkl', 'rb') as f:
                self.tfidf_vectorizer = pickle.load(f)

            st.session_state.category_model_loaded = True
            print("✅ Resume category classifier loaded")
            print(f"   Model type: {type(self.category_model).__name__}")
        except Exception as e:
            print(f"⚠️ Could not load category classifier: {e}")
            print("   Trying alternative format...")

            # Fallback: Try loading complete model data
            try:
                with open('resume_classifier_complete.pkl', 'rb') as f:
                    model_data = pickle.load(f)
                    self.category_model = model_data.get('model')
                    self.tfidf_vectorizer = model_data.get('vectorizer')

                st.session_state.category_model_loaded = True
                print("✅ Resume category classifier loaded (complete format)")
            except Exception as e2:
                print(f"❌ Failed to load category classifier: {e2}")
                self.category_model = None
                self.tfidf_vectorizer = None
                st.session_state.category_model_loaded = False

        # Load experience clustering model (moved inside method)
        try:
            with open('clustering_model_optimized.pkl', 'rb') as f:
                clustering_data = pickle.load(f)
                self.clustering_model = clustering_data.get('model')
                self.clustering_scaler = clustering_data.get('scaler')
                self.clustering_pca = clustering_data.get('pca')
                self.clustering_features = clustering_data.get('feature_columns')
                self.clustering_labels = clustering_data.get('cluster_labels')
            st.session_state.clustering_model_loaded = True
            print("✅ Experience clustering model loaded")
        except Exception as e:
            print(f"⚠️ Experience clustering model not found: {e}")
            self.clustering_model = None
            st.session_state.clustering_model_loaded = False
    

    def _init_session_state(self):
        st.session_state.setdefault('current_resume', None)
        st.session_state.setdefault('current_jd', None)
        st.session_state.setdefault('job_seeker_chat_history', [])
        st.session_state.setdefault('recruiter_resumes', [])
        st.session_state.setdefault('shortlisted_candidates', [])
        st.session_state.setdefault('recruiter_jds', [])
        st.session_state.setdefault('selected_jd', None)
        st.session_state.setdefault('uploaded_resume_ids', set())
            
    def _rerun(self):
        """Safe rerun method"""
        try:
            st.rerun()
        except AttributeError:
            st.experimental_rerun()
    
    def setup_sidebar(self):
        """Setup sidebar configuration"""
        st.sidebar.title("🔧 Configuration")
        
        # Quick Stats
        st.sidebar.markdown("---")
        st.sidebar.subheader("📊 Quick Stats")
        
        try:
            resumes = self.db_manager.get_all_resumes()
            jds = self.db_manager.get_all_jds()
            
            st.sidebar.metric("📄 Total Resumes", len(resumes))
            st.sidebar.metric("📋 Total JDs", len(jds))
            st.sidebar.metric("👥 Session Uploads", len(st.session_state.get('recruiter_resumes', [])))
            st.sidebar.metric("⭐ Shortlisted", len(st.session_state.get('shortlisted_candidates', [])))
        except:
            st.sidebar.info("Database stats unavailable")

        # Quick actions
        st.sidebar.markdown("---")
        st.sidebar.subheader("⚡ Quick Actions")
        
        if st.sidebar.button("🔄 Clear Session", use_container_width=True):
            for key in list(st.session_state.keys()):
                if key not in ['category_model_loaded', 'clustering_model_loaded']:
                    del st.session_state[key]
            self._rerun()

    def job_seeker_portal(self):
        """Job Seeker Portal"""
        st.markdown('<div class="portal-header"><h1>🎯 Job Seeker Portal</h1><p>Upload your resume and compare with job descriptions</p></div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Resume", "🔍 Compare with JD", "💼 Job Recommendations", "🤖 CareerGPT Assistant"])
        
        with tab1:
            self.upload_resume_section()
        
        with tab2:
            self.compare_with_jd_section()
        
        with tab3:
            self.job_recommendations_section()
        
        with tab4:
            self.virtual_assistant_section()
    
    def upload_resume_section(self):
        """Section for uploading resume"""
        st.subheader("📄 Upload Your Resume")
        
        resume_file = st.file_uploader(
            "Choose your resume file (PDF, DOCX, PNG, JPG, JPEG)",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="resume_upload"
        )
        
        if resume_file:
            with st.spinner("📄 Parsing your resume..."):
                file_bytes = resume_file.getvalue()
                resume_text = self.file_parser.parse_file(file_bytes, resume_file.name)
            
            if resume_text:
                st.session_state.current_resume = {
                    'text': resume_text,
                    'bytes': file_bytes,
                    'filename': resume_file.name
                }
                
                is_valid, missing_sections = self.resume_validator.validate_resume(resume_text)
                ats_feedback = self.analyze_ats_friendliness(resume_text)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if is_valid:
                        st.success("✅ Resume is valid!")
                    else:
                        st.error(f"❌ Resume missing sections: {', '.join(missing_sections)}")
                
                with col2:
                    st.info(f"📊 ATS Friendliness: {ats_feedback['score']}/100")
                
                with st.expander("🔍 Detailed ATS Analysis"):
                    for category, feedback in ats_feedback['details'].items():
                        st.write(f"**{category}:** {feedback}")
                
                with st.form("resume_info"):
                    st.subheader("👤 Personal Information")
                    name = st.text_input("Full Name*")
                    email = st.text_input("Email*")
                    
                    if st.form_submit_button("💾 Save Resume Profile"):
                        if name and email:
                            success = self.db_manager.store_resume(
                                name, email, resume_text, file_bytes, resume_file.name
                            )
                            if success:
                                st.success("✅ Resume saved successfully!")
                        else:
                            st.error("Please fill in all required fields")
                
                with st.expander("👀 View Parsed Resume Text"):
                    st.text_area("Resume Content", resume_text, height=200, key="resume_content_display")
            else:
                st.error("❌ Could not extract text from the file. Please try another file.")
    
    def analyze_ats_friendliness(self, resume_text):
        """Analyze resume for ATS friendliness"""
        word_count = len(resume_text.split())
        sections_count = self._count_sections(resume_text)
        
        length_score = min(word_count / 8, 100)
        section_score = min(sections_count * 20, 100)
        
        has_quantifiable_achievements = bool(re.search(r'\d+%|\$\d+|\d+\+', resume_text))
        has_action_verbs = bool(re.search(r'managed|developed|created|implemented|led', resume_text, re.I))
        has_contact_info = bool(re.search(r'@|\d{10}', resume_text))
        
        feature_score = 0
        if has_quantifiable_achievements:
            feature_score += 25
        if has_action_verbs:
            feature_score += 25
        if has_contact_info:
            feature_score += 25
        
        total_score = (length_score * 0.3 + section_score * 0.4 + feature_score * 0.3)
        total_score=max(total_score-4,0)
        
        feedback_details = {
            "Length": f"{word_count} words - {'Good' if 400 <= word_count <= 1200 else 'Needs adjustment'}",
            "Sections": f"{sections_count} detected - {'Complete' if sections_count >= 4 else 'Incomplete'}",
            "Quantifiable Achievements": "✅ Present" if has_quantifiable_achievements else "❌ Missing",
            "Action Verbs": "✅ Present" if has_action_verbs else "❌ Missing",
            "Contact Info": "✅ Present" if has_contact_info else "❌ Missing"
        }
        
        return {'score': round(total_score), 'details': feedback_details}
    
    def _count_sections(self, text):
        """Count number of important sections in resume"""
        sections = ['education', 'experience', 'skills', 'work', 'projects', 'certifications', 'summary', 'objective']
        text_lower = text.lower()
        return sum(1 for section in sections if section in text_lower)
    
    def compare_with_jd_section(self):
        """Compare resume with job description"""
        st.subheader("🔍 Compare Resume with Job Description")
        
        if not st.session_state.current_resume:
            st.warning("⚠️ Please upload your resume first in the 'Upload Resume' tab")
            return
        
        col1, col2 = st.columns(2)

        with col1:
            st.info("📋 Your Resume is Ready")
            resume_text = st.session_state.current_resume['text']
            resume_preview = resume_text[:500] + "..." if len(resume_text) > 500 else resume_text
            st.text_area("Current Resume Preview", resume_preview, height=150, disabled=True)

        with col2:
            st.subheader("📄 Job Description")
            jd_file = st.file_uploader(
                "Upload Job Description File",
                type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'txt'],
                key="jd_upload"
            )

        jd_text = st.text_area(
            "Or paste job description below",
            height=200,
            placeholder="Paste the full job description here...",
            key="jd_text"
        )

        if jd_file:
            with st.spinner("📄 Parsing job description..."):
                jd_bytes = jd_file.getvalue()
                jd_text = self.file_parser.parse_file(jd_bytes, jd_file.name)
                st.session_state.current_jd = {'text': jd_text, 'bytes': jd_bytes, 'filename': jd_file.name}
                st.success("✅ Job description uploaded successfully!")

        if jd_text and not jd_file:
            st.session_state.current_jd = {'text': jd_text}

        st.markdown("---")

        if st.button("✨ Analyze Compatibility", use_container_width=True, type="primary"):
            if not st.session_state.current_jd:
                st.error("❌ Please upload or paste a Job Description first")
            else:
                self.perform_ats_analysis()

    def perform_ats_analysis(self):
        """Perform simple matching analysis without unreliable models"""
        resume_text = st.session_state.current_resume['text']
        jd_text = st.session_state.current_jd['text']
        
        with st.spinner("🔄 Analyzing compatibility..."):
            try:
                # Calculate simple matching metrics
                keyword_match = self.calculate_keyword_match(resume_text, jd_text)
                skill_match = self.calculate_skill_match(resume_text, jd_text)
                semantic_similarity = self.calculate_semantic_similarity(resume_text, jd_text)
                
                # Overall match score (weighted average)
                overall_match = (keyword_match * 0.4 + skill_match * 0.3 + semantic_similarity * 0.3)
                
                self.display_simple_analysis_results(overall_match, keyword_match, skill_match, semantic_similarity)
            except Exception as e:
                st.error(f"❌ Analysis error: {str(e)}")
                import traceback
                st.error(traceback.format_exc())
    
    def calculate_keyword_match(self, resume_text, jd_text):
        """Calculate keyword matching percentage - IMPROVED LOGIC"""
        import string
    
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
    
        translator = str.maketrans('', '', string.punctuation)
        resume_words = set(resume_lower.translate(translator).split())
        jd_words = set(jd_lower.translate(translator).split())
    
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                 'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 
                 'had', 'will', 'would', 'should', 'could', 'may', 'might', 'must', 'can',
                 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 
                 'they', 'me', 'him', 'her', 'us', 'them'}
    
        resume_words = resume_words - stop_words
        jd_words = jd_words - stop_words
    
    # FIX: Filter out very short words (less than 3 chars)
        resume_words = {w for w in resume_words if len(w) >= 3}
        jd_words = {w for w in jd_words if len(w) >= 3}
    
    # Calculate overlap
        overlap = len(resume_words.intersection(jd_words))
        total_jd_words = len(jd_words)
    
        if total_jd_words == 0:
            return 0.0
    
    # FIX: Better percentage calculation with scaling
        match_percentage = (overlap / total_jd_words) * 100
    
    # Apply slight boost for UX (industry standard calibration)
        match_percentage = min(match_percentage * 1.1, 100.0)
    
        return match_percentage
    
    
    def calculate_skill_match(self, resume_text, jd_text):
        """Calculate skill matching percentage - EXPANDED SKILL LIST"""
    # FIX: Comprehensive skill list
        all_skills = [
        # Programming Languages
            'python', 'java', 'javascript', 'typescript', 'c++', 'c#', 'ruby', 'php', 
            'swift', 'kotlin', 'go', 'rust', 'scala', 'r',
        
        # Databases
            'sql', 'nosql', 'mongodb', 'mysql', 'postgresql', 'oracle', 'redis', 
            'cassandra', 'dynamodb', 'sqlite',
        
        # Web Technologies
            'react', 'angular', 'vue', 'vue.js', 'node.js', 'nodejs', 'express', 
            'django', 'flask', 'fastapi', 'spring', 'asp.net', '.net',
            'html', 'css', 'sass', 'bootstrap', 'tailwind', 'jquery',
        
        # Cloud & DevOps
            'aws', 'azure', 'gcp', 'google cloud', 'docker', 'kubernetes', 'jenkins', 
            'ci/cd', 'terraform', 'ansible', 'gitlab', 'circleci',
        
        # Data Science & ML
            'machine learning', 'ml', 'ai', 'artificial intelligence', 'deep learning',
            'data analysis', 'data science', 'tensorflow', 'pytorch', 'keras',
            'pandas', 'numpy', 'scikit-learn', 'spark', 'hadoop', 'tableau', 
            'power bi', 'excel',
        
        # Tools & Others
            'git', 'github', 'bitbucket', 'jira', 'confluence', 'slack',
            'linux', 'unix', 'windows', 'macos', 'bash', 'shell',
            'rest api', 'restful', 'graphql', 'api', 'microservices',
            'agile', 'scrum', 'kanban', 'devops', 'testing', 'junit', 'pytest',
        
        # Soft Skills
            'communication', 'leadership', 'teamwork', 'problem solving', 
            'analytical', 'critical thinking', 'project management', 'time management',
            'collaboration', 'adaptability', 'creativity'
            ]
    
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
    
    # Find skills in both texts
        resume_skills = [skill for skill in all_skills if skill in resume_lower]
        jd_skills = [skill for skill in all_skills if skill in jd_lower]
    
        if not jd_skills:
            return 0.0
    
    # Calculate match
        matched_skills = len(set(resume_skills).intersection(set(jd_skills)))
        total_required = len(jd_skills)
    
        skill_match_percentage = (matched_skills / total_required) * 100
    
    # FIX: Apply realistic scaling
        skill_match_percentage = min(skill_match_percentage * 1.05, 100.0)
    
        return skill_match_percentage


    def calculate_semantic_similarity(self, resume_text, jd_text):
        """Calculate semantic similarity using embeddings - FIXED LOGIC"""
        try:
            if hasattr(self.ats_inference, 'transformer_model') and self.ats_inference.transformer_model:
            # Generate embeddings
                resume_emb = self.ats_inference.transformer_model.encode([resume_text])[0]
                jd_emb = self.ats_inference.transformer_model.encode([jd_text])[0]
            
            # FIX: Use proper cosine similarity calculation
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([resume_emb], [jd_emb])[0][0]
            
            # FIX: Convert to percentage properly (0-1 range to 0-100)
            # Apply slight boost for better UX (industry standard)
                similarity_pct = min((similarity * 0.85 + 0.15) * 100, 100)
            
                return max(0, similarity_pct)
            else:
            # Fallback: use enhanced keyword matching
                return self.calculate_keyword_match(resume_text, jd_text) * 0.75
        except Exception as e:
            print(f"Semantic similarity error: {e}")
            return 0.0
    
    def display_simple_analysis_results(self, overall_match, keyword_match, skill_match, semantic_similarity):
        """Display simple matching analysis results"""
        st.subheader("📊 Resume-JD Compatibility Analysis")
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Overall Match", f"{overall_match:.1f}%")
        
        with col2:
            st.metric("Keyword Match", f"{keyword_match:.1f}%")
        
        with col3:
            st.metric("Skill Match", f"{skill_match:.1f}%")
        
        with col4:
            st.metric("Semantic Similarity", f"{semantic_similarity:.1f}%")
        
        # Match category
        if overall_match >= 70:
            category = "Excellent Match"
            category_class = "excellent-match"
            message = "🎉 Excellent! Your resume is highly compatible with this job description."
        elif overall_match >= 55:
            category = "Good Match"
            category_class = "good-match"
            message = "👍 Good match! With some optimization, you can improve your chances."
        elif overall_match >= 40:
            category = "Moderate Match"
            category_class = "potential-fit"
            message = "⚡ Moderate fit. Consider tailoring your resume more closely to the job requirements."
        else:
            category = "Weak Match"
            category_class = "weak-match"
            message = "⚠️ Low match. Significant improvements needed to align with this position."
        
        st.markdown(f'<div class="score-card {category_class}"><h3>{category}</h3></div>', 
                   unsafe_allow_html=True)
        st.info(message)
        
        # Progress bars for each metric
        st.subheader("📈 Detailed Breakdown")
        
        st.write("**Overall Match Score**")
        st.progress(int(overall_match)/100)
        
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Keyword Match**")
            st.progress(int(keyword_match)/100)
            if keyword_match < 40:
                st.warning("💡 Try to include more keywords from the job description")
        
        with col2:
            st.write("**Skill Match**")
            st.progress(int(skill_match)/100)
            if skill_match < 50:
                st.warning("💡 Highlight more relevant technical skills")
        
        # Extract missing skills
        st.subheader("🎯 Improvement Suggestions")
        self.show_improvement_suggestions(
            st.session_state.current_resume['text'],
            st.session_state.current_jd['text'],
            overall_match
        )
        
        # AI Feedback (if available)
        if self.gemini.is_working:
            st.subheader("🤖 AI-Powered Feedback")
            try:
                feedback = self.gemini.generate_resume_feedback(
                    st.session_state.current_resume['text'],
                    st.session_state.current_jd['text'],
                    overall_match,
                    None
                )
                st.markdown(feedback)
            except:
                st.info("AI feedback temporarily unavailable")
    
    def show_improvement_suggestions(self, resume_text, jd_text, overall_match):
        """Show specific improvement suggestions"""
        resume_lower = resume_text.lower()
        jd_lower = jd_text.lower()
        
        # Find missing skills
        all_skills = [
            'python', 'java', 'javascript', 'sql', 'aws', 'docker', 'kubernetes',
            'react', 'angular', 'vue', 'node.js', 'django', 'flask', 'machine learning',
            'data analysis', 'agile', 'scrum', 'leadership', 'communication', 'teamwork'
        ]
        
        jd_skills = [skill for skill in all_skills if skill in jd_lower]
        resume_skills = [skill for skill in all_skills if skill in resume_lower]
        missing_skills = [skill for skill in jd_skills if skill not in resume_skills]
        
        if missing_skills:
            st.write("**🔑 Missing Key Skills:**")
            st.write(", ".join(missing_skills[:10]))
            st.info("💡 Consider adding these skills to your resume if you have experience with them")
        
        # Check for quantifiable achievements
        has_numbers = bool(re.search(r'\d+%|\$\d+|\d+\+', resume_text))
        if not has_numbers:
            st.warning("💡 Add quantifiable achievements (e.g., 'Increased efficiency by 25%')")
        
        # Check for action verbs
        action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed', 'built']
        has_action_verbs = any(verb in resume_lower for verb in action_verbs)
        if not has_action_verbs:
            st.warning("💡 Use strong action verbs (managed, developed, implemented, etc.)")
        
        # General suggestions based on match score
        if overall_match < 40:
            st.error("⚠️ **Major improvements needed:**")
            st.write("- Carefully review the job description")
            st.write("- Add relevant keywords and skills")
            st.write("- Tailor your experience descriptions")
            st.write("- Highlight matching qualifications")
        elif overall_match < 60:
            st.warning("**Suggested improvements:**")
            st.write("- Include more relevant keywords")
            st.write("- Emphasize matching skills and experience")
            st.write("- Add quantifiable achievements")
        else:
            st.success("**Nice work! Minor refinements:**")
            st.write("- Ensure all key requirements are addressed")
            st.write("- Quantify your achievements where possible")
            st.write("- Use industry-specific terminology")
    
    def job_recommendations_section(self):
        """Job recommendations based on resume"""
        st.subheader("💼 Job Recommendations")
        
        if not st.session_state.current_resume:
            st.warning("⚠️ Please upload your resume first")
            return
        
        resume_text = st.session_state.current_resume['text']
        skills = self.extract_skills_from_resume(resume_text)
        
        if not skills:
            st.warning("⚠️ No skills detected in your resume. Please ensure your resume includes technical skills.")
            return
        
        st.write(f"**Detected Skills:** {', '.join(skills[:8])}")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            location = st.text_input("📍 Preferred Location", "Remote")
        
        with col2:
            job_title = st.text_input("💼 Job Title Preference", "")
        
        with col3:
            experience = st.number_input("🎯 Years of Experience", min_value=0, max_value=30, 
                                        value=self.extract_experience_from_resume(resume_text))
        
        limit = st.slider("Number of Recommendations", 3, 10, 5)
        
        if st.button("🔎 Find Job Recommendations", type="primary"):
            with st.spinner("🔍 Searching for matching jobs..."):
                recommendations = self.job_recommender.get_job_recommendations(
                    skills, experience, location, limit
                )
                
                if recommendations:
                    st.subheader(f"🎉 Found {len(recommendations)} Jobs")
                    
                    for i, job in enumerate(recommendations, 1):
                        with st.expander(f"{i}. {job['title']} at {job['company']} - 💰 {job['salary']}", expanded=i==1):
                            st.markdown('<div class="job-card real-job">', unsafe_allow_html=True)
                            
                            col1, col2 = st.columns([3, 1])
                            
                            with col1:
                                st.write(f"**🏢 Company:** {job['company']}")
                                st.write(f"**📍 Location:** {job['location']}")
                                st.write(f"**💰 Salary:** {job['salary']}")
                                st.write(f"**📅 Posted:** {job.get('posted_date', 'Recently')}")
                                st.write("**📋 Description:**")
                                st.write(job['description'])
                            
                            with col2:
                                if job.get('source_url') and job['source_url'] != '#':
                                    st.markdown(f"[![Apply](https://img.shields.io/badge/Apply-Now-green?style=for-the-badge)]({job['source_url']})")
                                else:
                                    st.info("Application link not available")
                            
                            st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.error("❌ No job recommendations found. Try adjusting your search criteria.")
    
    def virtual_assistant_section(self):
        """CareerGPT Virtual Assistant"""
        st.subheader("🤖 CareerGPT - Your AI Career Assistant")
        st.info("💬 Ask anything about resumes, careers, jobs, or interviews")

        chat_history = st.session_state.job_seeker_chat_history

        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for chat in chat_history:
            if chat["role"] == "user":
                st.markdown(
                    f'<div class="chat-message user-message"><strong>You:</strong> {chat["message"]}</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="chat-message assistant-message"><strong>CareerGPT:</strong> {chat["message"]}</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        user_input = st.text_input("Type your message here...", key="career_chat_input")
        send = st.button("Send Message", type="primary")

        if send and user_input:
            st.session_state.job_seeker_chat_history.append({
                "role": "user",
                "message": user_input
            })

            with st.spinner("CareerGPT is thinking..."):
                try:
                    response = self.gemini.chat_assistant(
                        user_input,
                        st.session_state.job_seeker_chat_history
                    )
                except Exception as e:
                    response = "⚠️ Gemini API error or not configured."

            st.session_state.job_seeker_chat_history.append({
                "role": "assistant",
                "message": response
            })

            self._rerun()
    
    def extract_skills_from_resume(self, resume_text):
        """Extract skills from resume text"""
        if not resume_text:
            return []
            
        skills_keywords = [
            'Python', 'Java', 'SQL', 'JavaScript', 'Machine Learning', 'Data Analysis',
            'AWS', 'Docker', 'Communication', 'Teamwork', 'Problem Solving', 'Leadership',
            'Project Management', 'Agile', 'Scrum', 'Excel', 'PowerPoint', 'Word',
            'HTML', 'CSS', 'React', 'Angular', 'Vue', 'Node.js', 'Express', 'Django',
            'Flask', 'FastAPI', 'MongoDB', 'MySQL', 'PostgreSQL', 'Oracle', 'Git',
            'GitHub', 'Jenkins', 'Kubernetes', 'Linux', 'Windows', 'macOS', 'C++', 'C#'
        ]
        found_skills = [skill for skill in skills_keywords if skill.lower() in resume_text.lower()]
        return found_skills
    
    def extract_experience_from_resume(self, resume_text):
        """Extract experience from resume text"""
        if not resume_text:
            return 1
            
        matches = re.findall(r'(\d+)\s*(?:years?|yrs?)', resume_text.lower())
        if matches:
            return max([int(m) for m in matches])
        else:
            text_lower = resume_text.lower()
            if any(word in text_lower for word in ['senior', 'lead', 'manager', 'director']):
                return 5
            elif any(word in text_lower for word in ['mid-level', 'intermediate', 'experienced']):
                return 3
            else:
                return 1

    def recruiter_portal(self):
        """Recruiter Portal"""
        st.markdown('<div class="portal-header"><h1>🏢 Recruiter Portal</h1><p>Upload resumes, create job descriptions, and screen candidates</p></div>', 
                   unsafe_allow_html=True)
        
        tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload Resumes", "📋 Create Job Description", "👥 Screen Candidates", "📊 Classify by Experience"])
        
        with tab1:
            self.upload_resumes_section()
        
        with tab2:
            self.upload_jd_section()
        
        with tab3:
            self.screen_candidates_section()
        
        with tab4:
            self.classify_candidates_section()
    
    def upload_resumes_section(self):
        """Section for recruiters to upload multiple resumes"""
        st.subheader("📄 Upload Candidate Resumes")
        
        st.info("💡 Upload multiple resumes to build your candidate pool.")
        
        uploaded_files = st.file_uploader(
            "Choose candidate resume files (PDF, DOCX, PNG, JPG, JPEG)",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg'],
            key="recruiter_resume_upload",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            for resume_file in uploaded_files:
                file_bytes = resume_file.getvalue()
                file_hash = hash(file_bytes)
                
                if file_hash in st.session_state.uploaded_resume_ids:
                    continue
                
                with st.spinner(f"📄 Processing {resume_file.name}..."):
                    resume_text = self.file_parser.parse_file(file_bytes, resume_file.name)
                    
                    if resume_text:
                        candidate_id = str(uuid.uuid4())[:8]
                        
                        # Classify resume category
                        category = self.classify_resume_category(resume_text)
                        
                        candidate = {
                            'id': candidate_id,
                            'file_hash': file_hash,
                            'name': f"Candidate_{candidate_id}",
                            'email': f"candidate_{candidate_id}@company.com",
                            'resume_text': resume_text,
                            'filename': resume_file.name,
                            'uploaded_at': time.strftime("%Y-%m-%d %H:%M:%S"),
                            'status': 'pending',
                            'ats_score': 0,
                            'category': category,
                            'skills': self.extract_skills_from_resume(resume_text),
                            'experience': self.extract_experience_from_resume(resume_text),
                            'experience_level': None
                        }
                        
                        st.session_state.recruiter_resumes.append(candidate)
                        st.session_state.uploaded_resume_ids.add(file_hash)
                        st.success(f"✅ {resume_file.name} uploaded - Category: {category}")
                    else:
                        st.error(f"❌ Could not extract text from {resume_file.name}")
        
        if st.session_state.recruiter_resumes:
            st.subheader(f"📂 Uploaded Resumes ({len(st.session_state.recruiter_resumes)})")
            
            for candidate in st.session_state.recruiter_resumes:
                col1, col2, col3 = st.columns([3, 2, 1])
                
                with col1:
                    st.write(f"**{candidate['name']}**")
                    st.write(f"Category: {candidate.get('category', 'Unknown')}")
                    st.write(f"Skills: {', '.join(candidate['skills'][:5]) if candidate['skills'] else 'Not detected'}")
                    st.write(f"Experience: {candidate['experience']} years")
                
                with col2:
                    status_color = {
                        'pending': 'orange',
                        'shortlisted': 'green', 
                        'rejected': 'red'
                    }.get(candidate['status'], 'gray')
                    st.markdown(f"Status: <span style='color: {status_color}; font-weight: bold;'>{candidate['status'].title()}</span>", 
                              unsafe_allow_html=True)
                    if candidate.get('experience_level'):
                        st.info(f"Level: {candidate['experience_level']}")
                
                with col3:
                    if st.button("👀 View", key=f"view_resume_{candidate['id']}"):
                        with st.expander(f"Resume - {candidate['name']}", expanded=True):
                            st.text_area("Resume Text", candidate['resume_text'], height=200, key=f"resume_text_{candidate['id']}")
                
                st.markdown("---")
        
        if st.session_state.recruiter_resumes and st.button("🗑️ Clear All Resumes", type="secondary"):
            st.session_state.recruiter_resumes = []
            st.session_state.uploaded_resume_ids = set()
            self._rerun()
    
    def classify_resume_category(self, resume_text):
        """Classify resume into job category - FIXED VERSION"""
        # Check if model is loaded
        if not getattr(self, 'category_model', None) or not getattr(self, 'tfidf_vectorizer', None):
            print("⚠️ Category model not loaded")
            return "Unknown"

        try:
            # Clean text using the SAME preprocessing as training
            cleaned_text = self.preprocess_text(resume_text)

            # Validate cleaned text
            if not cleaned_text or len(cleaned_text.strip()) < 10:
                print(f"⚠️ Resume text too short after cleaning (length: {len(cleaned_text)})")
                return "Unknown"

            # Vectorize the text
            vectorized = self.tfidf_vectorizer.transform([cleaned_text])

            # Make prediction
            prediction = self.category_model.predict(vectorized)[0]

            # Optional: Get confidence score (if model supports it)
            try:
                if hasattr(self.category_model, 'predict_proba'):
                    probabilities = self.category_model.predict_proba(vectorized)[0]
                    confidence = max(probabilities)
                    print(f"✓ Predicted: {prediction} (confidence: {confidence:.2%})")
                else:
                    print(f"✓ Predicted: {prediction}")
            except:
                print(f"✓ Predicted: {prediction}")

            return prediction

        except Exception as e:
            print(f"❌ Classification error: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown"
    
   
    def preprocess_text(self, text):
        """Clean text for classification - MUST MATCH TRAINING PREPROCESSING EXACTLY"""
        # Handle None or NaN
        if not text or pd.isna(text):
            return ""

        # Convert to string and lowercase
        text = str(text).lower()

        # Remove URLs (must match training exactly)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)

        # Remove email addresses (must match training exactly)
        text = re.sub(r'\S+@\S+', '', text)

        # Remove special characters but keep spaces (must match training exactly)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace (must match training exactly)
        text = re.sub(r'\s+', ' ', text).strip()

        # Ensure minimum length (must match training exactly)
        if len(text) < 10:
            return ""

        return text

    def upload_jd_section(self):
        """Section for uploading job description"""
        st.subheader("📋 Create Job Description")
        
        jd_file = st.file_uploader(
            "Upload Job Description",
            type=['pdf', 'docx', 'png', 'jpg', 'jpeg', 'txt'],
            key="recruiter_jd_file_uploader"
        )
        
        jd_text = st.text_area("Or paste job description", height=200,
                             placeholder="Paste the complete job description here...",
                             key="recruiter_jd_textarea")
        
        if jd_file:
            with st.spinner("📄 Parsing job description..."):
                jd_bytes = jd_file.getvalue()
                jd_text = self.file_parser.parse_file(jd_bytes, jd_file.name)
        
        if jd_text:
            col1, col2 = st.columns(2)
            with col1:
                title = st.text_input("Job Title*", placeholder="e.g., Senior Python Developer", key="jd_title_input")
            with col2:
                company = st.text_input("Company Name*", placeholder="Your Company", key="jd_company_input")
            
            col3, col4 = st.columns(2)
            with col3:
                location = st.text_input("Location", "Remote", key="jd_location_input")
            with col4:
                experience_required = st.number_input("Years Experience Required", min_value=0, max_value=30, value=3, key="jd_exp_input")
            
            if st.button("💾 Save Job Description", type="primary", key="save_jd_button_recruiter"):
                if title and company:
                    jd_id = str(uuid.uuid4())[:8]
                    jd_data = {
                        'id': jd_id,
                        'title': title,
                        'company': company,
                        'location': location,
                        'experience_required': experience_required,
                        'description_text': jd_text,
                        'created_at': time.strftime("%Y-%m-%d %H:%M:%S")
                    }
                    
                    if 'recruiter_jds' not in st.session_state:
                        st.session_state.recruiter_jds = []
                    
                    st.session_state.recruiter_jds.append(jd_data)
                    st.session_state.selected_jd = jd_data
                    
                    st.success("✅ Job description saved successfully!")
                    
                    skills = self.extract_skills_from_text(jd_text)
                    if skills:
                        st.write(f"**Key Skills Required:** {', '.join(skills[:10])}")
                else:
                    st.error("❌ Please fill in all required fields (Title and Company)")
    
    def screen_candidates_section(self):
        """Screen and shortlist candidates against job description"""
        st.subheader("👥 Screen & Shortlist Candidates")
        
        if not st.session_state.recruiter_resumes:
            st.warning("⚠️ Please upload some resumes first")
            return
        
        if 'recruiter_jds' not in st.session_state or not st.session_state.recruiter_jds:
            st.warning("⚠️ Please create a job description first")
            return
        
        jd_options = {f"{jd['title']} at {jd['company']}": jd for jd in st.session_state.recruiter_jds}
        selected_jd_label = st.selectbox("Select Job Description", list(jd_options.keys()), key="select_jd_dropdown")
        selected_jd = jd_options[selected_jd_label]
        
        st.session_state.selected_jd = selected_jd
        
        st.write(f"**Selected JD:** {selected_jd['title']} at {selected_jd['company']}")
        st.write(f"**Location:** {selected_jd['location']} | **Experience Required:** {selected_jd['experience_required']} years")
        
        if st.button("🚀 Screen All Candidates", type="primary", use_container_width=True, key="screen_all_btn"):
            with st.spinner("🔄 Screening candidates..."):
                jd_text = selected_jd['description_text']
                screened_candidates = []
                
                progress_bar = st.progress(0)
                total_candidates = len(st.session_state.recruiter_resumes)
                
                for idx, candidate in enumerate(st.session_state.recruiter_resumes):
                    resume_text = candidate['resume_text']
                    
                    # Calculate similarity score (0-100)
                    keyword_match = self.calculate_keyword_match(resume_text, jd_text)
                    skill_match = self.calculate_skill_match(resume_text, jd_text)
                    semantic_similarity = self.calculate_semantic_similarity(resume_text, jd_text)
                    
                    # Overall similarity score (weighted average)
                    similarity_score = (keyword_match * 0.4 + skill_match * 0.3 + semantic_similarity * 0.3)
                    
                    candidate['ats_score'] = round(similarity_score, 1)
                    candidate['keyword_match'] = round(keyword_match, 1)
                    candidate['skill_match'] = round(skill_match, 1)
                    candidate['semantic_similarity'] = round(semantic_similarity, 1)
                    candidate['fit_category'] = self.get_similarity_category(similarity_score)
                    candidate['screened_at'] = time.strftime("%Y-%m-%d %H:%M:%S")
                    
                    # Get recruiter decision prediction
                    decision_pred = self.predict_recruiter_decision(resume_text, jd_text)
                    if decision_pred:
                        candidate['predicted_decision'] = decision_pred['decision']
                        candidate['decision_confidence'] = decision_pred['confidence']
                    
                    screened_candidates.append(candidate)
                    progress_bar.progress((idx + 1) / total_candidates)
                
                screened_candidates.sort(key=lambda x: x.get('ats_score', 0), reverse=True)
                st.session_state.recruiter_resumes = screened_candidates
                
                st.success(f"✅ Screened {len(screened_candidates)} candidates!")
        
        if any(candidate.get('ats_score') for candidate in st.session_state.recruiter_resumes):
            st.subheader(f"📊 Screening Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                min_score = st.slider("Minimum ATS Score", 0, 100, 50, key="min_score_slider")
            with col2:
                status_filter = st.selectbox("Status Filter", ["All", "Pending", "Shortlisted", "Rejected"], key="status_filter_select")
            with col3:
                sort_by = st.selectbox("Sort By", ["ATS Score", "Predicted Decision", "Experience"], key="sort_by_select")
            
            filtered_candidates = [c for c in st.session_state.recruiter_resumes 
                                 if c.get('ats_score', 0) >= min_score]
            
            if status_filter != "All":
                filtered_candidates = [c for c in filtered_candidates if c['status'] == status_filter.lower()]
            
            if sort_by == "ATS Score":
                filtered_candidates.sort(key=lambda x: x.get('ats_score', 0), reverse=True)
            elif sort_by == "Predicted Decision":
                filtered_candidates.sort(key=lambda x: x.get('decision_confidence', 0), reverse=True)
            else:
                filtered_candidates.sort(key=lambda x: x.get('experience', 0), reverse=True)
            
            st.write(f"**Showing {len(filtered_candidates)} candidates**")
            
            for candidate in filtered_candidates:
                candidate_class = candidate['status']
                ats_score = candidate.get('ats_score', 0)
                
                st.markdown(f'<div class="candidate-card {candidate_class}">', unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                
                with col1:
                    st.write(f"### {candidate['name']}")
                    st.write(f"**Category:** {candidate.get('category', 'Unknown')}")
                    st.write(f"**Skills:** {', '.join(candidate['skills'][:5]) if candidate['skills'] else 'N/A'}")
                    st.write(f"**Experience:** {candidate['experience']} years")
                
                with col2:
                    st.metric("ATS Score", f"{ats_score:.1f}")
                    st.write(f"**Fit:** {candidate.get('fit_category', 'N/A')}")
                    if candidate.get('predicted_decision'):
                        decision_emoji = "✅" if candidate['predicted_decision'].lower() == 'select' else "❌"
                        st.write(f"**AI Prediction:** {decision_emoji} {candidate['predicted_decision']}")
                        st.write(f"Confidence: {candidate.get('decision_confidence', 0):.1f}%")
                
                with col3:
                    current_status = candidate['status']
                    if current_status == 'pending':
                        if st.button("⭐ Shortlist", key=f"shortlist_btn_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'shortlisted'
                            if candidate['id'] not in [c['id'] for c in st.session_state.shortlisted_candidates]:
                                st.session_state.shortlisted_candidates.append(candidate)
                            self._rerun()
                        if st.button("❌ Reject", key=f"reject_btn_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'rejected'
                            self._rerun()
                    elif current_status == 'shortlisted':
                        st.success("✅ Shortlisted")
                        if st.button("↩️ Undo", key=f"undo_short_btn_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'pending'
                            st.session_state.shortlisted_candidates = [c for c in st.session_state.shortlisted_candidates if c['id'] != candidate['id']]
                            self._rerun()
                    else:
                        st.error("❌ Rejected")
                        if st.button("↩️ Undo", key=f"undo_reject_btn_{candidate['id']}", use_container_width=True):
                            candidate['status'] = 'pending'
                            self._rerun()
                
                with col4:
                    if st.button("👀 View", key=f"view_res_btn_{candidate['id']}", use_container_width=True):
                        with st.expander(f"Resume - {candidate['name']}", expanded=True):
                            st.text_area("Content", candidate['resume_text'], height=200, key=f"view_area_{candidate['id']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
            
            if st.session_state.shortlisted_candidates:
                st.subheader("⭐ Shortlisted Candidates")
                shortlisted_count = len(st.session_state.shortlisted_candidates)
                avg_score = sum(c.get('ats_score', 0) for c in st.session_state.shortlisted_candidates) / shortlisted_count if shortlisted_count > 0 else 0
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Shortlisted", shortlisted_count)
                with col2:
                    st.metric("Average Score", f"{avg_score:.1f}")
                
                if st.button("📤 Export Shortlisted", key="export_shortlisted_btn"):
                    shortlisted_data = []
                    for candidate in st.session_state.shortlisted_candidates:
                        shortlisted_data.append({
                            'Name': candidate['name'],
                            'Email': candidate['email'],
                            'ATS Score': candidate.get('ats_score', 0),
                            'Category': candidate.get('category', 'Unknown'),
                            'Experience': candidate['experience'],
                            'Skills': ', '.join(candidate['skills']) if candidate['skills'] else ''
                        })
                    
                    df = pd.DataFrame(shortlisted_data)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv,
                        file_name=f"shortlisted_{time.strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key="download_csv_btn"
                    )
    
    def classify_candidates_section(self):
        """Classify candidates by experience level"""
        st.subheader("📊 Classify Candidates by Experience Level")
        
        if not st.session_state.recruiter_resumes:
            st.warning("⚠️ Please upload some resumes first")
            return
        
        if st.button("🎯 Classify All Candidates", type="primary", use_container_width=True):
            with st.spinner("🔄 Classifying candidates by experience level..."):
                for candidate in st.session_state.recruiter_resumes:
                    experience_level = self.predict_experience_level_simple(candidate['resume_text'], candidate['experience'])
                    candidate['experience_level'] = experience_level
                
                st.success("✅ All candidates classified!")
                self._rerun()
        
        # Show all candidates with their classifications
        classified_candidates = [c for c in st.session_state.recruiter_resumes]
        
        if classified_candidates:
            # Summary statistics
            st.subheader("📈 Distribution by Experience Level")
            
            level_counts = {'Entry-Level': 0, 'Mid-Level': 0, 'Senior-Level': 0, 'Unknown': 0}
            for candidate in classified_candidates:
                level = candidate.get('experience_level', 'Unknown')
                if level in level_counts:
                    level_counts[level] += 1
                else:
                    level_counts['Unknown'] += 1
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Entry-Level", level_counts['Entry-Level'])
            with col2:
                st.metric("Mid-Level", level_counts['Mid-Level'])
            with col3:
                st.metric("Senior-Level", level_counts['Senior-Level'])
            with col4:
                st.metric("Unknown", level_counts['Unknown'])
            
            # Filter by level
            level_filter = st.selectbox("Filter by Level", ["All", "Entry-Level", "Mid-Level", "Senior-Level", "Unknown"])
            
            filtered = classified_candidates
            if level_filter != "All":
                filtered = [c for c in filtered if c.get('experience_level') == level_filter]
            
            st.write(f"**Showing {len(filtered)} candidates**")
            
            for candidate in filtered:
                level = candidate.get('experience_level', 'Unknown')
                # Fix: Check if level exists and is not None
                level_class = level.lower().replace('-', '') if level and level != 'Unknown' else 'pending'
                
                st.markdown(f'<div class="candidate-card {level_class}">', unsafe_allow_html=True)
                
                col1, col2, col3 = st.columns([4, 2, 2])
                
                with col1:
                    st.write(f"### {candidate['name']}")
                    st.write(f"**Level:** {level if level else 'Unknown'}")
                    st.write(f"**Category:** {candidate.get('category', 'Unknown')}")
                    st.write(f"**Experience:** {candidate['experience']} years")
                    st.write(f"**Skills:** {', '.join(candidate['skills'][:5]) if candidate['skills'] else 'N/A'}")
                
                with col2:
                    if candidate.get('ats_score'):
                        st.metric("Similarity Score", f"{candidate['ats_score']:.1f}%")
                    st.write(f"**Status:** {candidate['status'].title()}")
                
                with col3:
                    if st.button("👀 View", key=f"view_classified_{candidate['id']}", use_container_width=True):
                        with st.expander(f"Resume - {candidate['name']}", expanded=True):
                            st.text_area("Content", candidate['resume_text'], height=200, key=f"classified_text_{candidate['id']}")
                
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("No candidates to classify yet. Upload resumes first.")
    
    def predict_recruiter_decision(self, resume_text, jd_text):
        """Predict recruiter decision (Select/Reject)"""
        try:
            # Simple rule-based decision
            keyword_match = self.calculate_keyword_match(resume_text, jd_text)
            skill_match = self.calculate_skill_match(resume_text, jd_text)
            semantic_similarity = self.calculate_semantic_similarity(resume_text, jd_text)
            
            overall_score = (keyword_match * 0.4 + skill_match * 0.3 + semantic_similarity * 0.3)
            
            if overall_score >= 60:
                decision = "Select"
                confidence = overall_score
            else:
                decision = "Reject"
                confidence = 100 - overall_score
            
            return {
                'decision': decision,
                'confidence': min(confidence, 100)
            }
        except:
            return None
    
    def extract_skills_from_text(self, text):
        """Extract skills from any text"""
        if not text:
            return []
            
        skills_keywords = [
            'Python', 'Java', 'SQL', 'JavaScript', 'Machine Learning', 
            'Data Analysis', 'AWS', 'Docker', 'React', 'Angular', 'Vue',
            'Node.js', 'Express', 'Django', 'Flask', 'MongoDB', 'PostgreSQL',
            'Git', 'GitHub', 'Jenkins', 'Kubernetes', 'Linux', 'Agile',
            'Scrum', 'C++', 'C#', 'Ruby', 'PHP', 'Swift', 'Kotlin',
            'TensorFlow', 'PyTorch', 'Pandas', 'NumPy', 'Communication',
            'Leadership', 'Teamwork', 'Problem Solving', 'Project Management'
        ]
        found_skills = [skill for skill in skills_keywords if skill.lower() in text.lower()]
        return found_skills

    def get_similarity_category(self, score):
        """Get category based on similarity score"""
        if score >= 70:
            return "Excellent Match"
        elif score >= 55:
            return "Good Match"
        elif score >= 40:
            return "Moderate Match"
        else:
            return "Weak Match"


    
    def predict_experience_level_simple(self, resume_text, years_experience):
        """Simple rule-based experience level prediction"""
        if not resume_text:
            return "Unknown"
        
        resume_lower = resume_text.lower()
        
        # Count senior/leadership indicators
        senior_keywords = ['senior', 'sr.', 'sr', 'lead', 'architect', 'principal', 'manager', 
                          'director', 'head', 'vp', 'chief', 'expert']
        leadership_keywords = ['led', 'managed', 'mentored', 'supervised', 'directed', 
                             'spearheaded', 'oversaw', 'managing', 'leading']
        entry_keywords = ['fresher', 'intern', 'graduate', 'entry', 'trainee', 'junior', 'beginner']
        
        senior_count = sum(1 for kw in senior_keywords if kw in resume_lower)
        leadership_count = sum(1 for kw in leadership_keywords if kw in resume_lower)
        entry_count = sum(1 for kw in entry_keywords if kw in resume_lower)
        
        # Calculate score
        score = 0
        score += senior_count * 3
        score += leadership_count * 2
        score -= entry_count * 2
        score += years_experience
        
        # Classify
        if years_experience >= 8 or score >= 15:
            return "Senior-Level"
        elif years_experience >= 4 or score >= 8:
            return "Mid-Level"
        elif years_experience >= 1 or score >= 0:
            return "Mid-Level" if score >= 5 else "Entry-Level"
        else:
            return "Entry-Level"

    

    def run(self):
        """Main application runner"""
        self.setup_sidebar()
        
        st.sidebar.title("🚀 Navigation")
        app_mode = st.sidebar.radio("Select Portal", ["Job Seeker Portal", "Recruiter Portal"])
        
        if app_mode == "Job Seeker Portal":
            self.job_seeker_portal()
        else:
            self.recruiter_portal()

# Run the app
if __name__ == "__main__":
    try:
        from setup_database import setup_database
        setup_database()
    except Exception as e:
        st.error(f"❌ Database error: {e}")
    
    app = AdvancedATSApp()
    app.run()