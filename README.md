🎯 Advanced ATS — AI-Powered Resume Screening & Job Matching

📋 Overview

Advanced ATS is an intelligent, AI-driven Applicant Tracking System that helps
job seekers and recruiters make fast, data-driven hiring decisions. It combines
machine learning, NLP, and embeddings to score resumes, classify roles, detect
experience levels, recommend jobs, and predict recruiter decisions.

✨Key Capabilities

ATS resume scoring (0–100) with explainable breakdowns
Automatic resume classification into job categories
Experience-level detection (Entry / Mid / Senior) via clustering
Job recommendations filtered by skills, experience, location
AI career assistant for resume feedback and guidance
Recruiter decision prediction (hire / reject prioritization)

🚀Features 

Intelligent Resume Scoring: Keyword + semantic scoring; ML refinement
Resume Classification: TF-IDF + optimized classifiers for 25 categories
Experience Detection: Engineered features, PCA, K-Means clustering
Job Recommender: Real-time job search via SerpAPI integration
AI Career Assistant: Google Gemini-based feedback & suggestions
Recruiter Prediction: Uses embeddings to model historical decisions

🔧Quick Tech Stack

ML / AI: scikit-learn, SentenceTransformers, NumPy, Pandas
LLM / Feedback: Google Gemini 
Web / UI: Streamlit
DB: SQLite
APIs: SerpAPI (jobs), Google Generative AI (feedback)

🎯Getting Started

Prerequisites
Python 3.8+
pip
Clone & Install
git clone https://github.com/jeevikar14/AdvAts
cd AdvAts
pip install -r requirements.txt

🔐Environment

Create a .env file in the project root with keys:
GEMINI_API_KEY=your_gemini_api_key_here
SERPAPI_KEY=your_serpapi_key_here

📥Initialize DB & Train 
python setup_database.py
python train_ats_score.py
python resume_classifier.py
python experience_classifier.py
python train_recruitor_decision_model.py

▶️Run App

streamlit run app.py

App available at ➡️http://localhost:8501.

🔍How It Works

1️⃣ATS Score Calculation

Keyword Matching (40%): skills, experience, education, completeness.
Semantic Similarity (60%): embeddings via all-MiniLM-L6-v2.
Final score: weighted features → Random Forest regression predicts 0–100.

2️⃣Resume Classification

Preprocess text, TF-IDF (top 3k features), optimized ML classifier,
Hyperparameter tuning with GridSearchCV.

3️⃣Experience Level Detection

Extract years from text, build 12+ features, PCA reduction,
K-Means clustering into Entry / Mid / Senior.

4️⃣Job Recommendation

Query SerpAPI, filter matches by skills, experience, and location.

5️⃣Recruiter Decision Prediction

Use embeddings (resumes, JDs, transcripts) to train a classifier that
Predicts likely hire/reject outcomes to help prioritization.

📊Model Performance 

ATS Scoring: R² ≈ 77.6%, MAE ≈ 4.59(Random Forest Regressor)

Resume Classifier: Accuracy ≈ 99%, F1 ≈ 0.99 (LogReg)

Experience Clustering: Silhouette ≈ 0.41 (PCA + K-Means)


🎯Use Cases

Job Seekers: instant ATS score, actionable feedback, role matching.
Recruiters: automated screening, candidate ranking, experience filtering.

🧪Testing

Run full system validation:

python test_system.py

Tests include imports, inference engine, Gemini integration, recommender,
and database checks.

🤖API Keys & Fallbacks

Google Gemini: optional; fallback → rule-based feedback if absent.

SerpAPI: optional; fallback → local job examples and setup instructions.

Get keys from respective provider dashboards and place them in .env.

📌Notes

Resume DB in database/ats_db.sqlite.
Uploaded files saved in uploads/.
Predictions run locally except optional external API calls.

📄Acknowledgments

Dataset: UpdatedResumeDataset.csv 
Sentence Transformers by UKPLab
Google Gemini AI
SerpAPI for job search


