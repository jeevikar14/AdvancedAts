"""
TRAIN RECRUITER DECISION MODEL
Uses dataset.csv with Interview Transcripts to predict hiring decisions
Fixed for exact column names: Resume, Job_Description, decision, Transcript
"""
import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, classification_report, 
                             confusion_matrix)
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Enhanced Feature Engineering ----------------------
def extract_interview_features(transcript):
    """Extract features from interview transcript"""
    features = []
    
    if not transcript or pd.isna(transcript):
        return np.zeros(8)
    
    transcript_lower = str(transcript).lower()
    
    # 1. Transcript length (normalized)
    words = transcript_lower.split()
    features.append(min(len(words) / 500, 10))
    
    # 2. Positive sentiment indicators
    positive_words = ['yes', 'absolutely', 'definitely', 'excellent', 'great', 
                     'successfully', 'achieved', 'completed', 'improved', 'led']
    positive_count = sum(1 for word in positive_words if word in transcript_lower)
    features.append(min(positive_count / 10, 5))
    
    # 3. Technical depth indicators
    technical_terms = ['algorithm', 'architecture', 'design', 'implement', 
                      'optimize', 'scalability', 'performance', 'testing']
    tech_count = sum(1 for term in technical_terms if term in transcript_lower)
    features.append(min(tech_count / 5, 5))
    
    # 4. Communication quality (sentence structure)
    sentences = transcript_lower.split('.')
    avg_sentence_len = np.mean([len(s.split()) for s in sentences if s.strip()])
    features.append(min(avg_sentence_len / 15, 3))
    
    # 5. Question engagement (looking for answers)
    question_indicators = transcript_lower.count('?')
    features.append(min(question_indicators / 10, 3))
    
    # 6. Confidence indicators
    confidence_words = ['confident', 'experienced', 'proficient', 'expert', 'skilled']
    confidence_count = sum(1 for word in confidence_words if word in transcript_lower)
    features.append(min(confidence_count / 3, 3))
    
    # 7. Problem-solving mentions
    problem_solving = ['solved', 'resolved', 'addressed', 'tackled', 'overcame', 'handled']
    ps_count = sum(1 for word in problem_solving if word in transcript_lower)
    features.append(min(ps_count / 3, 3))
    
    # 8. Team collaboration mentions
    team_words = ['team', 'collaborate', 'cooperation', 'together', 'group']
    team_count = sum(1 for word in team_words if word in transcript_lower)
    features.append(min(team_count / 5, 3))
    
    return np.array(features)

def extract_resume_features(resume_text, jd_text):
    """Extract features from resume relative to job description"""
    features = []
    
    if not resume_text or not jd_text or pd.isna(resume_text) or pd.isna(jd_text):
        return np.zeros(15)
    
    resume_lower = str(resume_text).lower()
    jd_lower = str(jd_text).lower()
    
    # 1. Resume length ratio
    resume_len = len(resume_lower.split())
    jd_len = len(jd_lower.split())
    features.append(resume_len / max(jd_len, 1))
    
    # 2-3. Keyword overlap
    resume_words = set(resume_lower.split())
    jd_words = set(jd_lower.split())
    overlap = len(resume_words.intersection(jd_words))
    features.append(overlap)
    features.append(overlap / max(len(jd_words), 1))
    
    # 4. Action verbs count
    action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed',
                    'built', 'improved', 'increased', 'reduced', 'achieved', 'delivered',
                    'launched', 'established', 'optimized', 'executed', 'coordinated',
                    'architected', 'spearheaded', 'drove']
    action_count = sum(1 for verb in action_verbs if verb in resume_lower)
    features.append(min(action_count, 15))
    
    # 5. Tech skills match
    tech_skills = ['python', 'java', 'sql', 'javascript', 'machine learning', 'aws', 
                   'docker', 'kubernetes', 'react', 'angular', 'django', 'flask', 
                   'tensorflow', 'pytorch', 'data science', 'agile', 'git']
    resume_tech = [s for s in tech_skills if s in resume_lower]
    jd_tech = [s for s in tech_skills if s in jd_lower]
    features.append(len(set(resume_tech).intersection(set(jd_tech))) / max(len(jd_tech), 1) if jd_tech else 0)
    
    # 6-8. Section presence
    for section in ['experience', 'education', 'skills']:
        features.append(1 if section in resume_lower else 0)
    
    # 9. Years of experience
    exp_matches = [int(m) for m in pd.Series([resume_lower]).str.extractall(r'(\d+)\s*(?:years?|yrs?)')[0]]
    features.append(max(exp_matches) if exp_matches else 0)
    
    # 10. Education level
    edu_score = 0
    if 'phd' in resume_lower or 'doctorate' in resume_lower:
        edu_score = 4
    elif 'master' in resume_lower or 'mba' in resume_lower:
        edu_score = 3
    elif 'bachelor' in resume_lower:
        edu_score = 2
    elif 'diploma' in resume_lower or 'associate' in resume_lower:
        edu_score = 1
    features.append(edu_score)
    
    # 11. Quantifiable achievements
    has_numbers = len([w for w in resume_lower.split() if any(c.isdigit() for c in w)])
    features.append(min(has_numbers / max(resume_len/50, 1), 10))
    
    # 12. Contact info completeness
    features.append(1 if '@' in resume_lower else 0)
    
    # 13. Certifications mentioned
    cert_keywords = ['certified', 'certification', 'certificate', 'license']
    features.append(1 if any(word in resume_lower for word in cert_keywords) else 0)
    
    # 14. Projects mentioned
    features.append(1 if 'project' in resume_lower else 0)
    
    # 15. Leadership indicators
    leadership_words = ['lead', 'manage', 'mentor', 'supervise', 'direct']
    leadership_count = sum(1 for word in leadership_words if word in resume_lower)
    features.append(min(leadership_count, 5))
    
    return np.array(features)

# ---------------------- Generate or Load Cached Embeddings ----------------------
def get_or_generate_embeddings(texts, cache_file):
    """Generate embeddings or load from cache"""
    if os.path.exists(cache_file):
        try:
            print(f"📦 Loading cached embeddings from {cache_file}...")
            embeddings = np.load(cache_file)
            if len(embeddings) == len(texts):
                print(f"✅ Embeddings loaded from cache!")
                return embeddings
            else:
                print(f"⚠️ Cache size mismatch. Regenerating...")
        except Exception as e:
            print(f"⚠️ Cache load error: {e}. Regenerating...")
    
    print(f"🧠 Generating embeddings for {len(texts)} texts...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    
    np.save(cache_file, embeddings)
    print(f"💾 Embeddings cached to {cache_file}")
    
    return embeddings

# ---------------------- Main Training Function ----------------------
def train_recruiter_decision_model():
    DATASET_PATH = 'dataset.csv'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'recruiter_decision_model.pkl')
    CACHE_DIR = 'embeddings_cache'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    print("=" * 80)
    print("🎯 RECRUITER DECISION PREDICTOR - TRAINING")
    print("=" * 80)
    
    # ---------------------- Load Dataset ----------------------
    print("\n📂 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    print(f"✅ Dataset loaded: {len(df)} records")
    print(f"   Columns: {list(df.columns)}")
    
    # ---------------------- Use Exact Column Names ----------------------
    print("\n🔍 Using dataset columns...")
    
    # Exact column names from your dataset
    resume_col = 'Resume'
    jd_col = 'Job_Description'
    decision_col = 'decision'  # lowercase 'd' - FIXED!
    transcript_col = 'Transcript'
    
    print(f"✅ Columns:")
    print(f"   Resume: '{resume_col}'")
    print(f"   Job Description: '{jd_col}'")
    print(f"   Decision: '{decision_col}'")
    print(f"   Transcript: '{transcript_col}'")
    
    # Validate required columns exist
    missing_cols = []
    for col in [resume_col, jd_col, decision_col]:
        if col not in df.columns:
            missing_cols.append(col)
    
    if missing_cols:
        print(f"\n❌ ERROR: Missing columns: {missing_cols}")
        print(f"Available columns: {list(df.columns)}")
        return
    
    # Check if transcript exists
    if transcript_col not in df.columns:
        print(f"   ⚠️ '{transcript_col}' column not found - will use empty strings")
        df[transcript_col] = ''
    
    # ---------------------- Data Cleaning ----------------------
    print("\n🧹 Cleaning data...")
    
    # Drop rows with missing critical fields - USING CORRECT COLUMN NAME
    initial_size = len(df)
    df = df.dropna(subset=[resume_col, jd_col, decision_col])
    print(f"   Removed {initial_size - len(df)} rows with missing critical data")
    
    # Fill missing transcripts with empty string
    df[transcript_col] = df[transcript_col].fillna('')
    
    # Encode decision labels
    label_encoder = LabelEncoder()
    df['Decision_Encoded'] = label_encoder.fit_transform(df[decision_col])
    
    # Check class distribution
    print(f"\n📊 Class Distribution:")
    print(df[decision_col].value_counts())
    print(f"   Ratio: {df[decision_col].value_counts(normalize=True).to_dict()}")
    
    # ---------------------- Feature Engineering ----------------------
    print("\n🔧 Extracting features...")
    
    # Extract handcrafted features
    print("   Extracting interview features...")
    interview_features = np.array([extract_interview_features(t) for t in df[transcript_col]])
    
    print("   Extracting resume features...")
    resume_features = np.array([extract_resume_features(r, jd) 
                               for r, jd in zip(df[resume_col], df[jd_col])])
    
    # Generate embeddings with caching
    print("\n🧠 Generating semantic embeddings...")
    
    resume_embs = get_or_generate_embeddings(
        df[resume_col].tolist(), 
        os.path.join(CACHE_DIR, 'resume_embs_decision.npy')
    )
    
    jd_embs = get_or_generate_embeddings(
        df[jd_col].tolist(),
        os.path.join(CACHE_DIR, 'jd_embs_decision.npy')
    )
    
    transcript_embs = get_or_generate_embeddings(
        df[transcript_col].tolist(),
        os.path.join(CACHE_DIR, 'transcript_embs_decision.npy')
    )
    
    # PCA for dimensionality reduction
    print("\n🔬 Applying PCA...")
    n_components = min(25, len(df) // 200)
    
    pca_resume = PCA(n_components=n_components, random_state=42)
    pca_jd = PCA(n_components=n_components, random_state=42)
    pca_transcript = PCA(n_components=n_components, random_state=42)
    
    resume_reduced = pca_resume.fit_transform(resume_embs)
    jd_reduced = pca_jd.fit_transform(jd_embs)
    transcript_reduced = pca_transcript.fit_transform(transcript_embs)
    
    print(f"   Resume variance retained: {pca_resume.explained_variance_ratio_.sum():.1%}")
    print(f"   JD variance retained: {pca_jd.explained_variance_ratio_.sum():.1%}")
    print(f"   Transcript variance retained: {pca_transcript.explained_variance_ratio_.sum():.1%}")
    
    # Cosine similarities
    print("\n🔧 Computing similarity features...")
    resume_jd_sim = np.array([cosine_similarity([resume_embs[i]], [jd_embs[i]])[0][0] 
                              for i in range(len(df))]).reshape(-1, 1)
    
    resume_transcript_sim = np.array([cosine_similarity([resume_embs[i]], [transcript_embs[i]])[0][0] 
                                     for i in range(len(df))]).reshape(-1, 1)
    
    # ---------------------- Combine All Features ----------------------
    X_combined = np.hstack([
        resume_reduced,
        jd_reduced,
        transcript_reduced,
        resume_jd_sim,
        resume_transcript_sim,
        interview_features,
        resume_features
    ])
    
    y = df['Decision_Encoded'].values
    
    print(f"\n✅ Feature matrix: {X_combined.shape}")
    print(f"   Total features: {X_combined.shape[1]}")
    print(f"   Samples: {X_combined.shape[0]}")
    
    # ---------------------- Train-Test Split ----------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Train-Test Split:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Test: {len(X_test)} samples")
    
    # ---------------------- Scaling ----------------------
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------- Model Training ----------------------
    print("\n🤖 Training classification models...")
    print("=" * 80)
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=2,
            reg_alpha=0.05,
            reg_lambda=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=3,
            min_samples_split=6,
            random_state=42
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_leaf=2,
            min_samples_split=5,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
    }
    
    best_model = None
    best_score = 0
    best_name = None
    results = {}
    
    for name, model_inst in models.items():
        print(f"\n{'─' * 80}")
        print(f"📊 Training {name}...")
        
        # Train
        model_inst.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = model_inst.predict(X_train_scaled)
        y_test_pred = model_inst.predict(X_test_scaled)
        y_test_proba = model_inst.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        train_acc = accuracy_score(y_train, y_train_pred)
        test_acc = accuracy_score(y_test, y_test_pred)
        precision = precision_score(y_test, y_test_pred, average='binary')
        recall = recall_score(y_test, y_test_pred, average='binary')
        f1 = f1_score(y_test, y_test_pred, average='binary')
        roc_auc = roc_auc_score(y_test, y_test_proba)
        
        # Cross-validation
        print(f"   🔄 5-Fold CV...")
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model_inst, X_train_scaled, y_train, 
                                    cv=cv, scoring='f1', n_jobs=-1)
        cv_mean = cv_scores.mean()
        
        results[name] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_f1': cv_mean
        }
        
        print(f"   ✅ Train Accuracy: {train_acc:.4f}")
        print(f"   ✅ Test Accuracy:  {test_acc:.4f}")
        print(f"   ✅ Precision:      {precision:.4f}")
        print(f"   ✅ Recall:         {recall:.4f}")
        print(f"   ✅ F1-Score:       {f1:.4f}")
        print(f"   ✅ ROC-AUC:        {roc_auc:.4f}")
        print(f"   ✅ CV F1:          {cv_mean:.4f}")
        
        # Select best based on F1 score
        if f1 > best_score:
            best_score = f1
            best_model = model_inst
            best_name = name
    
    # ---------------------- Final Results ----------------------
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"{'=' * 80}")
    
    best_results = results[best_name]
    print(f"\n🎯 FINAL PERFORMANCE:")
    print(f"   ✅ Test Accuracy:  {best_results['test_acc']:.4f}")
    print(f"   ✅ Precision:      {best_results['precision']:.4f}")
    print(f"   ✅ Recall:         {best_results['recall']:.4f}")
    print(f"   ✅ F1-Score:       {best_results['f1']:.4f}")
    print(f"   ✅ ROC-AUC:        {best_results['roc_auc']:.4f}")
    
    # Confusion Matrix
    y_test_pred = best_model.predict(X_test_scaled)
    cm = confusion_matrix(y_test, y_test_pred)
    print(f"\n📊 Confusion Matrix:")
    print(f"   {cm}")
    
    # Classification Report
    print(f"\n📋 Classification Report:")
    print(classification_report(y_test, y_test_pred, 
                               target_names=label_encoder.classes_))
    
    # ---------------------- Comparison Table ----------------------
    print(f"\n📊 MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Accuracy':<12} {'F1-Score':<12} {'ROC-AUC':<12}")
    print(f"{'─' * 80}")
    for name, res in results.items():
        marker = "🏆" if name == best_name else "  "
        print(f"{marker} {name:<18} {res['test_acc']:>10.4f}  {res['f1']:>10.4f}  {res['roc_auc']:>10.4f}")
    print(f"{'=' * 80}")
    
    # ---------------------- Save Model ----------------------
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'pca_resume': pca_resume,
        'pca_jd': pca_jd,
        'pca_transcript': pca_transcript,
        'label_encoder': label_encoder,
        'model_name': best_name,
        'metrics': {
            'accuracy': float(best_results['test_acc']),
            'precision': float(best_results['precision']),
            'recall': float(best_results['recall']),
            'f1_score': float(best_results['f1']),
            'roc_auc': float(best_results['roc_auc']),
            'cv_f1': float(best_results['cv_f1'])
        },
        'n_components': n_components,
        'feature_count': X_combined.shape[1]
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n💾 Model saved to: {MODEL_PATH}")
    print(f"   Model: {best_name}")
    print(f"   Features: {X_combined.shape[1]}")
    print(f"   Classes: {list(label_encoder.classes_)}")
    
    print("\n✅ Training Complete!")
    return model_data

if __name__ == "__main__":
    train_recruiter_decision_model()