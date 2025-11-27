"""
ENHANCED ATS SCORE MODEL TRAINING
Optimizations:
- Better feature engineering with interaction terms
- Ensemble stacking for improved accuracy
- Advanced regularization techniques
- Feature selection to reduce noise
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Enhanced Feature Engineering ----------------------
def extract_resume_jd_features(resume_text, jd_text):
    """Extract advanced handcrafted features"""
    features = []
    
    if not resume_text or not jd_text or pd.isna(resume_text) or pd.isna(jd_text):
        return np.zeros(25)  # Increased from 15
    
    resume_lower = str(resume_text).lower()
    jd_lower = str(jd_text).lower()
    
    resume_words = resume_lower.split()
    jd_words = jd_lower.split()
    
    # Original features (1-15)
    features.append(np.log1p(len(resume_words)) / np.log1p(len(jd_words)) if len(jd_words) > 0 else 0)
    
    resume_set = set(resume_words)
    jd_set = set(jd_words)
    overlap = len(resume_set.intersection(jd_set))
    features.append(min(overlap / max(len(resume_set), 1), 1.0))
    features.append(min(overlap / max(len(jd_set), 1), 1.0))
    
    action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed',
                    'built', 'improved', 'increased', 'reduced', 'achieved', 'delivered',
                    'launched', 'optimized', 'executed', 'coordinated', 'established']
    action_count = sum(1 for verb in action_verbs if verb in resume_lower)
    features.append(min(action_count / 10.0, 1.0))
    
    tech_skills = ['python', 'java', 'sql', 'javascript', 'machine learning', 'aws', 
                   'docker', 'kubernetes', 'react', 'angular', 'django', 'flask', 
                   'tensorflow', 'pytorch', 'data science', 'agile', 'cloud', 'api',
                   'git', 'linux', 'azure', 'gcp', 'mongodb', 'postgresql']
    resume_tech = [s for s in tech_skills if s in resume_lower]
    jd_tech = [s for s in tech_skills if s in jd_lower]
    tech_match = len(set(resume_tech).intersection(set(jd_tech))) / max(len(jd_tech), 1) if jd_tech else 0
    features.append(tech_match)
    
    for section in ['experience', 'education', 'skills']:
        features.append(1 if section in resume_lower else 0)
    
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
    
    numbers_count = len([w for w in resume_words if any(c.isdigit() for c in w)])
    features.append(min(numbers_count / max(len(resume_words) / 50, 1), 1.0))
    
    features.append(1 if '@' in resume_lower or 'email' in resume_lower else 0)
    
    pro_keywords = ['professional', 'certified', 'expert', 'senior', 'lead', 'principal', 'architect']
    pro_count = sum(1 for kw in pro_keywords if kw in resume_lower)
    features.append(min(pro_count / 5.0, 1.0))
    
    year_mentions = len([w for w in resume_words if w.isdigit() and 1 <= int(w) <= 30])
    features.append(min(year_mentions / 5.0, 1.0))
    
    completeness = sum([
        1 if 'experience' in resume_lower else 0,
        1 if 'education' in resume_lower else 0,
        1 if 'skills' in resume_lower else 0,
        1 if '@' in resume_lower else 0,
        1 if any(c.isdigit() for c in resume_lower) else 0
    ]) / 5.0
    features.append(completeness)
    
    soft_skills = ['communication', 'leadership', 'teamwork', 'problem solving', 
                   'analytical', 'creative', 'adaptable', 'collaborative']
    resume_soft = [s for s in soft_skills if s in resume_lower]
    jd_soft = [s for s in soft_skills if s in jd_lower]
    soft_match = len(set(resume_soft).intersection(set(jd_soft))) / max(len(jd_soft), 1) if jd_soft else 0
    features.append(soft_match)
    
    # NEW FEATURES (16-25) - More discriminative
    
    # 16. Unique words ratio (vocabulary diversity)
    features.append(len(resume_set) / max(len(resume_words), 1))
    
    # 17. Average word length (sophistication indicator)
    avg_word_len = np.mean([len(w) for w in resume_words if w.isalpha()]) if resume_words else 0
    features.append(min(avg_word_len / 10.0, 1.0))
    
    # 18. Certification mentions
    cert_keywords = ['certified', 'certification', 'certificate', 'license', 'accredited']
    cert_count = sum(1 for kw in cert_keywords if kw in resume_lower)
    features.append(min(cert_count / 3.0, 1.0))
    
    # 19. Project mentions
    project_count = resume_lower.count('project')
    features.append(min(project_count / 5.0, 1.0))
    
    # 20. Achievement indicators (percentage/numbers with context)
    achievement_patterns = ['%', 'increased', 'decreased', 'improved', 'reduced', 'grew', 'saved']
    achievement_score = sum(1 for p in achievement_patterns if p in resume_lower)
    features.append(min(achievement_score / 5.0, 1.0))
    
    # 21. Domain-specific keywords density
    jd_important_words = [w for w in jd_words if len(w) > 5 and w not in 
                          ['experience', 'skills', 'education', 'required', 'preferred']][:20]
    domain_match = sum(1 for w in jd_important_words if w in resume_lower) / max(len(jd_important_words), 1)
    features.append(domain_match)
    
    # 22. Resume structure quality (section headers count)
    structure_keywords = ['summary', 'objective', 'experience', 'education', 'skills', 
                         'projects', 'certifications', 'achievements', 'awards']
    structure_score = sum(1 for kw in structure_keywords if kw in resume_lower)
    features.append(min(structure_score / 6.0, 1.0))
    
    # 23. Job-specific keywords match (exact phrases)
    jd_bigrams = set([' '.join(jd_words[i:i+2]) for i in range(len(jd_words)-1)])
    resume_bigrams = set([' '.join(resume_words[i:i+2]) for i in range(len(resume_words)-1)])
    bigram_match = len(jd_bigrams.intersection(resume_bigrams)) / max(len(jd_bigrams), 1)
    features.append(min(bigram_match, 1.0))
    
    # 24. Company/organization mentions
    org_keywords = ['company', 'corporation', 'inc', 'ltd', 'llc', 'organization', 'university']
    org_count = sum(1 for kw in org_keywords if kw in resume_lower)
    features.append(min(org_count / 3.0, 1.0))
    
    # 25. URL/Link presence (portfolio, github, linkedin)
    url_indicators = ['http', 'www', 'github', 'linkedin', 'portfolio']
    url_score = sum(1 for ind in url_indicators if ind in resume_lower)
    features.append(min(url_score / 3.0, 1.0))
    
    return np.array(features)

# ---------------------- Create Interaction Features ----------------------
def create_interaction_features(similarity_features, handcrafted_features):
    """Create polynomial and interaction features"""
    interactions = []
    
    # Similarity interactions (multiplicative effects)
    resume_jd = similarity_features[:, 0]
    resume_trans = similarity_features[:, 1]
    trans_jd = similarity_features[:, 2]
    
    # Two-way interactions
    interactions.append((resume_jd * resume_trans).reshape(-1, 1))
    interactions.append((resume_jd * trans_jd).reshape(-1, 1))
    interactions.append((resume_trans * trans_jd).reshape(-1, 1))
    
    # Three-way interaction (overall alignment)
    interactions.append((resume_jd * resume_trans * trans_jd).reshape(-1, 1))
    
    # Squared terms (non-linear effects)
    interactions.append((resume_jd ** 2).reshape(-1, 1))
    interactions.append((trans_jd ** 2).reshape(-1, 1))
    
    # Key handcrafted feature interactions
    tech_skills = handcrafted_features[:, 4]
    edu_level = handcrafted_features[:, 8]
    
    # Tech skills × similarity
    interactions.append((tech_skills * resume_jd).reshape(-1, 1))
    
    # Education × similarity
    interactions.append((edu_level * resume_jd).reshape(-1, 1))
    
    return np.hstack(interactions)

# ---------------------- Main Training Function ----------------------
def train_ats_model():
    DATASET_PATH = 'dataset.csv'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'ats_score_model.pkl')
    CACHE_DIR = 'embeddings_cache'
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print("=" * 80)
    print("🎯 ENHANCED ATS SCORE MODEL TRAINING")
    print("=" * 80)
    
    # ---------------------- Load Dataset ----------------------
    print("\n📂 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    print(f"✅ Loaded: {len(df)} records")
    
    # ---------------------- Load Embeddings ----------------------
    print("\n📦 Loading embeddings...")
    
    try:
        resume_embs = np.load(os.path.join(CACHE_DIR, 'resume_embs_decision.npy'))
        jd_embs = np.load(os.path.join(CACHE_DIR, 'jd_embs_decision.npy'))
        transcript_embs = np.load(os.path.join(CACHE_DIR, 'transcript_embs_decision.npy'))
        
        print(f"✅ Embeddings loaded: {resume_embs.shape}")
        
        min_len = min(len(df), len(resume_embs), len(jd_embs), len(transcript_embs))
        df = df.iloc[:min_len]
        resume_embs = resume_embs[:min_len]
        jd_embs = jd_embs[:min_len]
        transcript_embs = transcript_embs[:min_len]
            
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # ---------------------- Create Target Variable ----------------------
    print("\n🎲 Creating target variable...")
    
    np.random.seed(42)
    
    cosine_scores = np.array([
        cosine_similarity([resume_embs[i]], [jd_embs[i]])[0][0]
        for i in range(len(df))
    ])
    base_scores = cosine_scores * 100
    
    noise = np.random.normal(0, 5, len(df))
    base_scores += noise
    
    if 'decision' in df.columns:
        decision_boost = df['decision'].apply(
            lambda x: 20 if str(x).lower() in ['selected', 'hire', 'accept'] 
                   else -15 if str(x).lower() in ['rejected', 'reject'] 
                   else 0
        )
        base_scores += decision_boost
    
    df['ATS_Score'] = np.clip(base_scores, 0, 100)
    
    print(f"   📊 Score range: {df['ATS_Score'].min():.1f} - {df['ATS_Score'].max():.1f}")
    print(f"   📊 Score mean: {df['ATS_Score'].mean():.1f} ± {df['ATS_Score'].std():.1f}")
    
    # ---------------------- Data Cleaning ----------------------
    print("\n🧹 Cleaning data...")
    df = df.dropna(subset=['Resume', 'Job_Description', 'ATS_Score'])
    
    valid_indices = df.index
    resume_embs = resume_embs[valid_indices]
    jd_embs = jd_embs[valid_indices]
    transcript_embs = transcript_embs[valid_indices]
    df = df.reset_index(drop=True)
    
    print(f"   Final dataset: {len(df)} records")
    
    # ---------------------- Feature Engineering ----------------------
    print("\n🔧 Extracting enhanced features...")
    
    handcrafted_features = np.array([
        extract_resume_jd_features(row['Resume'], row['Job_Description']) 
        for _, row in df.iterrows()
    ])
    print(f"   ✅ Handcrafted features: {handcrafted_features.shape}")
    
    # ---------------------- PCA with Optimal Components ----------------------
    print("\n🔬 Applying optimized PCA...")
    
    # Slightly more components for better information retention
    n_components = min(20, len(df) // 6, resume_embs.shape[1])
    print(f"   Using {n_components} components per embedding")
    
    pca_resume = PCA(n_components=n_components, random_state=42)
    pca_jd = PCA(n_components=n_components, random_state=42)
    pca_transcript = PCA(n_components=n_components, random_state=42)
    
    resume_reduced = pca_resume.fit_transform(resume_embs)
    jd_reduced = pca_jd.fit_transform(jd_embs)
    transcript_reduced = pca_transcript.fit_transform(transcript_embs)
    
    print(f"   Variance captured: {pca_resume.explained_variance_ratio_.sum():.1%}")
    
    # ---------------------- Similarity Features ----------------------
    print("\n📐 Computing similarity features...")
    
    resume_jd_sim = np.array([
        cosine_similarity([resume_embs[i]], [jd_embs[i]])[0][0]
        for i in range(len(df))
    ]).reshape(-1, 1)
    
    resume_transcript_sim = np.array([
        cosine_similarity([resume_embs[i]], [transcript_embs[i]])[0][0]
        for i in range(len(df))
    ]).reshape(-1, 1)
    
    transcript_jd_sim = np.array([
        cosine_similarity([transcript_embs[i]], [jd_embs[i]])[0][0]
        for i in range(len(df))
    ]).reshape(-1, 1)
    
    similarity_features = np.hstack([resume_jd_sim, resume_transcript_sim, transcript_jd_sim])
    
    # ---------------------- Create Interaction Features ----------------------
    print("\n🔀 Creating interaction features...")
    interaction_features = create_interaction_features(similarity_features, handcrafted_features)
    print(f"   ✅ Interaction features: {interaction_features.shape}")
    
    # ---------------------- Combine All Features ----------------------
    X_combined = np.hstack([
        resume_reduced,
        jd_reduced,
        transcript_reduced,
        similarity_features,
        handcrafted_features,
        interaction_features
    ])
    
    y = df['ATS_Score'].values
    
    print(f"\n✅ Final feature matrix: {X_combined.shape}")
    
    # ---------------------- Feature Selection ----------------------
    print("\n🎯 Performing feature selection...")
    
    # Select top K features based on correlation with target
    k_best = min(X_combined.shape[1] - 5, int(X_combined.shape[1] * 0.85))
    selector = SelectKBest(score_func=f_regression, k=k_best)
    X_selected = selector.fit_transform(X_combined, y)
    
    print(f"   Selected {X_selected.shape[1]} features from {X_combined.shape[1]}")
    
    # ---------------------- Train-Test Split ----------------------
    print("\n📊 Splitting data...")
    
    score_bins = pd.qcut(y, q=min(5, len(y)//10), labels=False, duplicates='drop')
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=score_bins
    )
    
    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")
    
    # ---------------------- Advanced Scaling ----------------------
    print("\n⚖️ Applying advanced scaling...")
    
    # QuantileTransformer for better handling of outliers
    scaler = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ---------------------- Enhanced Model Training ----------------------
    print("\n🤖 Training enhanced models...")
    print("=" * 80)
    
    # Define base models with optimal hyperparameters
    base_models = {
        'Ridge': Ridge(alpha=5.0, random_state=42),
        
        'Huber': HuberRegressor(epsilon=1.35, alpha=0.5, max_iter=1000),
        
        'LightGBM': LGBMRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_samples=15,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            verbose=-1
        ),
        
        'XGBoost': XGBRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=3,
            reg_alpha=0.3,
            reg_lambda=1.5,
            random_state=42,
            verbosity=0
        ),
        
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.75,
            min_samples_leaf=8,
            min_samples_split=15,
            max_features='sqrt',
            random_state=42
        )
    }
    
    # Create stacking ensemble
    print("\n🏗️ Building stacking ensemble...")
    
    stacking_model = StackingRegressor(
        estimators=[
            ('lgbm', base_models['LightGBM']),
            ('xgb', base_models['XGBoost']),
            ('gb', base_models['GradientBoosting'])
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )
    
    base_models['Stacking'] = stacking_model
    
    # Train and evaluate all models
    best_model = None
    best_score = -float('inf')
    best_name = None
    results = {}
    
    for name, model_inst in base_models.items():
        print(f"\n{'─' * 80}")
        print(f"📊 Training {name}...")
        
        model_inst.fit(X_train_scaled, y_train)
        
        y_train_pred = np.clip(model_inst.predict(X_train_scaled), 0, 100)
        y_test_pred = np.clip(model_inst.predict(X_test_scaled), 0, 100)
        
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # 5-fold CV
        print(f"   🔄 Running 5-fold CV...")
        cv_scores = cross_val_score(
            model_inst, X_train_scaled, y_train,
            cv=min(5, len(X_train)//20),
            scoring='r2',
            n_jobs=-1
        )
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        
        overfit_gap = train_r2 - test_r2
        
        results[name] = {
            'model': model_inst,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2': cv_mean,
            'cv_std': cv_std,
            'mae': mae,
            'rmse': rmse,
            'overfit_gap': overfit_gap
        }
        
        print(f"   ✅ Train R²: {train_r2:.4f}")
        print(f"   ✅ Test R²:  {test_r2:.4f}")
        print(f"   📊 Gap:      {overfit_gap:.4f}")
        print(f"   ✅ CV R²:    {cv_mean:.4f} (±{cv_std:.4f})")
        print(f"   📏 MAE:      {mae:.2f}")
        print(f"   📏 RMSE:     {rmse:.2f}")
        
        # Scoring: prioritize test R² and CV, penalize overfitting
        score = test_r2 * 0.5 + cv_mean * 0.4 - (overfit_gap * 0.3)
        
        if score > best_score:
            best_score = score
            best_model = model_inst
            best_name = name
    
    # ---------------------- Final Results ----------------------
    print(f"\n{'=' * 80}")
    print(f"🏆 BEST MODEL: {best_name}")
    print(f"{'=' * 80}")
    
    best_results = results[best_name]
    
    print(f"\n🎯 FINAL PERFORMANCE:")
    print(f"   ✅ Train R²:     {best_results['train_r2']:.4f}")
    print(f"   ✅ Test R²:      {best_results['test_r2']:.4f}")
    print(f"   📊 Overfit Gap:  {best_results['overfit_gap']:.4f}")
    print(f"   ✅ CV R²:        {best_results['cv_r2']:.4f} (±{best_results['cv_std']:.4f})")
    print(f"   📏 MAE:          {best_results['mae']:.2f} points")
    print(f"   📏 RMSE:         {best_results['rmse']:.2f} points")
    
    # ---------------------- Model Comparison ----------------------
    print(f"\n📊 MODEL COMPARISON")
    print(f"{'=' * 80}")
    print(f"{'Model':<20} {'Test R²':<12} {'Gap':<15} {'CV R²':<12} {'MAE':<10}")
    print(f"{'─' * 80}")
    for name, res in results.items():
        marker = "🏆" if name == best_name else "  "
        gap_marker = "✅" if res['overfit_gap'] < 0.15 else ("⚠️" if res['overfit_gap'] < 0.25 else "❌")
        print(f"{marker} {name:<18} {res['test_r2']:>10.4f}  {gap_marker} {res['overfit_gap']:>12.4f}  "
              f"{res['cv_r2']:>10.4f}  {res['mae']:>8.2f}")
    print(f"{'=' * 80}")
    
    # ---------------------- Save Model ----------------------
    print("\n💾 Saving model...")
    
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'selector': selector,
        'pca_resume': pca_resume,
        'pca_jd': pca_jd,
        'pca_transcript': pca_transcript,
        'model_name': best_name,
        'metrics': {
            'train_r2': float(best_results['train_r2']),
            'test_r2': float(best_results['test_r2']),
            'cv_r2': float(best_results['cv_r2']),
            'cv_std': float(best_results['cv_std']),
            'mae': float(best_results['mae']),
            'rmse': float(best_results['rmse']),
            'overfit_gap': float(best_results['overfit_gap'])
        }
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"✅ Model saved: {MODEL_PATH}")
    
    print("\n" + "=" * 80)
    print("✅ ENHANCED TRAINING COMPLETE!")
    print("=" * 80)
    
    return model_data

if __name__ == "__main__":
    train_ats_model()