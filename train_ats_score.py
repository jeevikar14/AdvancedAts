"""
SIMPLIFIED ATS SCORE PREDICTOR - RANDOM FOREST
Focus: Resume-JD Matching (Keyword + Semantic Similarity)

Pipeline Steps:
1. Load Data
2. Extract Keyword Match Features
3. Extract Semantic Similarity Features
4. Create Target Variable
5. Train-Test Split
6. Train Random Forest (Fixed Parameters)
7. Evaluate & Save Model
"""

import pandas as pd
import numpy as np
import joblib
import os
import re
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("SIMPLIFIED ATS SCORE PREDICTOR - RANDOM FOREST")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Data...")

df = pd.read_csv('dataset.csv')
print(f"✓ Loaded {len(df)} records")

# Load embeddings for semantic similarity
resume_embs = np.load('embeddings_cache/resume_embs_decision.npy')
jd_embs = np.load('embeddings_cache/jd_embs_decision.npy')

# Align data
min_len = min(len(df), len(resume_embs), len(jd_embs))
df = df.iloc[:min_len].reset_index(drop=True)
resume_embs = resume_embs[:min_len]
jd_embs = jd_embs[:min_len]

# Keep only valid rows
df = df.dropna(subset=['Resume', 'Job_Description'])
df = df.reset_index(drop=True)

print(f"✓ Clean data: {len(df)} samples")

# ============================================================================
# STEP 2: EXTRACT KEYWORD MATCH FEATURES
# ============================================================================
print("\n[STEP 2] Extracting Keyword Match Features...")

def extract_keywords_match(resume, jd):
    """Extract keyword matching features between resume and JD"""
    resume = str(resume).lower()
    jd = str(jd).lower()
    
    # Split into words
    resume_words = set(re.findall(r'\b\w+\b', resume))
    jd_words = set(re.findall(r'\b\w+\b', jd))
    
    # Remove very short words
    resume_words = {w for w in resume_words if len(w) > 2}
    jd_words = {w for w in jd_words if len(w) > 2}
    
    # 1. Word overlap ratio
    overlap = len(resume_words.intersection(jd_words))
    word_overlap_ratio = overlap / len(jd_words) if len(jd_words) > 0 else 0
    
    # 2. Technical skills matching
    tech_skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 
                   'kubernetes', 'react', 'angular', 'nodejs', 'mongodb', 
                   'postgresql', 'machine learning', 'deep learning', 'tensorflow',
                   'pytorch', 'pandas', 'numpy', 'scikit-learn', 'git', 'linux',
                   'django', 'flask', 'spring', 'hadoop', 'spark', 'tableau']
    
    resume_tech = [skill for skill in tech_skills if skill in resume]
    jd_tech = [skill for skill in tech_skills if skill in jd]
    tech_match_ratio = len(set(resume_tech).intersection(set(jd_tech))) / max(len(jd_tech), 1)
    
    # 3. Experience keywords
    exp_keywords = ['experience', 'years', 'worked', 'led', 'managed', 'developed',
                   'implemented', 'designed', 'built', 'created', 'delivered']
    resume_exp = sum(1 for kw in exp_keywords if kw in resume)
    jd_exp = sum(1 for kw in exp_keywords if kw in jd)
    exp_match = min(resume_exp / max(jd_exp, 1), 1.0)
    
    # 4. Education keywords
    edu_keywords = ['bachelor', 'master', 'phd', 'degree', 'university', 
                   'college', 'graduate', 'mba', 'btech', 'engineering']
    resume_edu = sum(1 for kw in edu_keywords if kw in resume)
    jd_edu = sum(1 for kw in edu_keywords if kw in jd)
    edu_match = min(resume_edu / max(jd_edu, 1), 1.0)
    
    # 5. Soft skills
    soft_skills = ['communication', 'leadership', 'teamwork', 'problem solving',
                  'analytical', 'creative', 'management', 'collaboration']
    resume_soft = [skill for skill in soft_skills if skill in resume]
    jd_soft = [skill for skill in soft_skills if skill in jd]
    soft_match_ratio = len(set(resume_soft).intersection(set(jd_soft))) / max(len(jd_soft), 1)
    
    # 6. Resume completeness (has key sections)
    has_experience = 1 if 'experience' in resume else 0
    has_education = 1 if 'education' in resume else 0
    has_skills = 1 if 'skill' in resume else 0
    completeness = (has_experience + has_education + has_skills) / 3
    
    return {
        'word_overlap_ratio': word_overlap_ratio,
        'tech_match_ratio': tech_match_ratio,
        'exp_match': exp_match,
        'edu_match': edu_match,
        'soft_match_ratio': soft_match_ratio,
        'completeness': completeness
    }

print("  Extracting keyword features for all resumes...")
keyword_features = []
for idx, row in df.iterrows():
    if idx % 1000 == 0:
        print(f"  Processing {idx}/{len(df)}...", end='\r')
    features = extract_keywords_match(row['Resume'], row['Job_Description'])
    keyword_features.append(features)

keyword_df = pd.DataFrame(keyword_features)
print(f"\n✓ Extracted {len(keyword_df.columns)} keyword-based features")

# ============================================================================
# STEP 3: EXTRACT SEMANTIC SIMILARITY FEATURES
# ============================================================================
print("\n[STEP 3] Extracting Semantic Similarity Features...")

print("  Computing cosine similarity from embeddings...")
semantic_similarity = []
for i in range(len(df)):
    sim = cosine_similarity([resume_embs[i]], [jd_embs[i]])[0][0]
    semantic_similarity.append(sim)

semantic_similarity = np.array(semantic_similarity).reshape(-1, 1)
print(f"✓ Computed semantic similarity for {len(semantic_similarity)} samples")

# ============================================================================
# STEP 4: COMBINE FEATURES
# ============================================================================
print("\n[STEP 4] Combining Features...")

# Combine keyword features and semantic similarity
X = np.hstack([
    keyword_df.values,
    semantic_similarity
])

print(f"✓ Total features: {X.shape[1]}")
print(f"  - Keyword features: {keyword_df.shape[1]}")
print(f"  - Semantic similarity: 1")

# ============================================================================
# STEP 5: CREATE TARGET VARIABLE (ATS SCORE)
# ============================================================================
print("\n[STEP 5] Creating ATS Score...")

# ATS Score = Weighted combination of keyword match and semantic similarity
keyword_weight = 0.4  # 40% from keywords
semantic_weight = 0.6  # 60% from semantic similarity

# Keyword score (average of all keyword features)
keyword_score = keyword_df.mean(axis=1).values * 100

# Semantic score
semantic_score = semantic_similarity.flatten() * 100

# Combined ATS score
ats_score = (keyword_weight * keyword_score) + (semantic_weight * semantic_score)

# Add realistic variation
np.random.seed(42)
noise = np.random.normal(0, 4, len(ats_score))
ats_score += noise

# Adjust based on decision (if available)
if 'decision' in df.columns:
    for i, decision in enumerate(df['decision']):
        if pd.notna(decision):
            if str(decision).lower() in ['selected', 'hire', 'accept']:
                ats_score[i] += 10
            elif str(decision).lower() in ['rejected', 'reject']:
                ats_score[i] -= 8

# Clip to 0-100
y = np.clip(ats_score, 0, 100)

print(f"✓ ATS Score Statistics:")
print(f"  Min:    {y.min():.1f}")
print(f"  Max:    {y.max():.1f}")
print(f"  Mean:   {y.mean():.1f}")
print(f"  Median: {np.median(y):.1f}")
print(f"  Std:    {y.std():.1f}")

# ============================================================================
# STEP 6: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 6] Splitting Data (85-15)...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.15,
    random_state=42
)

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# ============================================================================
# STEP 7: FEATURE SCALING
# ============================================================================
print("\n[STEP 7] Scaling Features...")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Scaled {X_train_scaled.shape[1]} features")

# ============================================================================
# STEP 8: TRAIN RANDOM FOREST WITH FAST HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 8] Training Random Forest with Hyperparameter Tuning...")
print("="*70)

# Define a smaller, focused parameter grid for faster tuning
rf_params = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20],
    'min_samples_split': [10, 20],
    'min_samples_leaf': [5, 10],
    'max_features': ['sqrt', 0.7]
}


rf = RandomForestRegressor(random_state=42, n_jobs=-1)

rf_search = RandomizedSearchCV(
    rf, 
    rf_params, 
    n_iter=15,           # Test only 15 random combinations
    cv=3,                # 3-fold CV instead of 5 (faster)
    scoring='r2', 
    n_jobs=-1,
    random_state=42,
    verbose=1
)

rf_search.fit(X_train_scaled, y_train)

rf_best = rf_search.best_estimator_

print("\n" + "="*70)
print("BEST MODEL FOUND")
print("="*70)
print(f"✓ Best Parameters:")
for param, value in rf_search.best_params_.items():
    print(f"  {param}: {value}")
print(f"\n✓ Best CV R² Score: {rf_search.best_score_:.4f}")

# ============================================================================
# STEP 9: EVALUATE MODEL
# ============================================================================
print("\n[STEP 9] Evaluating Model Performance...")
print("="*70)

# Predictions
y_train_pred = np.clip(rf_best.predict(X_train_scaled), 0, 100)
y_test_pred = np.clip(rf_best.predict(X_test_scaled), 0, 100)

# Metrics
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
mae = mean_absolute_error(y_test, y_test_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
gap = train_r2 - test_r2

print(f"\n📊 Performance Metrics:")
print(f"  Train R²: {train_r2:.4f} ({train_r2*100:.1f}%)")
print(f"  Test R²:  {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"  MAE:      {mae:.2f} points")
print(f"  RMSE:     {rmse:.2f} points")
print(f"  Gap:      {gap:.4f}", end="")



# Cross-validation
print("\n🔄 Cross-Validation (3-Fold):")
cv_scores = cross_val_score(rf_best, X_train_scaled, y_train, cv=3, scoring='r2', n_jobs=-1)
print(f"  CV R² Scores: {[f'{score:.4f}' for score in cv_scores]}")
print(f"  Mean CV R²:   {cv_scores.mean():.4f}")
print(f"  Std Dev:      {cv_scores.std():.4f}")

# ============================================================================
# STEP 10: FEATURE IMPORTANCE
# ============================================================================
print("\n[STEP 10] Feature Importance Analysis")
print("="*70)

feature_names = list(keyword_df.columns) + ['semantic_similarity']
importance = rf_best.feature_importances_

importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance
}).sort_values('Importance', ascending=False)

print("\n📊 Top Features Contributing to ATS Score:")
print(importance_df.to_string(index=False))

print("\n💡 Interpretation:")
print("  Higher importance = More influence on ATS score prediction")

# ============================================================================
# STEP 11: SAVE MODEL
# ============================================================================
print("\n[STEP 11] Saving Model...")

model_package = {
    'model': rf_best,
    'scaler': scaler,
    'model_name': 'Random Forest Regressor',
    'feature_names': feature_names,
    'best_params': rf_search.best_params_,
    'metrics': {
        'train_r2': float(train_r2),
        'test_r2': float(test_r2),
        'test_mae': float(mae),
        'test_rmse': float(rmse),
        'gap': float(gap),
        'best_cv_r2': float(rf_search.best_score_)
    },
    'feature_importance': importance_df.to_dict()
}

os.makedirs('models', exist_ok=True)
joblib.dump(model_package, 'models/ats_score_model.pkl')

print("✓ Model saved: models/ats_score_rf_model.pkl")

# ============================================================================
# SUMMARY
print("\n📊 How ATS Score is Calculated:")
print("  1. Keyword Matching (40%)")
print("     - Technical skills overlap")
print("     - Experience keywords match")
print("     - Education requirements match")
print("     - Soft skills alignment")
print("     - Resume completeness")
print("  2. Semantic Similarity (60%)")
print("     - Deep learning embeddings capture meaning")
print("     - Context-aware matching beyond keywords")

print(f"\n🎯 Final Model Performance:")
print(f"  ✓ Algorithm: Random Forest Regressor")
print(f"  ✓ Train R²: {train_r2:.4f} ({train_r2*100:.1f}%)")
print(f"  ✓ Test R²:  {test_r2:.4f} ({test_r2*100:.1f}%)")
print(f"  ✓ Average Prediction Error: {mae:.2f} points")
print(f"  ✓ CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")