"""
SIMPLE RECRUITER DECISION MODEL
Clean ML Pipeline with Good Accuracy

Pipeline Steps:
1. Load Data & Embeddings
2. Feature Engineering
3. Train-Test Split
4. Train Models with Hyperparameter Tuning
5. Evaluate & Select Best Model
6. Save Model
"""

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RECRUITER DECISION MODEL - SIMPLE ML PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Data & Embeddings...")

df = pd.read_csv('dataset.csv')
print(f"✓ Loaded {len(df)} records")

# Load embeddings
resume_emb = np.load('embeddings_cache/resume_embs_decision.npy')
jd_emb = np.load('embeddings_cache/jd_embs_decision.npy')
transcript_emb = np.load('embeddings_cache/transcript_embs_decision.npy')

# Align data
min_len = min(len(df), len(resume_emb), len(jd_emb), len(transcript_emb))
df = df.iloc[:min_len].reset_index(drop=True)
resume_emb = resume_emb[:min_len]
jd_emb = jd_emb[:min_len]
transcript_emb = transcript_emb[:min_len]

# Keep only rows with valid decisions
valid_idx = df['decision'].notna()
df = df[valid_idx].reset_index(drop=True)
resume_emb = resume_emb[valid_idx]
jd_emb = jd_emb[valid_idx]
transcript_emb = transcript_emb[valid_idx]

print(f"✓ Final data: {len(df)} samples")
print(f"\nDecision Distribution:")
print(df['decision'].value_counts())

# ============================================================================
# STEP 2: FEATURE ENGINEERING
# ============================================================================
print("\n[STEP 2] Creating Features...")

# Calculate cosine similarities
from sklearn.metrics.pairwise import cosine_similarity

print("  Computing similarities...")
resume_jd_sim = []
resume_transcript_sim = []
transcript_jd_sim = []

for i in range(len(df)):
    resume_jd_sim.append(cosine_similarity([resume_emb[i]], [jd_emb[i]])[0][0])
    resume_transcript_sim.append(cosine_similarity([resume_emb[i]], [transcript_emb[i]])[0][0])
    transcript_jd_sim.append(cosine_similarity([transcript_emb[i]], [jd_emb[i]])[0][0])

# Use first 50 dimensions from embeddings + similarities
print("  Combining features...")
features = np.hstack([
    resume_emb[:, :50],
    jd_emb[:, :50],
    transcript_emb[:, :50],
    np.array(resume_jd_sim).reshape(-1, 1),
    np.array(resume_transcript_sim).reshape(-1, 1),
    np.array(transcript_jd_sim).reshape(-1, 1)
])

print(f"✓ Created {features.shape[1]} features")

# Prepare X and y
X = features
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['decision'])

print(f"✓ Classes: {list(label_encoder.classes_)}")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT
# ============================================================================
print("\n[STEP 3] Splitting Data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# Scale features
print("\n  Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# STEP 4: TRAIN MODELS WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 4] Training Models with Hyperparameter Tuning...")
print("="*70)

# MODEL 1: Logistic Regression
print("\n[Model 1] Logistic Regression")
print("-" * 70)

lr_params = {
    'C': [0.1, 1.0, 10.0],
    'max_iter': [1000]
}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1_weighted', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

lr_best = lr_grid.best_estimator_
print(f"Best Parameters: {lr_grid.best_params_}")

y_pred_lr = lr_best.predict(X_test_scaled)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

print(f"Test Accuracy: {lr_accuracy:.4f}")
print(f"Test F1-Score: {lr_f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(lr_best, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# MODEL 2: Random Forest
print("\n[Model 2] Random Forest")
print("-" * 70)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20],
    'min_samples_split': [5, 10]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

rf_best = rf_grid.best_estimator_
print(f"Best Parameters: {rf_grid.best_params_}")

y_pred_rf = rf_best.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Test Accuracy: {rf_accuracy:.4f}")
print(f"Test F1-Score: {rf_f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_best, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# MODEL 3: Gradient Boosting
print("\n[Model 3] Gradient Boosting")
print("-" * 70)

gb_params = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5]
}

gb = GradientBoostingClassifier(random_state=42)
gb_grid = GridSearchCV(gb, gb_params, cv=5, scoring='f1_weighted', n_jobs=-1)
gb_grid.fit(X_train_scaled, y_train)

gb_best = gb_grid.best_estimator_
print(f"Best Parameters: {gb_grid.best_params_}")

y_pred_gb = gb_best.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, y_pred_gb)
gb_f1 = f1_score(y_test, y_pred_gb, average='weighted')

print(f"Test Accuracy: {gb_accuracy:.4f}")
print(f"Test F1-Score: {gb_f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(gb_best, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# STEP 5: COMPARE & SELECT BEST MODEL
# ============================================================================
print("\n[STEP 5] Model Comparison")
print("="*70)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest', 'Gradient Boosting'],
    'Accuracy': [lr_accuracy, rf_accuracy, gb_accuracy],
    'F1-Score': [lr_f1, rf_f1, gb_f1]
})

print(results.to_string(index=False))

# Select best model based on F1-Score
best_idx = results['F1-Score'].idxmax()
best_model_name = results.loc[best_idx, 'Model']

if best_model_name == 'Logistic Regression':
    best_model = lr_best
elif best_model_name == 'Random Forest':
    best_model = rf_best
else:
    best_model = gb_best

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Accuracy: {results.loc[best_idx, 'Accuracy']:.4f}")
print(f"  F1-Score: {results.loc[best_idx, 'F1-Score']:.4f}")

# ============================================================================
# STEP 6: DETAILED EVALUATION
# ============================================================================
print("\n[STEP 6] Detailed Evaluation")
print("="*70)

y_pred = best_model.predict(X_test_scaled)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                           target_names=label_encoder.classes_, 
                           digits=4))

# ============================================================================
# STEP 7: SAVE MODEL
# ============================================================================
print("\n[STEP 7] Saving Model...")

model_data = {
    'model': best_model,
    'scaler': scaler,
    'label_encoder': label_encoder,
    'model_name': best_model_name,
    'accuracy': float(results.loc[best_idx, 'Accuracy']),
    'f1_score': float(results.loc[best_idx, 'F1-Score'])
}

joblib.dump(model_data, 'recruiter_model.pkl')

print("✓ Model saved: recruiter_model.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("PIPELINE COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"✓ Best Model: {best_model_name}")
print(f"✓ Test Accuracy: {results.loc[best_idx, 'Accuracy']:.2%}")
print(f"✓ Test F1-Score: {results.loc[best_idx, 'F1-Score']:.4f}")
print("="*70)