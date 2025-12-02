"""
SIMPLE RESUME CATEGORY CLASSIFIER
Clean ML Pipeline with Good Accuracy

Pipeline Steps:
1. Load Data
2. Preprocess Text
3. Train-Test Split
4. Vectorization (TF-IDF)
5. Train Multiple Models with Hyperparameter Tuning
6. Evaluate & Select Best Model
7. Save Model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("RESUME CATEGORY CLASSIFIER - SIMPLE ML PIPELINE")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Data...")
df = pd.read_csv('UpdatedResumeDataset.csv')
print(f"✓ Loaded {len(df)} resumes")
print(f"✓ Categories: {df['Category'].nunique()}")

# ============================================================================
# STEP 2: PREPROCESS TEXT
# ============================================================================
print("\n[STEP 2] Preprocessing Text...")

def clean_text(text):
    """Clean resume text"""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'\S+@\S+', '', text)  # Remove emails
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    return text

df['Resume_Clean'] = df['Resume'].apply(clean_text)
print(f"✓ Cleaned {len(df)} resumes")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT (80-20)
# ============================================================================
print("\n[STEP 3] Splitting Data...")

X = df['Resume_Clean']
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

print(f"✓ Train: {len(X_train)} samples")
print(f"✓ Test: {len(X_test)} samples")

# ============================================================================
# STEP 4: VECTORIZATION (TF-IDF)
# ============================================================================
print("\n[STEP 4] Converting Text to Numbers (TF-IDF)...")

vectorizer = TfidfVectorizer(
    max_features=3000,      # Use top 3000 words
    min_df=2,               # Word must appear in at least 2 documents
    max_df=0.8,             # Word must appear in less than 80% of documents
    ngram_range=(1, 2),     # Use single words and word pairs
    stop_words='english'    # Remove common words
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Created {X_train_tfidf.shape[1]} features")

# ============================================================================
# STEP 5: TRAIN MODELS WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 5] Training Models with Hyperparameter Tuning...")
print("="*70)

# MODEL 1: Logistic Regression
print("\n[Model 1] Logistic Regression")
print("-" * 70)

lr_params = {
    'C': [0.1, 1.0, 10.0],  # Regularization strength
    'max_iter': [1000]
}

lr = LogisticRegression(random_state=42)
lr_grid = GridSearchCV(lr, lr_params, cv=5, scoring='f1_weighted', n_jobs=-1)
lr_grid.fit(X_train_tfidf, y_train)

lr_best = lr_grid.best_estimator_
print(f"Best Parameters: {lr_grid.best_params_}")

y_pred_lr = lr_best.predict(X_test_tfidf)
lr_accuracy = accuracy_score(y_test, y_pred_lr)
lr_f1 = f1_score(y_test, y_pred_lr, average='weighted')

print(f"Test Accuracy: {lr_accuracy:.4f}")
print(f"Test F1-Score: {lr_f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(lr_best, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# MODEL 2: Random Forest
print("\n[Model 2] Random Forest")
print("-" * 70)

rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [20, 30],
    'min_samples_split': [5, 10]
}

rf = RandomForestClassifier(random_state=42, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_params, cv=5, scoring='f1_weighted', n_jobs=-1)
rf_grid.fit(X_train_tfidf, y_train)

rf_best = rf_grid.best_estimator_
print(f"Best Parameters: {rf_grid.best_params_}")

y_pred_rf = rf_best.predict(X_test_tfidf)
rf_accuracy = accuracy_score(y_test, y_pred_rf)
rf_f1 = f1_score(y_test, y_pred_rf, average='weighted')

print(f"Test Accuracy: {rf_accuracy:.4f}")
print(f"Test F1-Score: {rf_f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(rf_best, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# STEP 6: COMPARE & SELECT BEST MODEL
# ============================================================================
print("\n[STEP 6] Model Comparison")
print("="*70)

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [lr_accuracy, rf_accuracy],
    'F1-Score': [lr_f1, rf_f1]
})

print(results.to_string(index=False))

# Select best model based on F1-Score
best_idx = results['F1-Score'].idxmax()
best_model_name = results.loc[best_idx, 'Model']
best_model = lr_best if best_model_name == 'Logistic Regression' else rf_best

print(f"\n✓ Best Model: {best_model_name}")
print(f"  Accuracy: {results.loc[best_idx, 'Accuracy']:.4f}")
print(f"  F1-Score: {results.loc[best_idx, 'F1-Score']:.4f}")

# ============================================================================
# STEP 7: DETAILED EVALUATION
# ============================================================================
print("\n[STEP 7] Detailed Evaluation")
print("="*70)

y_pred = best_model.predict(X_test_tfidf)

print("\nClassification Report:")
print(classification_report(y_test, y_pred, digits=4))

# ============================================================================
# STEP 8: SAVE MODEL
# ============================================================================
print("\n[STEP 8] Saving Model...")

model_data = {
    'model': best_model,
    'vectorizer': vectorizer,
    'model_name': best_model_name,
    'accuracy': float(results.loc[best_idx, 'Accuracy']),
    'f1_score': float(results.loc[best_idx, 'F1-Score'])
}

with open('resume_classifier_simple.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Model saved: resume_classifier_model.pkl")

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