

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print("RESUME CATEGORY CLASSIFIER - LOGISTIC REGRESSION ONLY")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Data...")
df = pd.read_csv('UpdatedResumeDataset.csv')
print(f"✓ Loaded {len(df)} resumes")
print(f"✓ Categories: {df['Category'].nunique()}")
print(f"✓ Category Distribution:")
print(df['Category'].value_counts().head(10))

# ============================================================================
# STEP 2: PREPROCESS TEXT - IMPROVED
# ============================================================================
print("\n[STEP 2] Preprocessing Text...")

def clean_text(text):
    """Enhanced text cleaning - MUST MATCH app.py preprocess_text()"""
    if not text or pd.isna(text):
        return ""
    
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Ensure minimum length
    if len(text) < 10:
        return ""
    
    return text

df['Resume_Clean'] = df['Resume'].apply(clean_text)

# Remove empty resumes
df = df[df['Resume_Clean'].str.len() > 10]
print(f"✓ Cleaned {len(df)} resumes (removed empty ones)")

# ============================================================================
# STEP 3: TRAIN-TEST SPLIT (80-20)
# ============================================================================
print("\n[STEP 3] Splitting Data (Stratified)...")

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
# STEP 4: VECTORIZATION (TF-IDF) - CONSERVATIVE SETTINGS
# ============================================================================
print("\n[STEP 4] TF-IDF Vectorization (Conservative to Prevent Overfitting)...")

vectorizer = TfidfVectorizer(
    max_features=2000,      # Reduced from 3000 to prevent overfitting
    min_df=3,               # Word must appear in at least 3 docs (increased)
    max_df=0.7,             # Word must appear in less than 70% of docs (reduced)
    ngram_range=(1, 2),     # Use single words and word pairs
    stop_words='english',   # Remove common words
    sublinear_tf=True       # Apply sublinear tf scaling
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"✓ Created {X_train_tfidf.shape[1]} features")

# ============================================================================
# STEP 5: TRAIN LOGISTIC REGRESSION WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 5] Training Logistic Regression with Hyperparameter Tuning...")
print("="*70)

# Logistic Regression with Strong Regularization
print("\nLogistic Regression (Strong Regularization)")
print("-" * 70)

lr_params = {
    'C': [0.01, 0.1, 0.5, 1.0, 2.0],  # Lower C = stronger regularization
    'penalty': ['l2'],
    'max_iter': [2000],
    'solver': ['lbfgs']
}

lr = LogisticRegression(random_state=42, class_weight='balanced')
lr_grid = GridSearchCV(
    lr, lr_params, 
    cv=5, 
    scoring='f1_weighted', 
    n_jobs=-1,
    verbose=1
)
print("Training Logistic Regression with GridSearchCV...")
lr_grid.fit(X_train_tfidf, y_train)

best_model = lr_grid.best_estimator_
print(f"\n✓ Best Parameters: {lr_grid.best_params_}")

y_train_pred = best_model.predict(X_train_tfidf)
y_test_pred = best_model.predict(X_test_tfidf)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
train_f1 = f1_score(y_train, y_train_pred, average='weighted')
test_f1 = f1_score(y_test, y_test_pred, average='weighted')

print(f"\n📊 RESULTS:")
print(f"   Train Accuracy: {train_acc:.4f}")
print(f"   Test Accuracy:  {test_acc:.4f}")

print(f"   Train F1-Score: {train_f1:.4f}")
print(f"   Test F1-Score:  {test_f1:.4f}")

# Cross-validation
cv_scores = cross_val_score(best_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"   Cross-Val Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# ============================================================================
# STEP 6: FINAL MODEL SUMMARY
# ============================================================================
print("\n" + "="*70)
print("[STEP 6] Final Model Summary")
print("="*70)

print(f"\n🏆 MODEL: Logistic Regression")
print(f"   Train Accuracy:  {train_acc:.4f}")
print(f"   Test Accuracy:   {test_acc:.4f}")

print(f"   Test F1-Score:   {test_f1:.4f}")
print(f"   CV Accuracy:     {cv_scores.mean():.4f}")

# ============================================================================
# STEP 7: DETAILED EVALUATION
# ============================================================================
print("\n" + "="*70)
print("[STEP 7] Detailed Evaluation")
print("="*70)

y_test_pred = best_model.predict(X_test_tfidf)

print("\n📋 Classification Report (Test Set):")
print(classification_report(y_test, y_test_pred, digits=4))

# ============================================================================
# STEP 8: SAVE MODEL (CORRECT FORMAT FOR APP.PY)
# ============================================================================
print("\n" + "="*70)
print("[STEP 8] Saving Model Files")
print("="*70)

# Save model (this is what app.py loads as 'resume_classifier_model.pkl')
with open('resume_classifier_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print("✓ Saved: resume_classifier_model.pkl")

# Save vectorizer (this is what app.py loads as 'tfidf_vectorizer.pkl')
with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
print("✓ Saved: tfidf_vectorizer.pkl")

# Also save complete model data for reference
model_data = {
    'model': best_model,
    'vectorizer': vectorizer,
    'model_name': 'Logistic Regression',
    'train_accuracy': float(train_acc),
    'test_accuracy': float(test_acc),
    'f1_score': float(test_f1),
    'overfitting_gap': float(train_acc - test_acc),
    'categories': list(y.unique())
}

with open('resume_classifier_complete.pkl', 'wb') as f:
    pickle.dump(model_data, f)
print("✓ Saved: resume_classifier_complete.pkl (reference)")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("✅ TRAINING COMPLETED SUCCESSFULLY!")
print("="*70)
print(f"\n🎯 FINAL RESULTS:")
print(f"   Model:           Logistic Regression")
print(f"   Train Accuracy:  {train_acc:.2%}")
print(f"   Test Accuracy:   {test_acc:.2%}")

print(f"   Test F1-Score:   {test_f1:.4f}")

print(f"\n📁 Files Created:")
print(f"   ✓ resume_classifier_model.pkl  (loaded by app.py)")
print(f"   ✓ tfidf_vectorizer.pkl         (loaded by app.py)")
print(f"   ✓ resume_classifier_complete.pkl (reference)")

