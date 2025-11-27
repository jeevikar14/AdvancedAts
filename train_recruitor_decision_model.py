# ==============================
# RECRUITER DECISION MODEL - TRAINING SCRIPT
# Optimized for Generalization | Uses Cached Embeddings
# ==============================

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.metrics import (accuracy_score, f1_score, precision_score, 
                            recall_score, classification_report, confusion_matrix)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


# ==============================
# CONFIGURATION
# ==============================
CONFIG = {
    'dataset_path': 'dataset.csv',
    'embeddings_cache': 'embeddings_cache',
    'model_save_path': 'recruiter_model.pkl',
    'random_state': 42,
    'test_size': 0.25,
    'cv_folds': 5
}

# ==============================
# 1. LOAD DATA
# ==============================
def load_data():
    """Load dataset and cached embeddings"""
    print("=" * 60)
    print("LOADING DATA")
    print("=" * 60)
    
    # Load CSV
    df = pd.read_csv(CONFIG['dataset_path'])
    print(f"✓ Loaded {len(df)} records from dataset")
    
    # Load embeddings
    emb_dir = CONFIG['embeddings_cache']
    resume_emb = np.load(os.path.join(emb_dir, 'resume_embs_decision.npy'))
    jd_emb = np.load(os.path.join(emb_dir, 'jd_embs_decision.npy'))
    transcript_emb = np.load(os.path.join(emb_dir, 'transcript_embs_decision.npy'))
    
    print(f"✓ Resume embeddings: {resume_emb.shape}")
    print(f"✓ Job Description embeddings: {jd_emb.shape}")
    print(f"✓ Transcript embeddings: {transcript_emb.shape}")
    
    # Ensure data alignment
    min_len = min(len(df), len(resume_emb), len(jd_emb), len(transcript_emb))
    df = df.iloc[:min_len].reset_index(drop=True)
    resume_emb = resume_emb[:min_len]
    jd_emb = jd_emb[:min_len]
    transcript_emb = transcript_emb[:min_len]
    
    # Remove any rows with missing decisions
    valid_idx = df['decision'].notna()
    df = df[valid_idx].reset_index(drop=True)
    resume_emb = resume_emb[valid_idx]
    jd_emb = jd_emb[valid_idx]
    transcript_emb = transcript_emb[valid_idx]
    
    print(f"✓ Final dataset size: {len(df)} records")
    
    return df, resume_emb, jd_emb, transcript_emb

# ==============================
# 2. FEATURE ENGINEERING
# ==============================
def create_features(resume_emb, jd_emb, transcript_emb):
    """Create smart features from embeddings"""
    print("\n" + "=" * 60)
    print("FEATURE ENGINEERING")
    print("=" * 60)
    
    # Normalize embeddings
    resume_norm = resume_emb / (np.linalg.norm(resume_emb, axis=1, keepdims=True) + 1e-8)
    jd_norm = jd_emb / (np.linalg.norm(jd_emb, axis=1, keepdims=True) + 1e-8)
    transcript_norm = transcript_emb / (np.linalg.norm(transcript_emb, axis=1, keepdims=True) + 1e-8)
    
    # Similarity scores
    resume_jd_sim = np.sum(resume_norm * jd_norm, axis=1, keepdims=True)
    resume_transcript_sim = np.sum(resume_norm * transcript_norm, axis=1, keepdims=True)
    jd_transcript_sim = np.sum(jd_norm * transcript_norm, axis=1, keepdims=True)
    
    # Concatenate: original embeddings + similarity features
    features = np.hstack([
        resume_emb,
        jd_emb,
        transcript_emb,
        resume_jd_sim,
        resume_transcript_sim,
        jd_transcript_sim
    ])
    
    print(f"✓ Created feature matrix: {features.shape}")
    print(f"  - Resume embedding: {resume_emb.shape[1]} dims")
    print(f"  - JD embedding: {jd_emb.shape[1]} dims")
    print(f"  - Transcript embedding: {transcript_emb.shape[1]} dims")
    print(f"  - Similarity features: 3")
    
    return features

# ==============================
# 3. PREPARE DATA FOR TRAINING
# ==============================
def prepare_data(df, features):
    """Encode labels and split data"""
    print("\n" + "=" * 60)
    print("PREPARING DATA")
    print("=" * 60)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['decision'])
    
    classes, counts = np.unique(y, return_counts=True)
    print(f"✓ Label encoding complete")
    print(f"  Class distribution:")
    for cls, count in zip(label_encoder.classes_, counts):
        print(f"    {cls}: {count} ({count/len(y)*100:.1f}%)")
    
    # Scale features using RobustScaler (better for outliers)
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(features)
    print(f"✓ Feature scaling complete")
    
    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=CONFIG['test_size'],
        stratify=y,
        random_state=CONFIG['random_state']
    )
    
    print(f"✓ Data split complete")
    print(f"  Train: {len(X_train)} samples")
    print(f"  Test:  {len(X_test)} samples")
    
    return X_train, X_test, y_train, y_test, scaler, label_encoder

# ==============================
# 4. TRAIN AND EVALUATE MODELS
# ==============================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """Train multiple models and select the best"""
    print("\n" + "=" * 60)
    print("MODEL TRAINING & EVALUATION")
    print("=" * 60)
    
    models = {
        'XGBoost': XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=CONFIG['random_state'],
            eval_metric='logloss',
            verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_samples=20,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=CONFIG['random_state'],
            verbosity=-1
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            max_features='sqrt',
            random_state=CONFIG['random_state'],
            n_jobs=-1
        )
    }
    
    results = {}
    best_model = None
    best_f1 = 0
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        
        # Train
        model.fit(X_train, y_train)
        
        # Cross-validation
        cv_scores = cross_validate(
            model, X_train, y_train,
            cv=CONFIG['cv_folds'],
            scoring=['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted'],
            n_jobs=-1
        )
        
        # Test predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, y_pred)
        test_f1 = f1_score(y_test, y_pred, average='weighted')
        test_precision = precision_score(y_test, y_pred, average='weighted')
        test_recall = recall_score(y_test, y_pred, average='weighted')
        
        # Store results
        results[name] = {
            'model': model,
            'cv_accuracy': cv_scores['test_accuracy'].mean(),
            'cv_f1': cv_scores['test_f1_weighted'].mean(),
            'cv_std': cv_scores['test_f1_weighted'].std(),
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
            'predictions': y_pred
        }
        
        # Print results
        print(f"  CV Accuracy:  {cv_scores['test_accuracy'].mean():.4f} (±{cv_scores['test_accuracy'].std():.4f})")
        print(f"  CV F1-Score:  {cv_scores['test_f1_weighted'].mean():.4f} (±{cv_scores['test_f1_weighted'].std():.4f})")
        print(f"  Train Acc:    {train_acc:.4f}")
        print(f"  Test Acc:     {test_acc:.4f}")
        print(f"  Test F1:      {test_f1:.4f}")
        print(f"  Test Prec:    {test_precision:.4f}")
        print(f"  Test Recall:  {test_recall:.4f}")
        print(f"  Overfitting:  {train_acc - test_acc:.4f} (lower is better)")
        
        # Track best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            best_model = name
    
    print(f"\n{'='*60}")
    print(f"BEST MODEL: {best_model} (F1-Score: {best_f1:.4f})")
    print(f"{'='*60}")
    
    return results, best_model

# ==============================
# 5. DETAILED EVALUATION
# ==============================
def detailed_evaluation(results, best_model, y_test, label_encoder):
    """Print detailed evaluation for best model"""
    print(f"\n{'='*60}")
    print(f"DETAILED EVALUATION - {best_model}")
    print(f"{'='*60}")
    
    best_result = results[best_model]
    y_pred = best_result['predictions']
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, 
                                target_names=label_encoder.classes_,
                                digits=4))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    return best_result

# ==============================
# 6. SAVE MODEL
# ==============================
def save_model(model, scaler, label_encoder):
    """Save trained model and preprocessing objects"""
    print(f"\n{'='*60}")
    print("SAVING MODEL")
    print(f"{'='*60}")
    
    model_package = {
        'model': model,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'version': '1.0',
        'features': 'resume + jd + transcript embeddings + similarities'
    }
    
    joblib.dump(model_package, CONFIG['model_save_path'])
    print(f"✓ Model saved to: {CONFIG['model_save_path']}")

# ==============================
# MAIN EXECUTION
# ==============================
def main():
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "RECRUITER DECISION MODEL TRAINING" + " " * 15 + "║")
    print("╚" + "═" * 58 + "╝")
    
    # Step 1: Load data
    df, resume_emb, jd_emb, transcript_emb = load_data()
    
    # Step 2: Create features
    features = create_features(resume_emb, jd_emb, transcript_emb)
    
    # Step 3: Prepare data
    X_train, X_test, y_train, y_test, scaler, label_encoder = prepare_data(df, features)
    
    # Step 4: Train and evaluate models
    results, best_model_name = train_and_evaluate(X_train, X_test, y_train, y_test)
    
    # Step 5: Detailed evaluation
    best_result = detailed_evaluation(results, best_model_name, y_test, label_encoder)
    
    # Step 6: Save best model
    save_model(results[best_model_name]['model'], scaler, label_encoder)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Model: {best_model_name}")
    print(f"Test F1-Score: {best_result['test_f1']:.4f}")
    print(f"Test Accuracy: {best_result['test_accuracy']:.4f}")
    print("=" * 60 + "\n")

if __name__ == "__main__":
    main()