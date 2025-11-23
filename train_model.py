import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import re

def extract_additional_features(resume_text, jd_text):
    """Extract additional handcrafted features for better performance"""
    features = []
    
    # Text length features
    features.append(len(resume_text.split()))
    features.append(len(jd_text.split()))
    features.append(len(resume_text.split()) / (len(jd_text.split()) + 1))
    
    # Keyword overlap features
    resume_words = set(resume_text.lower().split())
    jd_words = set(jd_text.lower().split())
    overlap = len(resume_words.intersection(jd_words))
    features.append(overlap)
    features.append(overlap / (len(jd_words) + 1))
    
    # Section presence features
    sections = ['education', 'experience', 'skills', 'work', 'projects', 'certifications']
    for section in sections:
        features.append(1 if section in resume_text.lower() else 0)
    
    # Action verbs count
    action_verbs = ['managed', 'developed', 'created', 'implemented', 'led', 'designed', 
                    'built', 'improved', 'increased', 'reduced', 'achieved']
    action_verb_count = sum(1 for verb in action_verbs if verb in resume_text.lower())
    features.append(action_verb_count)
    
    # Quantifiable achievements
    numbers_count = len(re.findall(r'\d+%|\$\d+|\d+\+', resume_text))
    features.append(numbers_count)
    
    # Technical skills
    tech_skills = ['python', 'java', 'sql', 'javascript', 'machine learning', 'aws', 
                   'docker', 'kubernetes', 'react', 'angular', 'django', 'flask']
    tech_skills_count = sum(1 for skill in tech_skills if skill in resume_text.lower())
    features.append(tech_skills_count)
    
    # Contact info presence
    has_email = 1 if '@' in resume_text else 0
    has_phone = 1 if re.search(r'\d{10}', resume_text) else 0
    features.append(has_email)
    features.append(has_phone)
    
    return np.array(features)

def train_and_save_model():
    print("🚀 Starting Enhanced Model Training Process...")
    
    # Configuration
    DATASET_PATH = 'cleaned_dataset.csv'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'optimized_ats_model.pkl')
    
    # Create directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 1. Load Data
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found at {DATASET_PATH}")
        print("Please run preprocess_dataset.py first to create the dataset")
        return
        
    print(f"📂 Loading dataset from {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    df = df.dropna(subset=['jd_text', 'resume_text', 'score'])
    
    print(f"📊 Dataset size: {len(df)} samples")
    print(f"📈 Score range: {df['score'].min():.2f} - {df['score'].max():.2f}")
    print(f"📊 Mean score: {df['score'].mean():.2f}")
    
    X_jd = df['jd_text'].values
    X_res = df['resume_text'].values
    y = df['score'].values
    
    # 2. Load Transformer Model
    print("🧠 Loading Transformer Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # 3. Generate Embeddings
    print("⚙️ Generating Embeddings (this may take a few minutes)...")
    
    def get_embeddings_batch(texts, batch_size=32):
        embeddings = []
        total_batches = (len(texts) + batch_size - 1) // batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings.extend(model.encode(batch, show_progress_bar=False))
            if (i // batch_size + 1) % 10 == 0:
                print(f"  Processed {i // batch_size + 1}/{total_batches} batches...")
        return np.array(embeddings)
    
    jd_embs = get_embeddings_batch(X_jd)
    res_embs = get_embeddings_batch(X_res)
    print("✅ Embeddings generated")
    
    # 4. Feature Engineering
    print("🔧 Engineering Features...")
    
    # Cosine Similarity
    cosine_sims = cosine_similarity(jd_embs, res_embs).diagonal().reshape(-1, 1)
    
    # Additional handcrafted features
    print("  Extracting handcrafted features...")
    additional_features = []
    for i in range(len(X_res)):
        add_feats = extract_additional_features(X_res[i], X_jd[i])
        additional_features.append(add_feats)
        if (i + 1) % 100 == 0:
            print(f"    Processed {i + 1}/{len(X_res)} samples...")
    
    additional_features = np.array(additional_features)
    
    # Combine all features
    # Structure: [JD_Embedding, Resume_Embedding, Cosine_Similarity, Handcrafted_Features]
    X_combined = np.hstack([jd_embs, res_embs, cosine_sims, additional_features])
    
    print(f"✅ Feature matrix shape: {X_combined.shape}")
    print(f"   - JD Embeddings: {jd_embs.shape[1]} dims")
    print(f"   - Resume Embeddings: {res_embs.shape[1]} dims")
    print(f"   - Cosine Similarity: 1 dim")
    print(f"   - Handcrafted Features: {additional_features.shape[1]} dims")
    
    # 5. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.2, random_state=42, stratify=None
    )
    
    # 6. Scaling
    print("📏 Scaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 7. Model Training with Multiple Algorithms
    print("🏋️ Training Models...")
    
    models = {
        'RandomForest': RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=42
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            random_state=42
        )
    }
    
    best_model = None
    best_score = -np.inf
    best_model_name = None
    
    results = {}
    
    for model_name, model_obj in models.items():
        print(f"\n  Training {model_name}...")
        model_obj.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred_train = model_obj.predict(X_train_scaled)
        y_pred_test = model_obj.predict(X_test_scaled)
        
        # Metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        results[model_name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'mae': test_mae,
            'rmse': test_rmse
        }
        
        print(f"    Train R²: {train_r2:.4f}")
        print(f"    Test R²: {test_r2:.4f}")
        print(f"    MAE: {test_mae:.4f}")
        print(f"    RMSE: {test_rmse:.4f}")
        
        # Select best model
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model_obj
            best_model_name = model_name
    
    # 8. Final Evaluation
    print(f"\n📊 Best Model: {best_model_name}")
    print(f"   Test R²: {results[best_model_name]['test_r2']:.4f}")
    print(f"   MAE: {results[best_model_name]['mae']:.4f}")
    print(f"   RMSE: {results[best_model_name]['rmse']:.4f}")
    
    if results[best_model_name]['test_r2'] < 0.5:
        print("⚠️ Warning: Model R² is below 0.5. Consider:")
        print("   - Getting more training data")
        print("   - Feature engineering improvements")
        print("   - Hyperparameter tuning")
    else:
        print("✅ Model performance is good!")
    
    # 9. Save Model
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'model_name': best_model_name,
        'feature_names': (
            ['jd_emb'] * jd_embs.shape[1] +
            ['res_emb'] * res_embs.shape[1] +
            ['cosine_sim'] +
            ['resume_len', 'jd_len', 'len_ratio', 'keyword_overlap', 'keyword_overlap_ratio'] +
            ['sec_' + s for s in ['education', 'experience', 'skills', 'work', 'projects', 'certifications']] +
            ['action_verbs', 'numbers', 'tech_skills', 'has_email', 'has_phone']
        ),
        'metrics': {
            'train_r2': results[best_model_name]['train_r2'],
            'test_r2': results[best_model_name]['test_r2'],
            'r2': results[best_model_name]['test_r2'],
            'mae': results[best_model_name]['mae'],
            'rmse': results[best_model_name]['rmse']
        }
    }
    
    with open(MODEL_PATH, 'wb') as f:
        joblib.dump(model_data, f)
        
    print("✅ Model training and saving complete!")
    print(f"\n🎯 Summary:")
    print(f"   Model: {best_model_name}")
    print(f"   R² Score: {results[best_model_name]['test_r2']:.4f}")
    print(f"   MAE: {results[best_model_name]['mae']:.2f}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Total features: {X_combined.shape[1]}")

if __name__ == "__main__":
    train_and_save_model()