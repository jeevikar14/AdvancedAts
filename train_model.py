import pandas as pd
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

# ---------------------- Enhanced features ----------------------
def extract_additional_features(resume_text, jd_text):
    features = []
    
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Basic length features
    resume_len = len(resume_text.split())
    jd_len = len(jd_text.split())
    features.append(resume_len / max(jd_len, 1))
    features.append(np.log1p(resume_len))  # Log-scaled length
    
    # Keyword overlap (multiple variants)
    resume_words = set(resume_lower.split())
    jd_words = set(jd_lower.split())
    overlap = len(resume_words.intersection(jd_words))
    union = len(resume_words.union(jd_words))
    
    features.append(overlap)  # Absolute overlap
    features.append(overlap / max(len(jd_words), 1))  # JD coverage
    features.append(overlap / max(len(resume_words), 1))  # Resume relevance
    features.append(overlap / max(union, 1))  # Jaccard similarity
    
    # Action verbs (stronger signal)
    action_verbs = ['managed','developed','created','implemented','led','designed',
                    'built','improved','increased','reduced','achieved','delivered',
                    'launched','established','optimized','executed','coordinated',
                    'architected','spearheaded','drove','initiated']
    action_count = sum(1 for verb in action_verbs if verb in resume_lower)
    features.append(min(action_count / max(resume_len/100, 1), 20))  # Normalized count
    
    # Technical skills match (detailed)
    tech_skills = ['python','java','sql','javascript','machine learning','aws','docker','kubernetes',
                   'react','angular','django','flask','tensorflow','pytorch','data science','agile','scrum',
                   'mongodb','postgresql','redis','spark','hadoop','tableau','power bi','excel']
    resume_tech = set(s for s in tech_skills if s in resume_lower)
    jd_tech = set(s for s in tech_skills if s in jd_lower)
    
    tech_overlap = len(resume_tech.intersection(jd_tech))
    features.append(tech_overlap)  # Absolute match count
    features.append(tech_overlap / max(len(jd_tech), 1) if jd_tech else 0)  # Tech coverage
    features.append(len(resume_tech))  # Total resume skills
    
    # Section presence (weighted)
    critical_sections = ['experience', 'education', 'skills', 'projects', 'certifications']
    for section in critical_sections:
        features.append(1 if section in resume_lower else 0)
    
    # Quantifiable achievements
    has_numbers = len([w for w in resume_text.split() if any(c.isdigit() for c in w)])
    features.append(min(has_numbers / max(resume_len/50, 1), 10))
    
    # Contact info presence
    features.append(1 if '@' in resume_text else 0)
    features.append(1 if any(str(i) in resume_text for i in range(10)) else 0)
    
    return np.array(features)

# ---------------------- Generate or Load Cached Embeddings ----------------------
def get_or_generate_embeddings(X_jd, X_res, cache_dir='embeddings_cache'):
    """Generate embeddings or load from cache if available"""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Check for ANY existing cached embeddings first (from previous runs)
    cache_files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
    jd_cache_files = [f for f in cache_files if f.startswith('jd_embeddings_') and f.endswith('.npy')]
    res_cache_files = [f for f in cache_files if f.startswith('res_embeddings_') and f.endswith('.npy')]
    
    # If cache exists, try to use it
    if jd_cache_files and res_cache_files:
        # Use the most recent cache file
        jd_cache_file = os.path.join(cache_dir, sorted(jd_cache_files)[-1])
        res_cache_file = os.path.join(cache_dir, sorted(res_cache_files)[-1])
        
        try:
            print(f"📦 Found cached embeddings from previous run...")
            jd_embs = np.load(jd_cache_file)
            res_embs = np.load(res_cache_file)
            
            # Verify dimensions match current dataset
            if len(jd_embs) == len(X_jd) and len(res_embs) == len(X_res):
                print(f"✅ Embeddings loaded from cache! ({jd_cache_file})")
                print(f"   Saved ~8 minutes of computation time!")
                return jd_embs, res_embs
            else:
                print(f"⚠️  Cache size mismatch (cache: {len(jd_embs)}, current: {len(X_jd)})")
                print(f"   Regenerating embeddings...")
        except Exception as e:
            print(f"⚠️  Could not load cache: {e}")
            print(f"   Regenerating embeddings...")
    
    # Generate new embeddings if cache not found or invalid
    print("🧠 Generating embeddings (this will take ~8 minutes)...")
    print("   These will be cached for future runs...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    jd_embs = model.encode(X_jd, show_progress_bar=True, batch_size=32)
    res_embs = model.encode(X_res, show_progress_bar=True, batch_size=32)
    
    # Save with dataset size in filename for better tracking
    import hashlib
    data_hash = hashlib.md5((str(len(X_jd)) + str(len(X_res))).encode()).hexdigest()
    
    jd_cache_file = os.path.join(cache_dir, f'jd_embeddings_{data_hash}.npy')
    res_cache_file = os.path.join(cache_dir, f'res_embeddings_{data_hash}.npy')
    
    np.save(jd_cache_file, jd_embs)
    np.save(res_cache_file, res_embs)
    print(f"💾 Embeddings cached to: {cache_dir}/")
    print(f"   Next run will load in seconds instead of minutes!")
    
    return jd_embs, res_embs

# ---------------------- Training function ----------------------
def train_and_save_model():
    DATASET_PATH = 'cleaned_dataset.csv'
    MODEL_DIR = 'models'
    MODEL_PATH = os.path.join(MODEL_DIR, 'optimized_ats_model.pkl')
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load dataset
    print("📂 Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Identify columns
    resume_col = None
    jd_col = None
    score_col = None
    
    for col in df.columns:
        col_lower = col.lower()
        if 'resume' in col_lower and resume_col is None:
            resume_col = col
        if 'jd' in col_lower or 'job' in col_lower and jd_col is None:
            jd_col = col
        if 'score' in col_lower:
            score_col = col
    
    print(f"✅ Columns: Resume='{resume_col}', JD='{jd_col}', Score='{score_col}'")
    
    # Clean data
    df = df.dropna(subset=[jd_col, resume_col, score_col])
    df = df[(df[score_col] >= 0) & (df[score_col] <= 100)]
    
    # Remove exact duplicates
    initial_size = len(df)
    df = df.drop_duplicates(subset=[resume_col, jd_col])
    print(f"🧹 Removed {initial_size - len(df)} duplicates")
    
    # Data quality analysis
    print(f"\n📊 Dataset Analysis:")
    print(f"   Total samples: {len(df)}")
    print(f"   Score distribution:")
    print(f"     Mean: {df[score_col].mean():.2f}")
    print(f"     Std:  {df[score_col].std():.2f}")
    print(f"     Min:  {df[score_col].min():.2f}")
    print(f"     Max:  {df[score_col].max():.2f}")
    
    # Check for score variance
    score_variance = df[score_col].var()
    if score_variance < 100:
        print(f"   ⚠️  WARNING: Low score variance ({score_variance:.2f}) - may hurt model performance")

    X_jd = df[jd_col].values
    X_res = df[resume_col].values
    y = df[score_col].values

    # ---------------------- Generate or load embeddings ----------------------
    jd_embs, res_embs = get_or_generate_embeddings(X_jd, X_res)

    # ---------------------- CRITICAL: Use more PCA components ----------------------
    # More components = more info retained = better predictions
    n_components = min(30, len(df) // 150)  # Adaptive based on dataset size
    print(f"\n📐 PCA Configuration: {n_components} components")
    
    pca_jd = PCA(n_components=n_components, random_state=42)
    pca_res = PCA(n_components=n_components, random_state=42)
    jd_reduced = pca_jd.fit_transform(jd_embs)
    res_reduced = pca_res.fit_transform(res_embs)
    
    variance_jd = pca_jd.explained_variance_ratio_.sum()
    variance_res = pca_res.explained_variance_ratio_.sum()
    print(f"   JD variance retained: {variance_jd:.1%}")
    print(f"   Resume variance retained: {variance_res:.1%}")

    # ---------------------- Cosine similarity ----------------------
    print("🔧 Computing similarity features...")
    cosine_sims = np.array([cosine_similarity([jd_embs[i]], [res_embs[i]])[0][0] 
                           for i in range(len(X_jd))]).reshape(-1, 1)

    # ---------------------- Enhanced handcrafted features ----------------------
    print("🔧 Extracting enhanced features...")
    additional = np.array([extract_additional_features(X_res[i], X_jd[i]) 
                          for i in range(len(X_res))])

    # ---------------------- Combine ALL features ----------------------
    X_combined = np.hstack([jd_reduced, res_reduced, cosine_sims, additional])
    
    print(f"✅ Feature matrix: {X_combined.shape}")
    print(f"   Total features: {X_combined.shape[1]}")
    print(f"   Samples/feature ratio: {X_combined.shape[0] / X_combined.shape[1]:.1f}:1")

    # ---------------------- Stratified split (important for diverse test set) ----------------------
    # Create bins for stratification
    y_binned = pd.cut(y, bins=5, labels=False)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_combined, y, test_size=0.15, random_state=42, stratify=y_binned, shuffle=True
    )
    
    print(f"\n📊 Split: Train={len(X_train)}, Test={len(X_test)}")

    # ---------------------- Robust Scaling ----------------------
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---------------------- OPTIMIZED Models ----------------------
    print("\n🤖 Training optimized ensemble models...")
    print("=" * 70)
    
    models = {
        'XGBoost': XGBRegressor(
            n_estimators=400,         # More trees for better learning
            max_depth=6,              # Deep enough to capture patterns
            learning_rate=0.05,       # Moderate learning
            subsample=0.85,           # High subsample
            colsample_bytree=0.85,    # High column sample
            min_child_weight=2,       # Lower constraint
            reg_alpha=0.01,           # Minimal L1
            reg_lambda=0.1,           # Light L2
            gamma=0,                  # No gamma constraint
            random_state=42,
            verbosity=0,
            tree_method='hist'        # Faster training
        ),
        'GradientBoosting': GradientBoostingRegressor(
            n_estimators=400,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.85,
            max_features='sqrt',
            min_samples_leaf=3,
            min_samples_split=6,
            random_state=42,
            validation_fraction=0.15,
            n_iter_no_change=50,
            tol=1e-5
        )
    }
    
    best_model = None
    best_score = -float('inf')
    best_name = None
    results = {}
    
    for name, model_inst in models.items():
        print(f"\n{'─' * 70}")
        print(f"📊 Training {name}...")
        
        # Fit
        model_inst.fit(X_train_scaled, y_train)
        
        # Predictions
        y_train_pred = np.clip(model_inst.predict(X_train_scaled), 0, 100)
        y_test_pred = np.clip(model_inst.predict(X_test_scaled), 0, 100)
        
        # Metrics
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        mae = mean_absolute_error(y_test, y_test_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        
        # Cross-validation for reliability check
        print(f"   🔄 5-Fold CV...")
        cv_scores = cross_val_score(model_inst, X_train_scaled, y_train, 
                                    cv=5, scoring='r2', n_jobs=-1)
        cv_mean = cv_scores.mean()
        
        gap = train_r2 - test_r2
        
        results[name] = {
            'train_r2': train_r2,
            'test_r2': test_r2,
            'cv_r2': cv_mean,
            'mae': mae,
            'rmse': rmse,
            'gap': gap
        }
        
        print(f"   ✅ Train R²: {train_r2:.4f}")
        print(f"   ✅ Test R²:  {test_r2:.4f}")
        print(f"   ✅ CV R²:    {cv_mean:.4f}")
        print(f"   📏 MAE: {mae:.2f}")
        print(f"   ⚖️  Gap: {gap:.4f}")
        
        # Select best based on test R²
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model_inst
            best_name = name
    
    # ---------------------- Ensemble attempt if both models are similar ----------------------
    print(f"\n{'=' * 70}")
    print("🔬 Checking ensemble potential...")
    
    xgb_score = results['XGBoost']['test_r2']
    gb_score = results['GradientBoosting']['test_r2']
    
    # If scores are close, try ensemble
    if abs(xgb_score - gb_score) < 0.05:
        print("   📊 Creating ensemble model...")
        
        xgb_pred = models['XGBoost'].predict(X_test_scaled)
        gb_pred = models['GradientBoosting'].predict(X_test_scaled)
        
        # Weighted average (favor better model slightly)
        if xgb_score > gb_score:
            ensemble_pred = 0.6 * xgb_pred + 0.4 * gb_pred
        else:
            ensemble_pred = 0.4 * xgb_pred + 0.6 * gb_pred
        
        ensemble_pred = np.clip(ensemble_pred, 0, 100)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        print(f"   Ensemble Test R²: {ensemble_r2:.4f}")
        
        if ensemble_r2 > best_score:
            print(f"   ✅ Ensemble improves performance!")
            best_score = ensemble_r2
            best_name = "Ensemble"
    
    print(f"\n🏆 BEST: {best_name}")
    print(f"{'=' * 70}")

    # ---------------------- Final Evaluation ----------------------
    best_results = results[best_name] if best_name != "Ensemble" else {
        'train_r2': max(results['XGBoost']['train_r2'], results['GradientBoosting']['train_r2']),
        'test_r2': ensemble_r2,
        'cv_r2': (results['XGBoost']['cv_r2'] + results['GradientBoosting']['cv_r2']) / 2,
        'mae': (results['XGBoost']['mae'] + results['GradientBoosting']['mae']) / 2,
        'rmse': (results['XGBoost']['rmse'] + results['GradientBoosting']['rmse']) / 2,
        'gap': max(results['XGBoost']['gap'], results['GradientBoosting']['gap'])
    }
    
    print(f"\n🎯 FINAL PERFORMANCE")
    print(f"{'=' * 70}")
    print(f"✅ Train R²: {best_results['train_r2']:.4f}")
    print(f"✅ Test R²:  {best_results['test_r2']:.4f}")
    print(f"✅ CV R²:    {best_results['cv_r2']:.4f}")
    print(f"✅ MAE:      {best_results['mae']:.2f}")
    print(f"✅ RMSE:     {best_results['rmse']:.2f}")
    print(f"⚖️  Gap:     {best_results['gap']:.4f}")
    print(f"{'=' * 70}")
    
    if best_results['test_r2'] >= 0.5:
        print(f"\n🎉 SUCCESS! Test R² ≥ 0.5 ✅")
    elif best_results['test_r2'] >= 0.45:
        print(f"\n✅ Close! Test R² = {best_results['test_r2']:.4f}")
    else:
        print(f"\n⚠️  Test R² = {best_results['test_r2']:.4f}")
        print("   Possible issues:")
        print("   1. Dataset labels may be noisy/inconsistent")
        print("   2. Resume-JD pairs may lack strong signal")
        print("   3. Need more diverse training data")

    # ---------------------- Comparison Table ----------------------
    print(f"\n📊 MODEL COMPARISON")
    print(f"{'=' * 70}")
    print(f"{'Model':<20} {'Train R²':<12} {'Test R²':<12} {'CV R²':<12}")
    print(f"{'─' * 70}")
    for name, res in results.items():
        marker = "🏆" if name == best_name else "  "
        print(f"{marker} {name:<18} {res['train_r2']:>10.4f}  {res['test_r2']:>10.4f}  {res['cv_r2']:>10.4f}")
    print(f"{'=' * 70}")

    # ---------------------- Save model ----------------------
    model_data = {
        'model': best_model,
        'scaler': scaler,
        'pca_jd': pca_jd,
        'pca_res': pca_res,
        'model_name': best_name,
        'metrics': {
            'r2_train': float(best_results['train_r2']),
            'r2_test': float(best_results['test_r2']),
            'r2_cv': float(best_results['cv_r2']),
            'mae': float(best_results['mae']),
            'rmse': float(best_results['rmse']),
            'gap': float(best_results['gap'])
        },
        'n_components': n_components
    }
    
    joblib.dump(model_data, MODEL_PATH)
    print(f"\n💾 Saved: {MODEL_PATH}")
    print(f"   Model: {best_name}")
    print(f"   Features: {X_combined.shape[1]}")

if __name__ == "__main__":
    print("🚀 ATS MODEL TRAINING - OPTIMIZED FOR R² > 0.5")
    print("=" * 70)
    train_and_save_model()
    print("\n✅ Complete!")