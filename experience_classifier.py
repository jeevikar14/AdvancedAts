"""
EXPERIENCE LEVEL CLUSTERING MODEL
Clean ML Pipeline with Hyperparameter Tuning

Pipeline Steps:
1. Load Data
2. Extract Features from Resumes
3. Feature Engineering
4. Feature Scaling
5. Hyperparameter Tuning (NEW!)
6. Train Clustering Model
7. Assign Experience Labels
8. Evaluate Clustering Quality
9. Save Model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import re
from datetime import datetime
import pickle
import warnings
from itertools import product
warnings.filterwarnings('ignore')

print("="*70)
print("EXPERIENCE LEVEL CLUSTERING - ML PIPELINE WITH HYPERPARAMETER TUNING")
print("="*70)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================
print("\n[STEP 1] Loading Data...")

df = pd.read_csv('UpdatedResumeDataset.csv')
print(f"✓ Loaded {len(df)} resumes")

df = df.dropna(subset=['Resume'])
print(f"✓ After removing missing: {len(df)} resumes")

# ============================================================================
# STEP 2: EXTRACT FEATURES FROM RESUMES
# ============================================================================
print("\n[STEP 2] Extracting Features from Text...")

def extract_years_experience(text):
    """Extract years of experience from resume text"""
    if pd.isna(text):
        return None
    
    text = str(text).lower()
    
    # Pattern 1: "5 years experience" or "5+ years experience"
    pattern1 = re.search(r'(\d+)\s*\+?\s*years?\s+(?:of\s+)?experience', text)
    if pattern1:
        return float(pattern1.group(1))
    
    # Pattern 2: "experience: 5 years"
    pattern2 = re.search(r'experience[:\s]+(\d+)\s*\+?\s*years?', text)
    if pattern2:
        return float(pattern2.group(1))
    
    # Pattern 3: "3-5 years" or "3 to 5 years"
    pattern3 = re.search(r'(\d+)\s*(?:to|-)\s*(\d+)\s*years?', text)
    if pattern3:
        return (float(pattern3.group(1)) + float(pattern3.group(2))) / 2
    
    # Pattern 4: Graduation year (estimate experience)
    current_year = datetime.now().year
    grad_pattern = re.search(r'(?:graduated|graduation|passed).*?(\d{4})', text)
    if grad_pattern:
        grad_year = int(grad_pattern.group(1))
        if 1990 <= grad_year <= current_year:
            return max(0, float(current_year - grad_year))
    
    return None

def count_keywords(text, keywords):
    """Count how many keywords appear in text"""
    if pd.isna(text):
        return 0
    text = str(text).lower()
    return sum(1 for keyword in keywords if keyword in text)

print("  Extracting years of experience...")
df['Years_Experience'] = df['Resume'].apply(extract_years_experience)

print("  Counting keywords...")

# Entry-level keywords
entry_keywords = ['fresher', 'intern', 'internship', 'graduate', 'entry level', 
                 'trainee', 'beginner', 'junior', 'recent graduate']
df['Entry_Keywords'] = df['Resume'].apply(lambda x: count_keywords(x, entry_keywords))

# Mid-level keywords
mid_keywords = ['experienced', 'specialist', 'consultant', 'professional', 
               'developer', 'engineer', 'analyst', 'coordinator']
df['Mid_Keywords'] = df['Resume'].apply(lambda x: count_keywords(x, mid_keywords))

# Senior-level keywords
senior_keywords = ['senior', 'sr.', 'sr', 'lead', 'manager', 'director', 
                  'head', 'expert', 'architect', 'principal', 'vp', 'chief']
df['Senior_Keywords'] = df['Resume'].apply(lambda x: count_keywords(x, senior_keywords))

# Leadership keywords
leadership_keywords = ['led', 'managed', 'mentored', 'supervised', 'coordinated', 
                      'directed', 'managed team', 'team lead', 'leadership']
df['Leadership_Keywords'] = df['Resume'].apply(lambda x: count_keywords(x, leadership_keywords))

print("  Extracting resume characteristics...")

# Resume length (indicator of experience)
df['Resume_Length'] = df['Resume'].apply(lambda x: len(str(x).split()))

# Projects count
df['Projects_Count'] = df['Resume'].apply(lambda x: str(x).lower().count('project'))

# Certifications
df['Certifications'] = df['Resume'].apply(
    lambda x: str(x).lower().count('certified') + str(x).lower().count('certification')
)

# Technical skills count
tech_skills = ['python', 'java', 'javascript', 'sql', 'aws', 'docker', 
              'kubernetes', 'react', 'angular', 'node', 'machine learning']
df['Tech_Skills'] = df['Resume'].apply(lambda x: count_keywords(x, tech_skills))

print(f"✓ Extracted 10 features from resumes")

# ============================================================================
# STEP 3: HANDLE MISSING YEARS (IMPUTATION)
# ============================================================================
print("\n[STEP 3] Handling Missing Years of Experience...")

missing_years = df['Years_Experience'].isna().sum()
print(f"  Missing years: {missing_years} ({missing_years/len(df)*100:.1f}%)")

def smart_impute_years(row):
    """Impute missing years based on other features"""
    if pd.notna(row['Years_Experience']):
        return row['Years_Experience']
    
    # Calculate a score based on keywords
    score = 0
    score += row['Senior_Keywords'] * 3
    score += row['Leadership_Keywords'] * 2
    score += row['Mid_Keywords'] * 1.5
    score += (row['Resume_Length'] / 500) * 1
    score += row['Certifications'] * 1
    score -= row['Entry_Keywords'] * 2
    
    # Use consistent random seed based on resume content
    np.random.seed(int(abs(hash(str(row['Resume'][:50]))) % 10000))
    
    # Assign years based on score
    if score >= 10:
        return np.random.uniform(8, 15)    # Senior
    elif score >= 5:
        return np.random.uniform(4, 8)     # Mid
    elif score >= 0:
        return np.random.uniform(2, 4)     # Entry-Mid
    else:
        return np.random.uniform(0, 2)     # Entry

df['Years_Experience'] = df.apply(smart_impute_years, axis=1)

print(f"✓ Imputed missing years")
print(f"\n  Years distribution:")
print(f"    0-3 years: {len(df[df['Years_Experience'] < 3])} ({len(df[df['Years_Experience'] < 3])/len(df)*100:.1f}%)")
print(f"    3-7 years: {len(df[(df['Years_Experience'] >= 3) & (df['Years_Experience'] < 7)])} ({len(df[(df['Years_Experience'] >= 3) & (df['Years_Experience'] < 7)])/len(df)*100:.1f}%)")
print(f"    7+ years: {len(df[df['Years_Experience'] >= 7])} ({len(df[df['Years_Experience'] >= 7])/len(df)*100:.1f}%)")

# ============================================================================
# STEP 4: FEATURE ENGINEERING FOR CLUSTERING
# ============================================================================
print("\n[STEP 4] Engineering Features for Clustering...")

# Create discriminative features
df['Years_Squared'] = df['Years_Experience'] ** 2  # Amplify differences
df['Seniority_Score'] = (df['Senior_Keywords'] * 3 + 
                         df['Leadership_Keywords'] * 2 - 
                         df['Entry_Keywords'])
df['Experience_Density'] = df['Years_Experience'] / (df['Resume_Length'] / 1000 + 1)

# Select features for clustering
feature_columns = [
    'Years_Experience',
    'Years_Squared',
    'Entry_Keywords',
    'Mid_Keywords',
    'Senior_Keywords',
    'Leadership_Keywords',
    'Resume_Length',
    'Projects_Count',
    'Certifications',
    'Tech_Skills',
    'Seniority_Score',
    'Experience_Density'
]

X = df[feature_columns].values
print(f"✓ Created {len(feature_columns)} features for clustering")

# ============================================================================
# STEP 5: FEATURE SCALING
# ============================================================================
print("\n[STEP 5] Scaling Features...")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"✓ Scaled {X_scaled.shape[1]} features")

# ============================================================================
# STEP 6: HYPERPARAMETER TUNING
# ============================================================================
print("\n[STEP 6] HYPERPARAMETER TUNING")
print("="*70)
print("Testing different configurations to find optimal parameters...")
print("-" * 70)

# Define hyperparameter grid
param_grid = {
    'n_clusters': [3],  # Fixed at 3 for Entry/Mid/Senior
    'n_components': [6, 8, 10],  # PCA components
    'n_init': [20, 50, 100],  # Number of KMeans initializations
    'max_iter': [300, 500, 1000],  # Max iterations
    'random_state': [42]  # Fixed for reproducibility
}

# Generate all combinations
param_combinations = list(product(
    param_grid['n_clusters'],
    param_grid['n_components'],
    param_grid['n_init'],
    param_grid['max_iter'],
    param_grid['random_state']
))

print(f"Total configurations to test: {len(param_combinations)}\n")

# Store results
tuning_results = []

for idx, (n_clusters, n_components, n_init, max_iter, random_state) in enumerate(param_combinations, 1):
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    
    # Train KMeans
    kmeans = KMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        max_iter=max_iter,
        random_state=random_state
    )
    labels = kmeans.fit_predict(X_pca)
    
    # Calculate metrics
    silhouette = silhouette_score(X_pca, labels)
    davies_bouldin = davies_bouldin_score(X_pca, labels)
    variance_explained = pca.explained_variance_ratio_.sum()
    inertia = kmeans.inertia_
    
    # Combined score (higher silhouette, lower davies_bouldin)
    combined_score = silhouette - (davies_bouldin / 10)
    
    result = {
        'config_id': idx,
        'n_clusters': n_clusters,
        'n_components': n_components,
        'n_init': n_init,
        'max_iter': max_iter,
        'silhouette': silhouette,
        'davies_bouldin': davies_bouldin,
        'variance_explained': variance_explained,
        'inertia': inertia,
        'combined_score': combined_score,
        'pca_model': pca,
        'kmeans_model': kmeans
    }
    
    tuning_results.append(result)
    
    print(f"Config {idx}/{len(param_combinations)}: "
          f"PCA={n_components}, n_init={n_init}, max_iter={max_iter} | "
          f"Silhouette={silhouette:.4f}, DB={davies_bouldin:.4f}, "
          f"Score={combined_score:.4f}")

# ============================================================================
# STEP 7: SELECT BEST CONFIGURATION
# ============================================================================
print("\n[STEP 7] Selecting Best Configuration...")
print("-" * 70)

# Sort by combined score (descending)
tuning_results_sorted = sorted(tuning_results, key=lambda x: x['combined_score'], reverse=True)

# Get top 5 configurations
print("\nTop 5 Configurations:")
for i, result in enumerate(tuning_results_sorted[:5], 1):
    print(f"\n{i}. Config {result['config_id']}:")
    print(f"   PCA Components: {result['n_components']}")
    print(f"   n_init: {result['n_init']}")
    print(f"   max_iter: {result['max_iter']}")
    print(f"   Silhouette Score: {result['silhouette']:.4f}")
    print(f"   Davies-Bouldin: {result['davies_bouldin']:.4f}")
    print(f"   Variance Explained: {result['variance_explained']:.2%}")
    print(f"   Combined Score: {result['combined_score']:.4f}")

# Select best configuration
best_config = tuning_results_sorted[0]

print("\n" + "="*70)
print("BEST CONFIGURATION SELECTED:")
print("="*70)
print(f"  PCA Components: {best_config['n_components']}")
print(f"  n_init: {best_config['n_init']}")
print(f"  max_iter: {best_config['max_iter']}")
print(f"  Silhouette Score: {best_config['silhouette']:.4f}")
print(f"  Davies-Bouldin Index: {best_config['davies_bouldin']:.4f}")
print(f"  Variance Explained: {best_config['variance_explained']:.2%}")
print("="*70)

# ============================================================================
# STEP 8: TRAIN FINAL MODEL WITH BEST PARAMETERS
# ============================================================================
print("\n[STEP 8] Training Final Model with Best Parameters...")

# Use best models
final_pca = best_config['pca_model']
final_model = best_config['kmeans_model']

# Apply transformations
X_pca = final_pca.transform(X_scaled)
df['Cluster'] = final_model.predict(X_pca)

print(f"✓ Trained optimal model")
print(f"  • PCA components: {best_config['n_components']}")
print(f"  • KMeans n_init: {best_config['n_init']}")
print(f"  • KMeans max_iter: {best_config['max_iter']}")

# ============================================================================
# STEP 9: ASSIGN EXPERIENCE LEVEL LABELS
# ============================================================================
print("\n[STEP 9] Assigning Experience Level Labels...")
print("-" * 70)

# Analyze each cluster
cluster_analysis = []
for cluster_id in range(3):
    cluster_data = df[df['Cluster'] == cluster_id]
    
    analysis = {
        'cluster': cluster_id,
        'size': len(cluster_data),
        'avg_years': cluster_data['Years_Experience'].mean(),
        'avg_senior_kw': cluster_data['Senior_Keywords'].mean(),
        'avg_leadership_kw': cluster_data['Leadership_Keywords'].mean(),
        'avg_entry_kw': cluster_data['Entry_Keywords'].mean()
    }
    cluster_analysis.append(analysis)

# Sort by average years to assign labels
cluster_analysis_sorted = sorted(cluster_analysis, key=lambda x: x['avg_years'])

# Create mapping
label_mapping = {}
level_names = ['Entry-Level', 'Mid-Level', 'Senior-Level']

for i, cluster_info in enumerate(cluster_analysis_sorted):
    cluster_id = cluster_info['cluster']
    label_mapping[cluster_id] = level_names[i]
    
    print(f"\nCluster {cluster_id} → {level_names[i]}")
    print(f"  Size: {cluster_info['size']} resumes ({cluster_info['size']/len(df)*100:.1f}%)")
    print(f"  Avg Years: {cluster_info['avg_years']:.2f}")
    print(f"  Avg Senior Keywords: {cluster_info['avg_senior_kw']:.2f}")
    print(f"  Avg Leadership Keywords: {cluster_info['avg_leadership_kw']:.2f}")

# Apply labels
df['Experience_Level'] = df['Cluster'].map(label_mapping)

# ============================================================================
# STEP 10: VALIDATION
# ============================================================================
print("\n[STEP 10] Validating Experience Level Progression...")
print("-" * 70)

for level in ['Entry-Level', 'Mid-Level', 'Senior-Level']:
    level_data = df[df['Experience_Level'] == level]
    
    print(f"\n{level}:")
    print(f"  Count: {len(level_data)}")
    print(f"  Avg Years: {level_data['Years_Experience'].mean():.2f}")
    print(f"  Years Range: {level_data['Years_Experience'].min():.1f} - {level_data['Years_Experience'].max():.1f}")
    print(f"  Avg Senior Keywords: {level_data['Senior_Keywords'].mean():.2f}")

# ============================================================================
# STEP 11: VISUALIZATIONS
# ============================================================================
print("\n[STEP 11] Creating Visualizations...")

# 1. Cluster visualization
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
colors = {'Entry-Level': 'green', 'Mid-Level': 'blue', 'Senior-Level': 'red'}
for level in ['Entry-Level', 'Mid-Level', 'Senior-Level']:
    level_data = df[df['Experience_Level'] == level]
    indices = level_data.index
    plt.scatter(X_pca[indices, 0], X_pca[indices, 1], 
               c=colors[level], label=level, alpha=0.6, s=30)
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.title('Experience Level Clustering Results')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. Hyperparameter tuning results
plt.subplot(1, 2, 2)
configs = [r['config_id'] for r in tuning_results_sorted]
scores = [r['combined_score'] for r in tuning_results_sorted]
plt.bar(range(len(configs)), scores, color='steelblue', alpha=0.7)
plt.xlabel('Configuration (sorted by score)')
plt.ylabel('Combined Score')
plt.title('Hyperparameter Tuning Results')
plt.axhline(y=best_config['combined_score'], color='red', linestyle='--', 
            label=f"Best: {best_config['combined_score']:.4f}")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('clustering_with_tuning.png', dpi=300, bbox_inches='tight')
print("✓ Saved visualization: clustering_with_tuning.png")
plt.close()

# ============================================================================
# STEP 12: SAVE MODEL AND RESULTS
# ============================================================================
print("\n[STEP 12] Saving Model and Results...")

model_package = {
    'kmeans_model': final_model,
    'scaler': scaler,
    'pca': final_pca,
    'feature_columns': feature_columns,
    'label_mapping': label_mapping,
    'best_params': {
        'n_components': best_config['n_components'],
        'n_init': best_config['n_init'],
        'max_iter': best_config['max_iter']
    },
    'metrics': {
        'silhouette_score': best_config['silhouette'],
        'davies_bouldin_score': best_config['davies_bouldin'],
        'variance_explained': best_config['variance_explained']
    },
    'tuning_results': tuning_results_sorted
}

with open('clustering_model_optimized.pkl', 'wb') as f:
    pickle.dump(model_package, f)
print("✓ Model saved: clustering_model_tuned.pkl")

# Save results
output_df = df[['Category', 'Years_Experience', 'Cluster', 'Experience_Level', 
                'Senior_Keywords', 'Entry_Keywords', 'Leadership_Keywords']]
output_df.to_csv('clustered_resumes_tuned.csv', index=False)
print("✓ Results saved: clustered_resumes_tuned.csv")

# Save tuning results
tuning_df = pd.DataFrame([{
    'config_id': r['config_id'],
    'n_components': r['n_components'],
    'n_init': r['n_init'],
    'max_iter': r['max_iter'],
    'silhouette': r['silhouette'],
    'davies_bouldin': r['davies_bouldin'],
    'variance_explained': r['variance_explained'],
    'combined_score': r['combined_score']
} for r in tuning_results_sorted])
tuning_df.to_csv('hyperparameter_tuning_results.csv', index=False)
print("✓ Tuning results saved: hyperparameter_tuning_results.csv")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("CLUSTERING WITH HYPERPARAMETER TUNING COMPLETED!")
print("="*70)

print(f"\nBest Model Performance:")
print(f"  ✓ Silhouette Score: {best_config['silhouette']:.4f}")
print(f"  ✓ Davies-Bouldin Index: {best_config['davies_bouldin']:.4f}")
print(f"  ✓ Variance Explained: {best_config['variance_explained']:.2%}")

print(f"\nOptimal Hyperparameters:")
print(f"  ✓ PCA Components: {best_config['n_components']}")
print(f"  ✓ KMeans n_init: {best_config['n_init']}")
print(f"  ✓ KMeans max_iter: {best_config['max_iter']}")

print(f"\nExperience Level Distribution:")
for level in ['Entry-Level', 'Mid-Level', 'Senior-Level']:
    count = len(df[df['Experience_Level'] == level])
    avg_years = df[df['Experience_Level'] == level]['Years_Experience'].mean()
    print(f"  {level}: {count} ({count/len(df)*100:.1f}%) - Avg {avg_years:.1f} years")

print("\n" + "="*70)