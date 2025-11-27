import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import re
import pickle
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess resume text"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

# Load your dataset
print("Loading dataset...")
df = pd.read_csv('UpdatedResumeDataset.csv')  # Change to your file path

# Check for missing values
print(f"\nDataset shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()}")

# Remove any rows with missing values
df = df.dropna()

# Preprocess resumes
print("\nPreprocessing text...")
df['Resume_Clean'] = df['Resume'].apply(preprocess_text)

# Check class distribution
print("\nClass distribution:")
print(df['Category'].value_counts())

# Prepare features and labels
X = df['Resume_Clean']
y = df['Category']

# Split data: 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Maintains class distribution
)

print(f"\nTraining samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# TF-IDF Vectorization with parameters to reduce overfitting
print("\nVectorizing text...")
tfidf = TfidfVectorizer(
    max_features=3000,      # Limit features to prevent overfitting
    min_df=2,               # Ignore terms that appear in less than 2 documents
    max_df=0.8,             # Ignore terms that appear in more than 80% of documents
    ngram_range=(1, 2),     # Use unigrams and bigrams
    stop_words='english',   # Remove common English words
    sublinear_tf=True       # Apply sublinear tf scaling
)

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print(f"Feature matrix shape: {X_train_tfidf.shape}")

# Model 1: Logistic Regression with L2 regularization (prevents overfitting)
print("\n" + "="*60)
print("Training Logistic Regression Model...")
print("="*60)

lr_model = LogisticRegression(
    C=1.0,                  # Regularization strength (lower = more regularization)
    max_iter=1000,
    random_state=42,
    class_weight='balanced' # Handle imbalanced classes
)

# Cross-validation to check generalization
cv_scores = cross_val_score(lr_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Train on full training set
lr_model.fit(X_train_tfidf, y_train)

# Predictions
y_train_pred = lr_model.predict(X_train_tfidf)
y_test_pred = lr_model.predict(X_test_tfidf)

# Evaluation
train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"\nLogistic Regression Results:")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")
print(f"Difference (Overfitting indicator): {train_acc - test_acc:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred))

# Model 2: Random Forest with constraints
print("\n" + "="*60)
print("Training Random Forest Model...")
print("="*60)

rf_model = RandomForestClassifier(
    n_estimators=100,       # Number of trees
    max_depth=20,           # Limit tree depth to prevent overfitting
    min_samples_split=10,   # Minimum samples required to split
    min_samples_leaf=4,     # Minimum samples in leaf node
    max_features='sqrt',    # Number of features to consider for splits
    random_state=42,
    class_weight='balanced',
    n_jobs=-1               # Use all CPU cores
)

# Cross-validation
cv_scores_rf = cross_val_score(rf_model, X_train_tfidf, y_train, cv=5, scoring='accuracy')
print(f"\nCross-validation scores: {cv_scores_rf}")
print(f"Mean CV accuracy: {cv_scores_rf.mean():.4f} (+/- {cv_scores_rf.std():.4f})")

# Train
rf_model.fit(X_train_tfidf, y_train)

# Predictions
y_train_pred_rf = rf_model.predict(X_train_tfidf)
y_test_pred_rf = rf_model.predict(X_test_tfidf)

# Evaluation
train_acc_rf = accuracy_score(y_train, y_train_pred_rf)
test_acc_rf = accuracy_score(y_test, y_test_pred_rf)

print(f"\nRandom Forest Results:")
print(f"Training Accuracy: {train_acc_rf:.4f}")
print(f"Testing Accuracy: {test_acc_rf:.4f}")
print(f"Difference (Overfitting indicator): {train_acc_rf - test_acc_rf:.4f}")

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_test_pred_rf))

# Choose best model based on test performance and generalization
print("\n" + "="*60)
print("MODEL COMPARISON")
print("="*60)
print(f"Logistic Regression - Test Accuracy: {test_acc:.4f}, Overfitting Gap: {train_acc - test_acc:.4f}")
print(f"Random Forest - Test Accuracy: {test_acc_rf:.4f}, Overfitting Gap: {train_acc_rf - test_acc_rf:.4f}")

# Save the best model
if test_acc >= test_acc_rf:
    best_model = lr_model
    model_name = "Logistic Regression"
else:
    best_model = rf_model
    model_name = "Random Forest"

print(f"\nBest Model: {model_name}")

# Save model and vectorizer
print("\nSaving model and vectorizer...")
with open('resume_classifier_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Model and vectorizer saved successfully!")

# Function to predict new resumes
def predict_resume_category(resume_text):
    """Predict category for a new resume"""
    # Load model and vectorizer
    with open('resume_classifier_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Preprocess and predict
    cleaned_text = preprocess_text(resume_text)
    vectorized = vectorizer.transform([cleaned_text])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized).max()
    
    return prediction, probability

# Example usage
print("\n" + "="*60)
print("EXAMPLE PREDICTION")
print("="*60)

sample_resume = X_test.iloc[0]
actual_category = y_test.iloc[0]

prediction, confidence = predict_resume_category(sample_resume)
print(f"Predicted Category: {prediction}")
print(f"Confidence: {confidence:.4f}")
print(f"Actual Category: {actual_category}")
print(f"Correct: {prediction == actual_category}")