from datasets import load_dataset
import pandas as pd
import re

def clean_text(text):
    """Clean and normalize text"""
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep important punctuation
    text = re.sub(r'[^\w\s@\.\-\+\%\$\#\(\)]', ' ', text)
    
    # Remove multiple spaces again
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def split_resume_jd(text):
    """Enhanced split function for resume and JD separation"""
    if not text:
        return "", ""
    
    # Try multiple separator patterns
    separators = ['SEP', '===', '---', 'JOB DESCRIPTION:', 'RESUME:', 'JOBDESCRIPTION', 'Job Description']
    
    for sep in separators:
        if sep in text:
            parts = text.split(sep, 1)
            if len(parts) == 2:
                resume = parts[0].replace("RESUME:", "").strip()
                jd = parts[1].replace("JD:", "").replace("Job Description:", "").strip()
                
                # Clean up any remaining separator markers
                resume = re.sub(r'SEP|===|---|RESUME:|JD:', '', resume).strip()
                jd = re.sub(r'SEP|===|---|RESUME:|JD:', '', jd).strip()
                
                # Validate lengths
                if len(resume) > 100 and len(jd) > 100:
                    return clean_text(resume), clean_text(jd)
    
    # Alternative: Try to detect resume vs JD by content
    text_lower = text.lower()
    
    # JD typically has "requirements", "responsibilities", "qualifications"
    # Resume typically has "experience", "education", "skills"
    
    jd_markers = ['requirements', 'responsibilities', 'qualifications', 'we are looking for', 'ideal candidate']
    resume_markers = ['summary', 'objective', 'work experience', 'education', 'certifications']
    
    # Find positions of markers
    jd_positions = [text_lower.find(marker) for marker in jd_markers if marker in text_lower]
    resume_positions = [text_lower.find(marker) for marker in resume_markers if marker in text_lower]
    
    if jd_positions and resume_positions:
        avg_jd_pos = sum(jd_positions) / len(jd_positions)
        avg_resume_pos = sum(resume_positions) / len(resume_positions)
        
        if avg_resume_pos < avg_jd_pos:
            # Resume comes first
            split_point = int((avg_resume_pos + avg_jd_pos) / 2)
            resume = text[:split_point].strip()
            jd = text[split_point:].strip()
            if len(resume) > 100 and len(jd) > 100:
                return clean_text(resume), clean_text(jd)
    
    # Last resort: split by length
    if len(text) > 500:
        split_point = len(text) // 2
        return clean_text(text[:split_point]), clean_text(text[split_point:])
    
    return "", ""

def validate_sample(resume_text, jd_text, score):
    """Validate that a sample is good for training"""
    # Check minimum lengths
    if len(resume_text.split()) < 50 or len(jd_text.split()) < 50:
        return False
    
    # Check maximum lengths (avoid corrupted data)
    if len(resume_text.split()) > 3000 or len(jd_text.split()) > 3000:
        return False
    
    # Check score is valid
    if score < 0 or score > 100:
        return False
    
    # Check for minimum content quality
    resume_lower = resume_text.lower()
    jd_lower = jd_text.lower()
    
    # Resume should have some career-related words
    resume_keywords = ['experience', 'education', 'skill', 'work', 'project', 'company']
    if not any(keyword in resume_lower for keyword in resume_keywords):
        return False
    
    # JD should have some job-related words
    jd_keywords = ['requirement', 'responsibility', 'qualification', 'candidate', 'position', 'role']
    if not any(keyword in jd_lower for keyword in jd_keywords):
        return False
    
    return True

def main():
    """Main preprocessing function"""
    print("🚀 Starting Dataset Preprocessing...")
    
    # Load dataset
    DATASET_NAME = "0xnbk/resume-ats-score-v1-en"
    
    try:
        print(f"📥 Loading dataset: {DATASET_NAME}")
        dataset = load_dataset(DATASET_NAME)
        print(f"✅ Dataset loaded successfully")
        print(f"   Total samples: {len(dataset['train'])}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    # Process dataset
    print("\n🔄 Processing samples...")
    data = []
    skipped = 0
    
    for idx, sample in enumerate(dataset['train']):
        if (idx + 1) % 100 == 0:
            print(f"   Processed {idx + 1}/{len(dataset['train'])} samples...")
        
        text = sample.get('text', '')
        
        # Try multiple score field names
        score = (sample.get('ats_score') or 
                sample.get('score') or 
                sample.get('label') or 
                sample.get('rating'))
        
        if text and score is not None:
            try:
                # Split into resume and JD
                resume_text, jd_text = split_resume_jd(text)
                
                # Convert score to float
                score = float(score)
                
                # Validate sample
                if validate_sample(resume_text, jd_text, score):
                    data.append({
                        'resume_text': resume_text,
                        'jd_text': jd_text,
                        'score': score
                    })
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue
        else:
            skipped += 1
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    print(f"\n✅ Processing complete!")
    print(f"   Valid samples: {len(df)}")
    print(f"   Skipped samples: {skipped}")
    print(f"   Success rate: {len(df)/(len(df)+skipped)*100:.1f}%")
    
    if len(df) == 0:
        print("❌ No valid samples found. Please check the dataset format.")
        return
    
    # Save to CSV
    output_file = "cleaned_dataset.csv"
    df.to_csv(output_file, index=False)
    print(f"\n💾 Dataset saved as '{output_file}'")
    
    # Display statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(df)}")
    print(f"   Score range: {df['score'].min():.2f} - {df['score'].max():.2f}")
    print(f"   Mean score: {df['score'].mean():.2f}")
    print(f"   Median score: {df['score'].median():.2f}")
    print(f"   Std deviation: {df['score'].std():.2f}")
    
    # Score distribution
    print("\n📈 Score Distribution:")
    bins = [0, 25, 50, 75, 100]
    labels = ['0-25', '25-50', '50-75', '75-100']
    df['score_bin'] = pd.cut(df['score'], bins=bins, labels=labels)
    print(df['score_bin'].value_counts().sort_index())
    
    # Sample preview
    print("\n👀 Sample Preview:")
    sample_row = df.iloc[0]
    print(f"\nResume preview (first 200 chars):")
    print(sample_row['resume_text'][:200] + "...")
    print(f"\nJD preview (first 200 chars):")
    print(sample_row['jd_text'][:200] + "...")
    print(f"Score: {sample_row['score']}")
    
    print("\n✅ Preprocessing complete! You can now run train_model.py")

if __name__ == "__main__":
    main()