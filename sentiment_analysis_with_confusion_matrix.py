import os
import glob
import pandas as pd
from dotenv import load_dotenv
from transformers import pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score

# Load environment variables
load_dotenv()

# Get model name from .env
model_name = os.getenv('sentiment_model', 'cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual')

print(f"Loading model: {model_name}...")
# Initialize the pipeline
# The model returns labels like 'Positive', 'Negative', 'Neutral'
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name, truncation=True, max_length=512)

# Label mapping to match our dataset
LABEL_MAP = {
    'positive': 'positive',
    'negative': 'negative',
    'neutral': 'neutral',
    # Handle possible capitalized outputs
    'Positive': 'positive',
    'Negative': 'negative',
    'Neutral': 'neutral'
}

def analyze_sentiments(comments):
    """Run sentiment analysis on a list of comments and map labels."""
    # Handle empty or nan comments securely
    safe_comments = [str(c) if pd.notna(c) and str(c).strip() != "" else "neutral statement" for c in comments]
    
    predictions = sentiment_pipeline(safe_comments)
    
    mapped_predictions = []
    for p in predictions:
        # Map label, default to neutral if unknown
        label = p.get('label', 'neutral')
        mapped_predictions.append(LABEL_MAP.get(label, label.lower()))
    
    return mapped_predictions

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_data")
    output_dir = os.path.join(base_dir, "output")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # We will aggregate true and predicted labels for final evaluation
    all_true_labels = []
    all_pred_labels = []
    
    # Store per-course sentiment counts
    course_sentiment_counts = []
    
    # Find all CSV files in input_dir and its subdirectories
    csv_files = glob.glob(os.path.join(input_dir, "**", "*.csv"), recursive=True)
    print(f"Found {len(csv_files)} CSV files to process.")
    new_data/sentiment_analysis_with_confusion_matrix.py
    for file_path in csv_files:
        print(f"Processing: {file_path}")
        try:
            df = pd.read_csv(file_path)
            
            if 'comment' not in df.columns:
                print(f"Skipping {file_path}: Missing 'comment' column.")
                continue

            comments = df['comment'].tolist()

            print(f"  Running inference on {len(comments)} comments...")
            predicted_sentiments = analyze_sentiments(comments)

            df['predicted sentiment'] = predicted_sentiments

            # Use actual sentiment for evaluation only if column exists
            if 'actual sentiment' in df.columns:
                true_labels = df['actual sentiment'].fillna('neutral').str.lower().tolist()
                all_true_labels.extend(true_labels)
                all_pred_labels.extend(predicted_sentiments)
            
            # Save the updated dataframe
            # Recreate directory structure or just use prefix
            rel_path = os.path.relpath(file_path, input_dir)
            out_file_path = os.path.join(output_dir, rel_path)
            
            os.makedirs(os.path.dirname(out_file_path), exist_ok=True)
            df.to_csv(out_file_path, index=False)
            print(f"  Saved to {out_file_path}")
            
            # Extract per-course counts
            course_id = str(df['content_id'].iloc[0]) if 'content_id' in df.columns else os.path.splitext(os.path.basename(file_path))[0]
            counts = pd.Series(predicted_sentiments).value_counts().to_dict()
            course_sentiment_counts.append({
                'course_id': course_id,
                'positive': counts.get('positive', 0),
                'neutral': counts.get('neutral', 0),
                'negative': counts.get('negative', 0)
            })
            
            # Generate and save per-course classification report (only if actual labels exist)
            if 'actual sentiment' in df.columns:
                true_labels_course = df['actual sentiment'].fillna('neutral').str.lower().tolist()
                course_report = classification_report(true_labels_course, predicted_sentiments, zero_division=0)
                course_report_path = os.path.splitext(out_file_path)[0] + "_classification_report.txt"
                with open(course_report_path, "w") as f:
                    f.write(f"Sentiment Analysis Classification Report for {course_id}\n")
                    f.write("========================================\n\n")
                    f.write(course_report)
                print(f"  Classification report saved to {course_report_path}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            
    # Save per-course sentiment counts
    if course_sentiment_counts:
        counts_df = pd.DataFrame(course_sentiment_counts, columns=['course_id', 'positive', 'neutral', 'negative'])
        counts_path = os.path.join(output_dir, "course_sentiment_counts.txt")
        with open(counts_path, "w") as f:
            f.write(counts_df.to_string(index=False, justify='left'))
        print(f"Per-course sentiment counts saved to {counts_path}")
        
    # Evaluation
    print("\n--- Evaluation ---")
    if all_true_labels and all_pred_labels:
        report = classification_report(all_true_labels, all_pred_labels, zero_division=0)
        conf_matrix = confusion_matrix(all_true_labels, all_pred_labels)
        accuracy = accuracy_score(all_true_labels, all_pred_labels)
        
        # Save to txt
        report_path = os.path.join(output_dir, "classification_report.txt")
        with open(report_path, "w") as f:
            f.write("Sentiment Analysis Classification Report\n")
            f.write("========================================\n\n")
            f.write(f"Overall Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            # Create a simple string representation
            labels = sorted(list(set(all_true_labels + all_pred_labels)))
            f.write(f"Labels: {labels}\n")
            for row in conf_matrix:
                f.write(f"{row}\n")
                
        print(f"Evaluation metrics saved to {report_path}")
        print(report)
    else:
        print("No labels to evaluate.")

if __name__ == "__main__":
    main()
