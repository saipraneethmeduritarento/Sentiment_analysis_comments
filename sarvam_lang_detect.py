import os
import time
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from sarvamai import SarvamAI

# Load env variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_data", "full_comments")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "language_detection", "sarvam")
SUMMARY_FILE = os.path.join(BASE_DIR, "output", "language_detection", "Sarvam_summary.txt")

# Configuration
MAX_FILES = None  # Set to None to process all files, or a number to limit for testing
MAX_ROWS_PER_FILE = None  # Set to None to process all rows, or a number like 50 to limit for testing
REQUEST_DELAY = 0.05  # Delay in seconds between API requests (50ms)
BATCH_DELAY = 1.5  # Additional delay in seconds every BATCH_SIZE requests
BATCH_SIZE = 50  # Number of requests before applying BATCH_DELAY

# Initialize Sarvam AI client
client = SarvamAI(
    api_subscription_key=os.getenv("SARVAM_API_SUBSCRIPTION_KEY"),
)

def detect_language_with_sarvam(text):
    if pd.isna(text) or not str(text).strip():
        return "UNKNOWN", 0.0, "UNKNOWN", 0.0
    
    max_retries = 5
    retry_delay = 2  # seconds
    
    for attempt in range(max_retries):
        try:
            start = time.time()
            result = client.text.identify_language(input=str(text))
            latency = time.time() - start
            
            # Extract language code and script
            language_code = result.language_code if hasattr(result, 'language_code') else "UNKNOWN"
            script_code = result.script_code if hasattr(result, 'script_code') else "UNKNOWN"
            
            # Sarvam doesn't provide confidence scores, so we'll use 1.0 for successful detections
            confidence = 1.0 if language_code != "UNKNOWN" else 0.0
            
            # Format top predicted labels
            top_labels = f"{language_code}:1.0000"
            
            # Add delay to respect rate limits
            time.sleep(REQUEST_DELAY)
            
            return language_code, confidence, top_labels, latency
        
        except Exception as e:
            error_str = str(e)
            # Check if it's a rate limit error
            if "rate_limit_exceeded" in error_str or "429" in error_str or "rate_limit" in error_str.lower():
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    print(f"⏳ Rate limit hit for '{str(text)[:20]}...', waiting {wait_time}s (attempt {attempt+1}/{max_retries})")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"❌ Rate limit exceeded after {max_retries} attempts: {str(text)[:30]}...")
                    return "ERROR", 0.0, "RATE_LIMIT_ERROR", 0.0
            else:
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                print(f"❌ Error for text: {str(text)[:30]}... -> {str(e)[:100]}")
                return "ERROR", 0.0, "ERROR", 0.0
    
    return "ERROR", 0.0, "ERROR", 0.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    
    print("="*60)
    print("Sarvam AI Language Detection")
    print("="*60)
    if MAX_FILES:
        print(f"⚙️  MODE: Testing (processing max {MAX_FILES} files)")
    else:
        print(f"⚙️  MODE: Full processing (all files)")
    if MAX_ROWS_PER_FILE:
        print(f"⚙️  Row limit: {MAX_ROWS_PER_FILE} rows per file")
    else:
        print(f"⚙️  Row limit: All rows per file")
    print(f"⏱️  Request delay: {REQUEST_DELAY}s per request + {BATCH_DELAY}s every {BATCH_SIZE} requests")
    print("="*60)
    
    csv_files = list(Path(INPUT_DIR).rglob("*.csv"))
    if MAX_FILES:
        csv_files = csv_files[:MAX_FILES]
    print(f"Found {len(csv_files)} CSV files to process")
    
    total_samples = 0
    total_time = 0.0
    lang_counts = {}
    
    summary_lines = []
    summary_lines.append("📊 Sarvam AI Language Detection Summary")
    summary_lines.append("="*40)
    
    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Limit rows if MAX_ROWS_PER_FILE is set
        if MAX_ROWS_PER_FILE is not None:
            df = df.head(MAX_ROWS_PER_FILE)
            print(f"  Limited to first {len(df)} rows for testing")
        
        # Check text column
        text_col = "comment" if "comment" in df.columns else ("text" if "text" in df.columns else None)
        if not text_col:
            print(f"⚠️ Skipping {file_path.name}: No applicable text column found.")
            continue
        
        # Prepare output file path
        rel_path = file_path.relative_to(INPUT_DIR)
        out_file = Path(OUTPUT_DIR) / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if we're resuming from a previous run
        start_idx = 0
        if out_file.exists():
            try:
                existing_df = pd.read_csv(out_file)
                # Check if language detection columns exist and find last processed row
                if "predicted_language" in existing_df.columns:
                    # Find the first row that doesn't have results
                    unprocessed = existing_df["predicted_language"].isna()
                    if unprocessed.any():
                        start_idx = unprocessed.idxmax()
                        print(f"  📍 Resuming from row {start_idx} (found existing progress)")
                        # Use the existing dataframe with partial results
                        df = existing_df
                    else:
                        print(f"  ✅ File already fully processed, skipping...")
                        continue
            except Exception as e:
                print(f"  ⚠️  Could not read existing file, starting fresh: {e}")
        
        # Initialize language detection columns if they don't exist
        if "predicted_language" not in df.columns:
            df["predicted_language"] = None
            df["confidence_score"] = None
            df["top_predicted_labels"] = None
            df["latency"] = None
            
        # Process row by row starting from start_idx
        for idx in range(start_idx, len(df)):
            row = df.iloc[idx]
            text = row[text_col]
            res = detect_language_with_sarvam(text)
            
            # Update the DataFrame with results
            df.at[idx, "predicted_language"] = res[0]
            df.at[idx, "confidence_score"] = res[1]
            df.at[idx, "top_predicted_labels"] = res[2]
            df.at[idx, "latency"] = res[3]
            
            # Save every 50 rows
            if (idx + 1) % 50 == 0:
                df.to_csv(out_file, index=False)
                print(f"  Processed {idx + 1}/{len(df)} rows... 💾 Saved")
                
            # Add batch delay every BATCH_SIZE requests
            if (idx + 1) % BATCH_SIZE == 0:
                time.sleep(BATCH_DELAY)
        
        # Final save for remaining rows
        df.to_csv(out_file, index=False)
        print(f"✅ Completed and saved all results to {out_file}")
        
        # Stats
        file_samples = len(df)
        file_time = df["latency"].sum()
        
        total_samples += file_samples
        total_time += file_time
        
        for lang in df["predicted_language"].dropna():
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        summary_lines.append(f"\nFile: {rel_path}")
        summary_lines.append(f"  • Samples: {file_samples}")
        summary_lines.append(f"  • Avg Latency: {df['latency'].mean():.4f} sec/sample")
        
    # Final summary
    overall_throughput = total_samples / total_time if total_time > 0 else 0
    overall_avg_latency = total_time / total_samples if total_samples > 0 else 0
    
    detected_total = 0
    failed_total = 0
    valid_langs = {}
    
    for lang, count in lang_counts.items():
        if lang in ["UNKNOWN", "ERROR"]:
            failed_total += count
        else:
            detected_total += count
            valid_langs[lang] = count
            
    summary_lines.append("\n--- SARVAM AI ---")
    summary_lines.append(f"Total Comments Processed: {total_samples}")
    
    if total_samples > 0:
        summary_lines.append(f"Blank/Failed Detections: {failed_total} ({failed_total/total_samples*100:.2f}%)")
        summary_lines.append(f"Successful Detections: {detected_total} ({detected_total/total_samples*100:.2f}%)\n")
        
        summary_lines.append("Language Breakdown:")
        sorted_langs = sorted(valid_langs.items(), key=lambda x: x[1], reverse=True)
        for lang, count in sorted_langs:
            summary_lines.append(f"  {lang}: {count} ({count/total_samples*100:.2f}%)")
            
    summary_lines.append("\n" + "="*40)
    summary_lines.append("📈 Overall Benchmark Results:")
    summary_lines.append(f"  • Total Files Processed: {len(csv_files)}")
    summary_lines.append(f"  • Total Samples Processed: {total_samples}")
    summary_lines.append(f"  • Overall Average Latency: {overall_avg_latency:.4f} sec/sample")
    summary_lines.append(f"  • Overall Throughput: {overall_throughput:.2f} samples/sec")
    
    summary_text = "\n".join(summary_lines)
    print(summary_text) 
    
    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"\n✅ Summary saved to {SUMMARY_FILE}")

if __name__ == "__main__":
    main()
