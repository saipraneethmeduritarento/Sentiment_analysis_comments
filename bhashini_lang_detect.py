import os
import time
import pandas as pd
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load env variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "input_data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "language_detection", "bhashini")
SUMMARY_FILE = os.path.join(BASE_DIR, "output", "language_detection", "Bhashini_summary.txt")

# API endpoint
API_URL = "https://meity-auth.ulcacontrib.org/ulca/apis/v0/model/compute"

# Credentials from .env
BHASHINI_USER_ID = os.getenv("BHASHINI_USER_ID", None)
BHASHINI_AUTH_TOKEN = os.getenv("BHASHINI_AUTH_TOKEN", "")

def detect_language_with_ulca(text):
    if pd.isna(text) or not str(text).strip():
        return "en", 0.0, "UNKNOWN", 0.0
    
    payload = {
        "modelId": "631736990154d6459973318e",   # lang detection modelId
        "task": "txt-lang-detection",
        "input": [{"source": str(text)}],
        "userId": BHASHINI_USER_ID
    }
    headers = {
        "accept": "*/*",
        "content-type": "application/json",
        "origin": "https://bhashini.gov.in",
        "referer": "https://bhashini.gov.in/",
        "user-agent": "python-requests"
    }
    if BHASHINI_AUTH_TOKEN:
        headers["Authorization"] = BHASHINI_AUTH_TOKEN
    elif os.getenv("BHASHINI_API_KEY"):
        headers["Authorization"] = os.getenv("BHASHINI_API_KEY")
    
    try:
        start = time.time()
        response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
        latency = time.time() - start
        
        response.raise_for_status()
        data = response.json()
        
        predictions = data.get("output", [])[0].get("langPrediction", [])
        if not predictions:
            return "en", 0.0, "UNKNOWN", latency
        
        predictions = sorted(predictions, key=lambda x: x["langScore"], reverse=True)
        best = predictions[0]

        lang_code = best["langCode"] if best["langCode"] != "unknown" else "hinglish"
        conf_str = "; ".join([f"{p['langCode']}:{p['langScore']:.4f}" for p in predictions])
        return lang_code, float(best["langScore"]), conf_str, latency
    
    except Exception as e:
        print(f"❌ Error for text: {str(text)[:30]}... -> {e}")
        return "ERROR", 0.0, "ERROR", 0.0

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(SUMMARY_FILE), exist_ok=True)
    
    full_comments_dir = Path(INPUT_DIR) / "full_comments"
    csv_files = list(full_comments_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in {full_comments_dir}")
    
    total_samples = 0
    total_time = 0.0
    lang_counts = {}
    
    summary_lines = []
    summary_lines.append("📊 Bhashini Language Detection Summary")
    summary_lines.append("="*40)
    
    for file_path in csv_files:
        print(f"\nProcessing: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Check text column
        text_col = "comment" if "comment" in df.columns else ("text" if "text" in df.columns else None)
        if not text_col:
            print(f"⚠️ Skipping {file_path.name}: No applicable text column found.")
            continue
            
        # We process row by row
        results = []
        for idx, row in df.iterrows():
            text = row[text_col]
            res = detect_language_with_ulca(text)
            results.append(res)
            
            if (idx + 1) % 50 == 0:
                print(f"  Processed {idx + 1}/{len(df)} rows...")
                
        df[["predicted_language", "confidence_score", "top_predicted_labels", "latency"]] = pd.DataFrame(results, index=df.index)
        
        # Save to output folder
        out_file = Path(OUTPUT_DIR) / "full_comments" / file_path.name
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(out_file, index=False)
        print(f"✅ Saved results to {out_file}")
        
        # Stats
        file_samples = len(df)
        file_time = df["latency"].sum()
        
        total_samples += file_samples
        total_time += file_time
        
        for lang in df["predicted_language"].dropna():
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        
        summary_lines.append(f"\nFile: {file_path.name}")
        summary_lines.append(f"  • Samples: {file_samples}")
        summary_lines.append(f"  • Avg Latency: {df['latency'].mean():.4f} sec/sample")
        
    # Final summary
    overall_throughput = total_samples / total_time if total_time > 0 else 0
    overall_avg_latency = total_time / total_samples if total_samples > 0 else 0
    
    detected_total = 0
    failed_total = 0
    valid_langs = {}
    
    for lang, count in lang_counts.items():
        if lang in ["ERROR"]:
            failed_total += count
        else:
            detected_total += count
            valid_langs[lang] = count
            
    summary_lines.append("\n--- BHASHINI ---")
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
 