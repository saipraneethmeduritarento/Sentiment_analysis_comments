import os
import glob
import pandas as pd
import yaml
import json
import time
import threading
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Paths
INPUT_DIRS = [
    os.path.join(BASE_DIR, "output", "full_comments"),
]
OUTPUT_DIR_NEGATIVE = os.path.join(BASE_DIR, "gemini_analysis", "negative")
OUTPUT_DIR_POSITIVE = os.path.join(BASE_DIR, "gemini_analysis", "positive")
NEGATIVE_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "negative_comment_prompt.yaml")
POSITIVE_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "positive_comment_prompt.yaml")
NEGATIVE_SUMMARY_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "negative_sumamry_action_items.yaml")
POSITIVE_SUMMARY_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "positive_summary.yaml")

# Gemini settings from .env
PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
LOCATION = os.environ.get("VERTEX_LOCATION")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

# Initialize Gemini Client for Vertex AI
client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)

def load_prompts() -> tuple[str, str, str, str]:
    """Load all four prompt templates from their YAML files.

    Returns:
        Tuple of (negative_comment, positive_comment, negative_summary, positive_summary)
        prompt strings.
    """
    with open(NEGATIVE_PROMPT_FILE, 'r') as file:
        neg_data = yaml.safe_load(file)
        negative_prompt = neg_data.get('v3', neg_data.get('prompt', ''))
    with open(POSITIVE_PROMPT_FILE, 'r') as file:
        pos_data = yaml.safe_load(file)
        positive_prompt = pos_data.get('v1', pos_data.get('prompt', ''))
    with open(NEGATIVE_SUMMARY_PROMPT_FILE, 'r') as file:
        neg_sum_data = yaml.safe_load(file)
        negative_summary_prompt = neg_sum_data.get('v3', neg_sum_data.get('prompt', ''))
    with open(POSITIVE_SUMMARY_PROMPT_FILE, 'r') as file:
        pos_sum_data = yaml.safe_load(file)
        positive_summary_prompt = pos_sum_data.get('v3', pos_sum_data.get('prompt', ''))
    return negative_prompt, positive_prompt, negative_summary_prompt, positive_summary_prompt

def analyze_with_gemini(prompt_text):
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            )
        )
        raw_text = response.text.strip()
        
        usage = {
            "input": 0,
            "output": 0,
            "thinking": 0,
            "total": 0
        }
        
        if response.usage_metadata:
            usage["input"] = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage["output"] = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            usage["total"] = getattr(response.usage_metadata, "total_token_count", 0) or 0
            # Try to get thinking tokens if available
            # It might be present in newer versions or under candidates_token_details
            usage["thinking"] = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
        
        try:
            result = json.loads(raw_text)
            return result, usage
        except json.JSONDecodeError:
            print(f"Failed to parse JSON from Gemini. Raw: {raw_text}")
            return {"Error": "Failed to parse JSON", "Raw Output": raw_text}, usage
    except Exception as e:
        print(f"Error communicating with Gemini: {e}")
        return {"Error": "Could not analyze"}, None

MAX_WORKERS = 10  # Number of parallel threads for Gemini API calls


def _group_by_category(analysis_data: list, category_key: str) -> dict:
    """Group per-comment analysis records by their category value.

    Args:
        analysis_data: List of per-comment analysis dicts loaded from JSON.
        category_key: The Gemini Analysis field to group by (e.g. "Issue Category").

    Returns:
        Dict keyed by category name, each holding count, comments, and metadata.
    """
    categories: dict = {}
    for item in analysis_data:
        ga = item.get("Gemini Analysis", {})
        cat = ga.get(category_key, "Other") if isinstance(ga, dict) else "Other"
        if cat not in categories:
            categories[cat] = {
                "category": cat,
                "count": 0,
                "comments": [],
                "root_cause": ga.get("Root Cause", "") if isinstance(ga, dict) else "",
                "owner": ga.get("Owner", "") if isinstance(ga, dict) else "",
                "priority": ga.get("Priority", "") if isinstance(ga, dict) else "",
                "sample_actions": ga.get("Recommended Actions", []) if isinstance(ga, dict) else [],
            }
        categories[cat]["count"] += 1
        categories[cat]["comments"].append(item.get("comment", "").strip())
    return categories


def _build_category_data_str(categories: dict, total_count: int) -> str:
    """Build a human-readable block of category data to embed in a summary prompt.

    Args:
        categories: Dict from :func:`_group_by_category`.
        total_count: Total number of comments (used for percentage calculation).

    Returns:
        Formatted multi-line string describing all categories.
    """
    lines = []
    for cat_name, cat in sorted(categories.items(), key=lambda x: x[1]["count"], reverse=True):
        pct = (cat["count"] / max(1, total_count)) * 100
        lines.append(f"\n### Category: {cat_name}")
        lines.append(f"- Comment count: {cat['count']} ({pct:.1f}% of total)")
        lines.append(f"- Root Cause: {cat['root_cause']}")
        lines.append(f"- Owner: {cat['owner']}")
        lines.append(f"- Priority: {cat['priority']}")
        sample_actions = cat["sample_actions"][:3]
        if sample_actions:
            lines.append(f"- Sample Recommended Actions: {', '.join(sample_actions)}")
        lines.append("- Sample Comments (up to 5):")
        for comment in cat["comments"][:5]:
            lines.append(f'  * "{comment}"')
    return "\n".join(lines)


def _run_category_summary(
    json_path: str,
    prompt_template: str,
    category_key: str,
    label: str,
) -> tuple[dict | None, dict]:
    """Generate and save a category-level summary for one per-comment analysis JSON.

    Skips generation if the output ``_category_summary.json`` already exists.

    Args:
        json_path: Path to the per-comment analysis JSON.
        prompt_template: Formatted prompt string with ``{content_name}``,
            ``{total_count}``, and ``{category_data}`` placeholders.
        category_key: Gemini Analysis field to group comments by.
        label: Human-readable label used in log messages (e.g. "negative").

    Returns:
        Tuple of (parsed summary dict or None, token usage dict).
    """
    empty_usage: dict = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    summary_path = json_path.replace(".json", "_category_summary.json")

    if os.path.exists(summary_path):
        print(f"Category summary already exists, skipping: {os.path.basename(summary_path)}")
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f), empty_usage

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not data:
        print(f"Empty analysis file, skipping: {os.path.basename(json_path)}")
        return None, empty_usage

    content_name = data[0].get("content_name", "")
    categories = _group_by_category(data, category_key)
    total_count = len(data)
    category_data_str = _build_category_data_str(categories, total_count)

    prompt = prompt_template.format(
        content_name=content_name,
        total_count=total_count,
        category_data=category_data_str,
    )

    print(f"Generating {label} category summary for: {content_name}")
    result, usage = analyze_with_gemini(prompt)
    if result and "Error" not in result:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        print(f"Saved: {os.path.basename(summary_path)}")
    return result, usage

def process_csv(
    csv_path: str,
    negative_prompt_template: str,
    positive_prompt_template: str,
    neg_summary_prompt_template: str,
    pos_summary_prompt_template: str,
) -> None:
    """Process one CSV file: run per-comment analysis then category summaries.

    Splits comments by sentiment, calls Gemini for each comment in parallel,
    then generates category-level summaries.  Token usage is tracked separately
    for each of the four prompt types.

    Args:
        csv_path: Path to the input CSV file.
        negative_prompt_template: Per-comment prompt for negative sentiments.
        positive_prompt_template: Per-comment prompt for positive sentiments.
        neg_summary_prompt_template: Category-summary prompt for negative groups.
        pos_summary_prompt_template: Category-summary prompt for positive groups.
    """
    print(f"Processing {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Failed to read {csv_path}: {e}")
        return

    required_cols = ['content_id', 'content_name', 'comment', 'predicted sentiment']
    if not all(col in df.columns for col in required_cols):
        print(f"Skipping {csv_path}: missing required columns.")
        return

    os.makedirs(OUTPUT_DIR_NEGATIVE, exist_ok=True)
    os.makedirs(OUTPUT_DIR_POSITIVE, exist_ok=True)

    base_filename = os.path.basename(csv_path).replace('.csv', '.json')
    out_path_negative = os.path.join(OUTPUT_DIR_NEGATIVE, base_filename)
    out_path_positive = os.path.join(OUTPUT_DIR_POSITIVE, base_filename)

    # Split df by sentiment
    df_negative = df[df['predicted sentiment'].str.strip().str.lower() == 'negative'].copy()
    df_positive = df[df['predicted sentiment'].str.strip().str.lower() == 'positive'].copy()

    # Load existing outputs if they exist to resume progress
    if os.path.exists(out_path_negative):
        print(f"Resuming negative from existing {out_path_negative}...")
        df_resume_neg = pd.read_json(out_path_negative, orient='records')
        # Re-apply sentiment filter in case old file had all records
        if 'predicted sentiment' in df_resume_neg.columns:
            df_resume_neg = df_resume_neg[df_resume_neg['predicted sentiment'].str.strip().str.lower() == 'negative'].copy()
        # Merge resume state: keep analysis from resumed file, structure from filtered df
        df_negative = df_negative.merge(
            df_resume_neg[['content_id', 'comment', 'Gemini Analysis']],
            on=['content_id', 'comment'], how='left'
        ) if 'Gemini Analysis' in df_resume_neg.columns else df_negative
        if 'Gemini Analysis' not in df_negative.columns:
            df_negative['Gemini Analysis'] = None
    else:
        if 'Gemini Analysis' not in df_negative.columns:
            df_negative['Gemini Analysis'] = None

    if os.path.exists(out_path_positive):
        print(f"Resuming positive from existing {out_path_positive}...")
        df_resume_pos = pd.read_json(out_path_positive, orient='records')
        # Re-apply sentiment filter in case old file had all records
        if 'predicted sentiment' in df_resume_pos.columns:
            df_resume_pos = df_resume_pos[df_resume_pos['predicted sentiment'].str.strip().str.lower() == 'positive'].copy()
        # Merge resume state: keep analysis from resumed file, structure from filtered df
        df_positive = df_positive.merge(
            df_resume_pos[['content_id', 'comment', 'Gemini Analysis']],
            on=['content_id', 'comment'], how='left'
        ) if 'Gemini Analysis' in df_resume_pos.columns else df_positive
        if 'Gemini Analysis' not in df_positive.columns:
            df_positive['Gemini Analysis'] = None
    else:
        if 'Gemini Analysis' not in df_positive.columns:
            df_positive['Gemini Analysis'] = None

    token_lock = threading.Lock()
    neg_comment_tokens: dict = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    pos_comment_tokens: dict = {"input": 0, "output": 0, "thinking": 0, "total": 0}

    def process_row(index, row, prompt_template, sentiment_label):
        comment = row['comment']
        content_name = row['content_name']

        if pd.isna(comment) or not str(comment).strip():
            return index, None, None

        existing = row.get('Gemini Analysis')
        if pd.notna(existing) and bool(existing):
            return index, None, None  # already analyzed, skip

        prompt_text = prompt_template.format(
            content_name=content_name,
            comment=comment,
            predicted_sentiment=sentiment_label
        )
        analysis, usage = analyze_with_gemini(prompt_text)
        return index, analysis, usage

    def run_parallel(df_subset, prompt_template, out_path, sentiment_label, desc, token_dict):
        save_lock = threading.Lock()

        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for index, row in df_subset.iterrows():
                future = executor.submit(process_row, index, row, prompt_template, sentiment_label)
                futures[future] = index

            for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
                index, analysis, usage = future.result()
                if analysis is not None:
                    df_subset.at[index, 'Gemini Analysis'] = analysis
                    if usage:
                        with token_lock:
                            token_dict["input"] += usage['input']
                            token_dict["output"] += usage['output']
                            token_dict["thinking"] += usage['thinking']
                            token_dict["total"] += usage['total']
                    with save_lock:
                        df_subset.to_json(out_path, orient='records', indent=4, force_ascii=False)

        return df_subset

    # --- Process Negative Comments ---
    df_negative = run_parallel(
        df_negative, negative_prompt_template, out_path_negative,
        'negative', f"Negative comments: {os.path.basename(csv_path)}", neg_comment_tokens,
    )

    # --- Process Positive Comments ---
    df_positive = run_parallel(
        df_positive, positive_prompt_template, out_path_positive,
        'positive', f"Positive comments: {os.path.basename(csv_path)}", pos_comment_tokens,
    )

    # --- Negative Category Summary ---
    neg_summary_tokens: dict = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    if os.path.exists(out_path_negative):
        _, usage = _run_category_summary(
            out_path_negative, neg_summary_prompt_template, "Issue Category", "negative"
        )
        for k in neg_summary_tokens:
            neg_summary_tokens[k] += usage[k]

    # --- Positive Category Summary ---
    pos_summary_tokens: dict = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    if os.path.exists(out_path_positive):
        _, usage = _run_category_summary(
            out_path_positive, pos_summary_prompt_template, "Strength Category", "positive"
        )
        for k in pos_summary_tokens:
            pos_summary_tokens[k] += usage[k]

    summary_dir = os.path.join(BASE_DIR, "gemini_analysis")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, os.path.basename(csv_path).replace('.csv', '_summary.txt'))

    base_name = os.path.basename(csv_path)
    mode = 'a' if os.path.exists(summary_path) else 'w'
    with open(summary_path, mode) as f:
        if mode == 'w':
            f.write(f"Token Usage Summary for {base_name}\n")
            f.write("=" * 50 + "\n")
        f.write("\n--- Run Update ---\n")
        f.write("\n[1] Negative comment analysis\n")
        f.write(f"  input:    {neg_comment_tokens['input']}\n")
        f.write(f"  output:   {neg_comment_tokens['output']}\n")
        f.write(f"  thinking: {neg_comment_tokens['thinking']}\n")
        f.write(f"  total:    {neg_comment_tokens['total']}\n")
        f.write("\n[2] Positive comment analysis\n")
        f.write(f"  input:    {pos_comment_tokens['input']}\n")
        f.write(f"  output:   {pos_comment_tokens['output']}\n")
        f.write(f"  thinking: {pos_comment_tokens['thinking']}\n")
        f.write(f"  total:    {pos_comment_tokens['total']}\n")
        f.write("\n[3] Negative category summary\n")
        f.write(f"  input:    {neg_summary_tokens['input']}\n")
        f.write(f"  output:   {neg_summary_tokens['output']}\n")
        f.write(f"  thinking: {neg_summary_tokens['thinking']}\n")
        f.write(f"  total:    {neg_summary_tokens['total']}\n")
        f.write("\n[4] Positive category summary\n")
        f.write(f"  input:    {pos_summary_tokens['input']}\n")
        f.write(f"  output:   {pos_summary_tokens['output']}\n")
        f.write(f"  thinking: {pos_summary_tokens['thinking']}\n")
        f.write(f"  total:    {pos_summary_tokens['total']}\n")

    print(f"Negative output saved to {out_path_negative}")
    print(f"Positive output saved to {out_path_positive}")
    print(f"Token summary saved to {summary_path}\n")

def main() -> None:
    """Entry point: load prompts and process all CSVs in each input directory."""
    neg_comment, pos_comment, neg_summary, pos_summary = load_prompts()
    if not neg_comment:
        print("Warning: Could not load negative comment prompt (v3) from YAML.")
    if not pos_comment:
        print("Warning: Could not load positive comment prompt (v1) from YAML.")
    if not neg_summary:
        print("Warning: Could not load negative summary prompt (v3) from YAML.")
    if not pos_summary:
        print("Warning: Could not load positive summary prompt (v3) from YAML.")

    for target_dir in INPUT_DIRS:
        if not os.path.exists(target_dir):
            print(f"Directory {target_dir} does not exist. Skipping.")
            continue

        csv_files = glob.glob(os.path.join(target_dir, "*.csv"))
        for csv_file in csv_files:
            process_csv(csv_file, neg_comment, pos_comment, neg_summary, pos_summary)

if __name__ == "__main__":
    main()
