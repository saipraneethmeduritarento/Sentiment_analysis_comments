"""
Generate per-course, per-category AI summaries and action items from existing top2 Gemini analysis JSONs.

Reads:
  top2_gemini_analysis/negative/do_*.json  — per-comment negative analysis
  top2_gemini_analysis/positive/do_*.json  — per-comment positive analysis

Writes:
  top2_gemini_analysis/negative/do_*_category_summary.json
  top2_gemini_analysis/positive/do_*_category_summary.json
"""

import os
import glob
import json
import logging
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GEMINI_NEG_DIR = os.path.join(BASE_DIR, "top2_gemini_analysis", "negative")
GEMINI_POS_DIR = os.path.join(BASE_DIR, "top2_gemini_analysis", "positive")
NEG_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "negative_sumamry_action_items.yaml")
POS_PROMPT_FILE = os.path.join(BASE_DIR, "prompt", "positive_summary.yaml")

PROJECT_ID = os.environ.get("VERTEX_PROJECT_ID")
LOCATION = os.environ.get("VERTEX_LOCATION")
MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

client = genai.Client(vertexai=True, project=PROJECT_ID, location=LOCATION)


def _load_prompt(yaml_file: str, version: str = "v1") -> str:
    with open(yaml_file, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get(version, data.get("prompt", ""))


def _group_by_category(analysis_data: list, category_key: str) -> dict:
    """Group per-comment analysis records by their category value."""
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
                "sample_actions": (
                    ga.get("Recommended Actions", []) if isinstance(ga, dict) else []
                ),
            }
        categories[cat]["count"] += 1
        categories[cat]["comments"].append(item.get("comment", "").strip())
    return categories


def _build_category_data_str(categories: dict, total_count: int) -> str:
    """Build a human-readable block of category data to embed in the prompt."""
    lines = []
    for cat_name, cat in sorted(
        categories.items(), key=lambda x: x[1]["count"], reverse=True
    ):
        pct = (cat["count"] / max(1, total_count)) * 100
        lines.append(f"\n### Category: {cat_name}")
        lines.append(f"- Comment count: {cat['count']} ({pct:.1f}% of total)")
        lines.append(f"- Root Cause: {cat['root_cause']}")
        lines.append(f"- Owner: {cat['owner']}")
        lines.append(f"- Priority: {cat['priority']}")
        sample_actions = cat["sample_actions"][:3]
        if sample_actions:
            lines.append(
                f"- Sample Recommended Actions: {', '.join(sample_actions)}"
            )
        lines.append("- Sample Comments (up to 5):")
        for comment in cat["comments"][:5]:
            lines.append(f'  * "{comment}"')
    return "\n".join(lines)


def _call_gemini(prompt_text: str) -> tuple[dict | None, dict]:
    usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )
        if response.usage_metadata:
            usage["input"] = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
            usage["output"] = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
            usage["thinking"] = getattr(response.usage_metadata, "thoughts_token_count", 0) or 0
            usage["total"] = getattr(response.usage_metadata, "total_token_count", 0) or 0
        return json.loads(response.text.strip()), usage
    except json.JSONDecodeError as e:
        log.error("JSON parse error from Gemini: %s", e)
        return None, usage
    except Exception as e:
        log.error("Gemini API error: %s", e)
        return None, usage


def _run_summary(
    json_path: str,
    prompt_template: str,
    category_key: str,
    label: str,
) -> dict | None:
    summary_path = json_path.replace(".json", "_category_summary.json")
    if os.path.exists(summary_path):
        log.info("Summary already exists, skipping: %s", os.path.basename(summary_path))
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f), {"input": 0, "output": 0, "thinking": 0, "total": 0}

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data:
        log.warning("Empty analysis file: %s", json_path)
        return None, {"input": 0, "output": 0, "thinking": 0, "total": 0}

    content_name = data[0].get("content_name", "")
    categories = _group_by_category(data, category_key)
    total_count = len(data)
    category_data_str = _build_category_data_str(categories, total_count)

    prompt = prompt_template.format(
        content_name=content_name,
        total_count=total_count,
        category_data=category_data_str,
    )

    log.info("Generating %s category summary for: %s", label, content_name)
    result, usage = _call_gemini(prompt)
    if result:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        log.info("Saved: %s", os.path.basename(summary_path))
    return result, usage


def main() -> None:
    neg_prompt = _load_prompt(NEG_PROMPT_FILE, "v3")
    pos_prompt = _load_prompt(POS_PROMPT_FILE, "v3")

    total_usage = {"input": 0, "output": 0, "thinking": 0, "total": 0}
    run_details: list[dict] = []

    for json_file in sorted(glob.glob(os.path.join(GEMINI_NEG_DIR, "*.json"))):
        if "_category_summary" not in json_file:
            result, usage = _run_summary(json_file, neg_prompt, "Issue Category", "negative")
            if usage["total"] > 0:
                course_id = os.path.splitext(os.path.basename(json_file))[0]
                run_details.append({"file": f"{course_id} (negative)", **usage})
                for k in total_usage:
                    total_usage[k] += usage[k]

    for json_file in sorted(glob.glob(os.path.join(GEMINI_POS_DIR, "*.json"))):
        if "_category_summary" not in json_file:
            result, usage = _run_summary(json_file, pos_prompt, "Strength Category", "positive")
            if usage["total"] > 0:
                course_id = os.path.splitext(os.path.basename(json_file))[0]
                run_details.append({"file": f"{course_id} (positive)", **usage})
                for k in total_usage:
                    total_usage[k] += usage[k]

    lines = []
    lines.append("Token Usage Summary — top2_category_summary")
    lines.append("=" * 52)
    for detail in run_details:
        lines.append(f"\n  {detail['file']}")
        lines.append(f"    input:    {detail['input']:,}")
        lines.append(f"    output:   {detail['output']:,}")
        lines.append(f"    thinking: {detail['thinking']:,}")
        lines.append(f"    total:    {detail['total']:,}")
    lines.append("\n" + "-" * 52)
    lines.append("  GRAND TOTAL")
    lines.append(f"    input:    {total_usage['input']:,}")
    lines.append(f"    output:   {total_usage['output']:,}")
    lines.append(f"    thinking: {total_usage['thinking']:,}")
    lines.append(f"    total:    {total_usage['total']:,}")
    lines.append("=" * 52)

    summary_text = "\n".join(lines)
    print("\n" + summary_text)

    summary_txt_path = os.path.join(BASE_DIR, "top2_gemini_analysis", "category_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write(summary_text + "\n")
    log.info("Token summary saved to: %s", summary_txt_path)
    log.info("All category summaries generated.")


if __name__ == "__main__":
    main()
