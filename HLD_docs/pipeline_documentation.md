# Sentiment Analysis Pipeline — Detailed Documentation

## Overview

This document describes the end-to-end pipeline for analyzing learner comments on iGOT course content. The pipeline covers data splitting, sentiment classification, language detection, and deep Gemini-driven analysis with actionable summaries.

---

## High-Level Pipeline Flow

```mermaid
flowchart TD
    A([Raw Comments CSV]) --> B[Step 1 - Split Comments]
    B --> C[comments_with_less_than_3_strings]
    B --> D[full_comments - more than 3 words]
    D --> E[Step 2 - Sentiment Analysis]
    D --> F[Step 3 - Language Detection via Bhashini ]
    E --> G[Labelled Comments - positive / negative / neutral]
    F --> H[Language-Tagged Comments with predicted_language]
    G --> I[Step 4 - Gemini Analysis]
    I --> J[Loop A - Per-Comment Analysis for negative and positive]
    J --> K[Loop B - Category Summary and Action Items]
```

---

## Step 1 — Split Comments by Length

```mermaid
flowchart LR
    A(["Input CSV: content_id, content_name, comment, comment_date"]) --> B{Word count 3 or fewer?}
    B -- Yes --> C[("comments_with_less_than_3_strings / contentId.csv")]
    B -- No --> D[("full_comments / contentId.csv")]
    B -- Empty or blank --> E[Skip row]
    C --> CS(["count_summary.txt"])
    D --> FS(["count_summary.txt"])
```

### What happens here

- The input CSV is read and grouped by `content_id`.
- For each content, a **separate output CSV** is written under both `comments_with_less_than_3_strings/` and `full_comments/`.
- The `comment_date` field is normalised to `YYYY-MM-DD` format (strips time component).
- Comments with zero tokens are silently skipped.
- A `count_summary.txt` is written into each output directory with per-file comment counts.

| Bucket | Condition | Use |
|---|---|---|
| `comments_with_less_than_3_strings/` | `word_count ≤ 3` | Archived; excluded from further analysis |
| `full_comments/` | `word_count > 3` | Fed into Steps 2, 3, and 4 |

---

## Step 2 — Sentiment Analysis

```mermaid
flowchart TD
    A(["input_data/**/*.csv"]) --> B["Load HuggingFace Pipeline: cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual"]
    B --> C["Batch Inference — truncation=True, max_length=512"]
    C --> D{Label Mapping}
    D -- Positive --> E[positive]
    D -- Negative --> F[negative]
    D -- Neutral --> G[neutral]
    D -- "Blank or NaN" --> H[neutral - default fallback]
    E & F & G & H --> I(["Output CSV + predicted sentiment column"])
    I --> J[Save per-course sentiment counts to course_sentiment_counts.txt]
    I --> K{actual sentiment column present?}
    K -- Yes --> L["Per-course classification report (*_classification_report.txt)"]
    K -- Yes --> M["Global report: Accuracy, Recall, Confusion Matrix saved to classification_report.txt"]
    K -- No --> N[Skip evaluation]
```

### Model details

| Property | Value |
|---|---|
| Model | `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual` |
| Task | `sentiment-analysis` |
| Input truncation | `max_length = 512 tokens` |
| Output column | `predicted sentiment` (space-separated) |
| Output labels | `positive`, `negative`, `neutral` |
| Fallback for blank input | Replaced with `"neutral statement"` before inference |

### Outputs

| File | Description |
|---|---|
| `output/<rel_path>/contentId.csv` | Original CSV with appended `predicted sentiment` column |
| `output/course_sentiment_counts.txt` | Per-course positive / neutral / negative counts |
| `output/<rel_path>/contentId_classification_report.txt` | Per-course classification report (if ground-truth present) |
| `output/classification_report.txt` | Aggregated global report with accuracy, recall, and confusion matrix |

### Evaluation (when ground-truth `actual sentiment` column exists)

- Overall accuracy score
- Per-class recall
- Full classification report per course and globally
- Confusion matrix with label breakdown

---

## Step 3 — Language Detection (Bhashini )

```mermaid
flowchart TD
    A(["input_data/full_comments/*.csv"]) --> B[Iterate comments row by row]
    B --> C{Is text blank or NaN?}
    C -- Yes --> D["Return lang=en, score=0.0, labels=UNKNOWN"]
    C -- No --> E["POST to Bhashini  API — task: txt-lang-detection"]
    E --> F[Parse langPrediction array]
    F --> G{langCode is unknown?}
    G -- Yes --> H[Remap to hinglish]
    G -- No --> I[Use returned langCode]
    H & I --> J[Sort by langScore descending]
    J --> K[Top-1 language code and confidence]
    K --> L(["Output CSV + predicted_language, confidence_score, top_predicted_labels, latency"])
    L --> M["Save to output/language_detection/bhashini/full_comments/"]    
    M --> N[Write Bhashini_summary.txt with benchmark and language distribution stats]
```

### API details

| Property | Value |
|---|---|
| Endpoint | `https://meity-auth.contrib.org//apis/v0/model/compute` |
| Model ID | `631736990154d6459973318e` |
| Task type | `txt-lang-detection` |
| Auth | `BHASHINI_USER_ID` + `BHASHINI_AUTH_TOKEN` from `.env` |
| Timeout | 30 seconds per request |
| Error handling | Returns `ERROR / 0.0` on exception; processing continues |

### Fallback and remapping logic

| Condition | Behaviour |
|---|---|
| Blank or NaN text | `predicted_language = en`, `confidence_score = 0.0`, `top_predicted_labels = UNKNOWN` |
| No predictions returned | `predicted_language = en`, `confidence_score = 0.0`, `top_predicted_labels = UNKNOWN` |
| Bhashini returns `unknown` langCode | Remapped to `hinglish` |
| API / network error | `predicted_language = ERROR`, `confidence_score = 0.0` |

### Output fields appended per comment

| Field | Description |
|---|---|
| `predicted_language` | ISO code of the top language (or `hinglish` / `en` / `ERROR`) |
| `confidence_score` | Confidence of top prediction (0–1) |
| `top_predicted_labels` | Semicolon-separated `langCode:score` for all candidates |
| `latency` | Round-trip API time in seconds |

### Benchmark summary (`Bhashini_summary.txt`)

- Per-file: sample count and average latency
- Overall: total files, total samples, average latency, throughput (samples/sec)
- Language distribution: per-language count and percentage
- Separate counts for successful detections vs. failures (`ERROR`)

---

## Step 4 — Gemini Analysis (Two Loops)

```mermaid
flowchart TD
    A(["full_comments CSV with predicted_sentiment"]) --> B[Split by sentiment]
    B --> NEG[Negative comments]
    B --> POS[Positive comments]

    subgraph LOOP_A [Loop A - Per-Comment Analysis]
        NEG --> NA["Gemini: negative_comment_prompt"]
        POS --> PA["Gemini: positive_comment_prompt"]
        NA --> NO(["gemini_analysis/negative/contentId.json"])
        PA --> PO(["gemini_analysis/positive/contentId.json"])
    end

    subgraph LOOP_B [Loop B - Category Summary and Action Items]
        NO --> NS[Group by Issue Category]
        PO --> PS[Group by Positive Theme]
        NS --> NSP["Gemini: negative_summary_action_items prompt"]
        PS --> PSP["Gemini: positive_summary prompt"]
        NSP --> NSO(["negative/contentId_category_summary.json"])
        PSP --> PSO(["positive/contentId_category_summary.json"])
    end
```

---

### Loop A — Per-Comment Analysis

```mermaid
sequenceDiagram
    participant Pipeline
    participant Gemini as Gemini API (Vertex AI)

    loop For each negative comment
        Pipeline->>Gemini: negative_comment_prompt with content_name and comment
        Gemini-->>Pipeline: JSON with Issue Category, Root Cause, Owner, Priority, Recommended Actions
        Pipeline->>Pipeline: Append Gemini Analysis field to row
    end

    loop For each positive comment
        Pipeline->>Gemini: positive_comment_prompt with content_name and comment
        Gemini-->>Pipeline: JSON with Positive Theme, Strength, Highlight, Recommended Actions
        Pipeline->>Pipeline: Append Gemini Analysis field to row
    end

    Pipeline->>Pipeline: Save per-comment JSON to gemini_analysis/negative/ and positive/
```

**Execution model:** Up to `MAX_WORKERS = 10` threads run concurrently via `ThreadPoolExecutor`. Already-processed comments are skipped on re-runs (resume support).

**Token tracking per call:**

| Counter | Source |
|---|---|
| `input` | `prompt_token_count` |
| `output` | `candidates_token_count` |
| `thinking` | `thoughts_token_count` |
| `total` | `total_token_count` |

---

### Loop B — Category Summary & Action Items



```mermaid
flowchart TD
    A(["per-comment JSON — contentId.json"]) --> B[Load all records]
    B --> C[Group by Issue Category or Positive Theme]
    C --> D["Per category: count, % of total, root cause, owner, priority, sample comments"]
    D --> E[Build category data block]
    E --> F["Format summary prompt with content_name, total_count, category_data"]
    F --> G[Gemini - generate category-level summary]
    G --> H{Parse JSON response}
    H -- Success --> I(["Save contentId_category_summary.json"])
    H -- Failure --> J[Log raw output and skip save]
```

**What the summary captures:**

- Overall breakdown of issue categories (negative) or positive themes (positive)
- Per-category percentage contribution
- Consolidated root causes and owners
- Prioritised, actionable recommendations sourced from per-comment Gemini output
- Top representative sample comments per category

---

## End-to-End Data Flow

```mermaid
flowchart LR
    RAW([Raw CSV]) -->|Step 1| SPLIT
    SPLIT --> SHORT[("comments_with_less_than_3_strings/")]
    SPLIT --> FULL[("input_data/full_comments/")]
    FULL -->|Step 2| SENT[("output/ + predicted sentiment")]
    FULL -->|Step 3| LANG[("output/language_detection/bhashini/full_comments/")]
    SENT -->|Step 4A| GA_NEG[("gemini_analysis/negative/*.json")]
    SENT -->|Step 4A| GA_POS[("gemini_analysis/positive/*.json")]
    GA_NEG -->|Step 4B| SUM_NEG[("gemini_analysis/negative/*_category_summary.json")]
    GA_POS -->|Step 4B| SUM_POS[("gemini_analysis/positive/*_category_summary.json")]
```

---

## Configuration Reference

| Parameter | Source | Description |
|---|---|---|
| `sentiment_model` | `.env` | HuggingFace model ID (default: `cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual`) |
| `BHASHINI_USER_ID` | `.env` | Bhashini  user ID |
| `BHASHINI_AUTH_TOKEN` | `.env` | Bhashini API auth token |
| `VERTEX_PROJECT_ID` | `.env` | GCP project for Vertex AI |
| `VERTEX_LOCATION` | `.env` | GCP region for Vertex AI |
| `GEMINI_MODEL` | `.env` | Gemini model name (default: `gemini-2.5-flash`) |
| `MAX_WORKERS` | Hardcoded | Thread pool size for parallel Gemini calls (default: `10`) |
