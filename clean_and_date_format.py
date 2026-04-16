import csv
import os

SOURCE_FILE = 'new_data/top 10 course comments.csv'
BASE_DEST_DIR = 'new_data/input_data'

SHORT_COMMENTS_DIR = os.path.join(BASE_DEST_DIR, 'comments_with_less_than_3_strings')
FULL_COMMENTS_DIR = os.path.join(BASE_DEST_DIR, 'full_comments')

COLUMNS = ["content_id", "content_name", "comment", "comment_date"]

# Ensure directories exist
os.makedirs(SHORT_COMMENTS_DIR, exist_ok=True)
os.makedirs(FULL_COMMENTS_DIR, exist_ok=True)

print(f"Processing file: {SOURCE_FILE}...")

if not os.path.exists(SOURCE_FILE):
    print(f"Source file not found: {SOURCE_FILE}")
    exit(1)

# Group comments by content_id
comments_by_content = {}
try:
    with open(SOURCE_FILE, newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            content_id = row.get("content_id", "").strip()
            if not content_id:
                continue
            if content_id not in comments_by_content:
                comments_by_content[content_id] = []
            comments_by_content[content_id].append(row)
except Exception as e:
    print(f"Error reading source file: {e}")
    exit(1)

short_counts_summary = {}
full_counts_summary = {}

for content_id, rows in comments_by_content.items():
    filename = f"{content_id}.csv"
    short_output_path = os.path.join(SHORT_COMMENTS_DIR, filename)
    full_output_path = os.path.join(FULL_COMMENTS_DIR, filename)

    print(f"Processing {content_id} ({len(rows)} comments)...")

    short_count = 0
    full_count = 0
    skipped_count = 0

    with open(short_output_path, "w", newline="", encoding="utf-8") as short_outfile, \
         open(full_output_path, "w", newline="", encoding="utf-8") as full_outfile:

        short_writer = csv.DictWriter(short_outfile, fieldnames=COLUMNS)
        short_writer.writeheader()

        full_writer = csv.DictWriter(full_outfile, fieldnames=COLUMNS)
        full_writer.writeheader()

        for row in rows:
            comment = row.get("comment", "").strip()
            if not comment:
                skipped_count += 1
                continue

            word_count = len(comment.split())

            cleaned_row = {}
            for col in COLUMNS:
                val = row.get(col, "").strip()
                if col == "comment_date":
                    if val:
                        if "T" in val:
                            val = val.split("T")[0]
                        elif " " in val:
                            val = val.split(" ")[0]
                cleaned_row[col] = val

            if word_count <= 3:
                short_writer.writerow(cleaned_row)
                short_count += 1
            else:
                full_writer.writerow(cleaned_row)
                full_count += 1

    print(f"  Short comments (<=3 words): {short_count}")
    print(f"  Full comments (>3 words): {full_count}")
    print(f"  Skipped empty: {skipped_count}")

    short_counts_summary[filename] = short_count
    full_counts_summary[filename] = full_count

# Write summary reports
with open(os.path.join(SHORT_COMMENTS_DIR, 'count_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("Comments (<= 3 words) count summary:\n")
    f.write("-" * 40 + "\n")
    total_short = 0
    for fname, count in short_counts_summary.items():
        f.write(f"{fname}: {count}\n")
        total_short += count
    f.write("-" * 40 + "\n")
    f.write(f"Total: {total_short}\n")

with open(os.path.join(FULL_COMMENTS_DIR, 'count_summary.txt'), 'w', encoding='utf-8') as f:
    f.write("Comments (> 3 words) count summary:\n")
    f.write("-" * 40 + "\n")
    total_full = 0
    for fname, count in full_counts_summary.items():
        f.write(f"{fname}: {count}\n")
        total_full += count
    f.write("-" * 40 + "\n")
    f.write(f"Total: {total_full}\n")

print("Processing complete.")



