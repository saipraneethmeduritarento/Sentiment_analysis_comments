import os
import json
import csv
import glob
from jinja2 import Environment, FileSystemLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_FULL_DIR = os.path.join(BASE_DIR, 'output', 'full_comments')
SARVAM_FULL_DIR = os.path.join(BASE_DIR, 'output', 'language_detection', 'sarvam')
GEMINI_DIR = os.path.join(BASE_DIR, 'gemini_analysis')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'ui', 'template')
TEMPLATE_FILE = 'dashboard_template_v1.html'
OUTPUT_FILE = os.path.join(BASE_DIR, 'ui', 'output', 'sentiment_analysis_report_sarvam.html')

LANG_MAP = {
    # Sarvam uses en-IN style codes
    'en-in': 'English', 'hi-in': 'Hindi', 'ta-in': 'Tamil', 'te-in': 'Telugu',
    'mr-in': 'Marathi', 'bn-in': 'Bengali', 'gu-in': 'Gujarati', 'kn-in': 'Kannada',
    'ml-in': 'Malayalam', 'or-in': 'Odia', 'pa-in': 'Punjabi', 'as-in': 'Assamese',
    'ur-in': 'Urdu',
    # Also handle plain codes as fallback
    'hi': 'Hindi', 'en': 'English', 'ta': 'Tamil', 'te': 'Telugu',
    'mr': 'Marathi', 'bn': 'Bengali', 'gu': 'Gujarati', 'kn': 'Kannada',
    'ml': 'Malayalam', 'or': 'Odia', 'pa': 'Punjabi', 'as': 'Assamese',
    'ur': 'Urdu', 'unknown': 'Unknown'
}

# Per-language accumulator structure
def empty_lang_bucket():
    return {
        'positive': 0, 'negative': 0, 'neutral': 0,
        'issues': {},
        'courses': {}   # course_id -> {name, pos, neg, neu, langs set}
    }

def analyze_dataset(sentiment_dir, sarvam_dir, gemini_dir):
    flat_buckets = []
    course_stats = []
    issue_counts = {}
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    language_counts = {}
    language_sentiment = {}  # {lang_name: {pos, neg, neu}}
    total_valid_comments = 0
    
    # Build a lookup: course_id -> {comment_text -> sentiment + issue + analysis}
    comment_sentiment_map = {}

    # Load Gemini analysis JSONs into a lookup: course_id -> {norm_comment -> analysis_dict}
    gemini_lookup = {}
    if os.path.exists(gemini_dir):
        for gfile in glob.glob(os.path.join(gemini_dir, '*.json')):
            course_id = os.path.splitext(os.path.basename(gfile))[0]
            try:
                with open(gfile, 'r', encoding='utf-8') as f:
                    gdata = json.load(f)
                gemini_lookup[course_id] = {}
                for item in gdata:
                    raw_c = str(item.get('comment', ''))
                    norm_c = " ".join(raw_c.lower().split())
                    ga = item.get('Gemini Analysis', {})
                    if isinstance(ga, dict):
                        gemini_lookup[course_id][norm_c] = ga
            except Exception as e:
                print(f"Warning: Could not load gemini file {gfile}: {e}")

    # Read sentiment CSV files
    csv_files = glob.glob(os.path.join(sentiment_dir, '*.csv'))
    
    for csv_file in csv_files:
        course_id = os.path.splitext(os.path.basename(csv_file))[0]
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        course_name = ""
        c_pos, c_neg, c_neu = 0, 0, 0
        c_issues = {}
        
        for item in rows:
            if not course_name:
                course_name = item.get('content_name', course_id)
            
            comment_text = str(item.get('comment', ''))
            total_valid_comments += 1
            sentiment = str(item.get('predicted sentiment', 'neutral')).strip().lower()
            
            # Get Gemini analysis for this comment
            norm_comment = " ".join(comment_text.lower().split())
            gemini_analysis = gemini_lookup.get(course_id, {}).get(norm_comment, {})
            
            issue = 'Other Issue'
            if isinstance(gemini_analysis, dict):
                detected_issue = gemini_analysis.get('Issue Category', 'Other Issue')
                if detected_issue and detected_issue != 'Other Issue':
                    issue = detected_issue

            if sentiment == 'positive':
                c_pos += 1
                sentiment_counts['positive'] += 1
            elif sentiment == 'negative':
                c_neg += 1
                sentiment_counts['negative'] += 1
                if issue != 'Other Issue':
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    c_issues[issue] = c_issues.get(issue, 0) + 1
            else:
                c_neu += 1
                sentiment_counts['neutral'] += 1

            # Store for language join
            if course_id not in comment_sentiment_map:
                comment_sentiment_map[course_id] = {}
            comment_sentiment_map[course_id][norm_comment] = {
                'sentiment': sentiment,
                'issue': issue,
                'analysis': gemini_analysis,
                'date': item.get('comment_date', '')
            }
                
        # Get language data from corresponding Sarvam CSV
        c_langs_counts = {}
        course_lang_buckets = {}
        csv_path = os.path.join(sarvam_dir, f"{course_id}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    comment_text = str(row.get('comment', ''))
                    lang_code = str(row.get('predicted_language', 'unknown')).strip().lower()
                    lang_name = LANG_MAP.get(lang_code, lang_code.title())
                    
                    c_langs_counts[lang_code] = c_langs_counts.get(lang_code, 0) + 1

                    # Per language sentiment join
                    raw_csv_comment = str(row.get('comment', ''))
                    norm_csv_comment = " ".join(raw_csv_comment.lower().split())
                    
                    comment_data = comment_sentiment_map.get(course_id, {}).get(norm_csv_comment, {'sentiment': 'neutral', 'issue': 'Other Issue', 'analysis': {}})
                    sent = comment_data['sentiment']
                    
                    if lang_name not in language_sentiment:
                        language_sentiment[lang_name] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    language_sentiment[lang_name][sent] += 1
                    
                    c_date = comment_data.get('date', row.get('comment_date', ''))
                    c_lang = lang_name

                    # Track for flat grouping
                    if lang_name not in course_lang_buckets:
                        course_lang_buckets[lang_name] = {
                            'pos': 0, 'neg': 0, 'neu': 0, 'issues': {}, 
                            'negative_comments': [], 'positive_comments': [], 'neutral_comments': []
                        }
                        
                    if sent == 'positive':
                        course_lang_buckets[lang_name]['pos'] += 1
                        course_lang_buckets[lang_name]['positive_comments'].append({
                            'comment': comment_text.strip(),
                            'date': c_date,
                            'language': c_lang,
                            'sentiment': sent,
                            'analysis': comment_data.get('analysis', {})
                        })
                    elif sent == 'negative':
                        course_lang_buckets[lang_name]['neg'] += 1
                        issue = comment_data['issue']
                        if issue != 'Other Issue':
                            course_lang_buckets[lang_name]['issues'][issue] = course_lang_buckets[lang_name]['issues'].get(issue, 0) + 1
                        course_lang_buckets[lang_name]['negative_comments'].append({
                            'comment': comment_text.strip(),
                            'date': c_date,
                            'language': c_lang,
                            'sentiment': sent,
                            'analysis': comment_data.get('analysis', {})
                        })
                    else:
                        course_lang_buckets[lang_name]['neu'] += 1
                        course_lang_buckets[lang_name]['neutral_comments'].append({
                            'comment': comment_text.strip(),
                            'date': c_date,
                            'language': c_lang,
                            'sentiment': sent,
                            'analysis': comment_data.get('analysis', {})
                        })

        for lang_name, b in course_lang_buckets.items():
            b_total = b['pos'] + b['neg'] + b['neu']
            if b_total == 0: continue
            sorted_issues = sorted(b['issues'].items(), key=lambda x: x[1], reverse=True)[:5]
            
            unique_neg_comments = []
            seen_neg_comments = set()
            for nc in sorted(b['negative_comments'], key=lambda x: x.get('date', '1970-01-01'), reverse=True):
                if nc['comment'] not in seen_neg_comments:
                    seen_neg_comments.add(nc['comment'])
                    unique_neg_comments.append(nc)

            unique_pos_comments = []
            seen_pos_comments = set()
            for pc in sorted(b['positive_comments'], key=lambda x: x.get('date', '1970-01-01'), reverse=True):
                if pc['comment'] not in seen_pos_comments:
                    seen_pos_comments.add(pc['comment'])
                    unique_pos_comments.append(pc)

            unique_neu_comments = []
            seen_neu_comments = set()
            for nc in sorted(b['neutral_comments'], key=lambda x: x.get('date', '1970-01-01'), reverse=True):
                if nc['comment'] not in seen_neu_comments:
                    seen_neu_comments.add(nc['comment'])
                    unique_neu_comments.append(nc)

            flat_buckets.append({
                'course': course_name,
                'course_id': course_id,
                'lang': lang_name,
                'pos': b['pos'],
                'neg': b['neg'],
                'neu': b['neu'],
                'issues': [{'name': ik, 'count': iv} for ik, iv in sorted_issues],
                'negative_comments': unique_neg_comments,
                'positive_comments': unique_pos_comments,
                'neutral_comments': unique_neu_comments
            })

        # Determine course-level languages (top 3)
        sorted_langs = sorted(c_langs_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_langs = [k.upper() if k != 'unknown' else 'UNK' for k, v in sorted_langs]
        
        # update global language counts
        for k, v in c_langs_counts.items():
            lang_name = LANG_MAP.get(k, k.title())
            language_counts[lang_name] = language_counts.get(lang_name, 0) + v

        # Formatting values for the UI template
        total_c = max(1, c_pos + c_neg + c_neu)
        
        # Avoid showing empty courses
        if total_c == 1 and c_pos == 0 and c_neg == 0:
            continue
            
        sorted_c_issues = sorted(c_issues.items(), key=lambda x: x[1], reverse=True)[:5]
        course_top_issues = [{'name': k, 'count': v} for k, v in sorted_c_issues]
        
        course_stats.append({
            'name': course_name,
            'id': course_id,
            'total_comments': f"{total_c:,}",
            'languages': top_langs,
            'positive_count': c_pos,
            'negative_count': c_neg,
            'neutral_count': c_neu,
            'positive_width': max(1, int((c_pos / total_c) * 120)),
            'negative_width': max(1, int((c_neg / total_c) * 120)),
            'neutral_width': max(1, int((c_neu / total_c) * 120)),
            'alert_class': 'high' if (c_neg / total_c) > 0.3 else 'med' if (c_neg / total_c) > 0.15 else 'low',
            'alert_label': 'High' if (c_neg / total_c) > 0.3 else 'Medium' if (c_neg / total_c) > 0.15 else 'Low',
            'issues': course_top_issues,
            'sort_key': total_c
        })

    # Sorting
    course_stats.sort(key=lambda x: x['sort_key'], reverse=True)
    
    # Process overall issues
    top_issues_list = []
    if issue_counts:
        total_issues = sum(issue_counts.values()) or 1
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:11]
        for k, v in sorted_issues:
            pct = (v / total_issues) * 100
            top_issues_list.append({'name': k, 'count': f"{v:,}", 'percent': f"{pct:.1f}"})

    # Language summary with per-language sentiment
    lang_summary = []
    if language_counts:
        total_langs = sum(language_counts.values()) or 1
        
        CUSTOM_LANG_ORDER = {
            'English': 1, 'Hindi': 2, 'Telugu': 3, 'Marathi': 4, 'Tamil': 5,
            'Urdu': 6, 'Odia': 7, 'Unknown': 8, 'Error': 9
        }
        
        sorted_languages = sorted(
            language_counts.items(), 
            key=lambda x: (CUSTOM_LANG_ORDER.get(x[0], 99), -x[1])
        )
        
        for k, v in sorted_languages:
            pct = (v / total_langs) * 100
            if pct > 20: skew_class, skew_label = 'dom', 'Dominant' if pct > 40 else 'High'
            elif pct > 5: skew_class, skew_label = 'bal', 'Moderate'
            else: skew_class, skew_label = 'min', 'Low'
            
            ls = language_sentiment.get(k, {'positive': 0, 'negative': 0, 'neutral': 0})
            lt = max(1, ls['positive'] + ls['negative'] + ls['neutral'])
            
            lang_summary.append({
                'name': k,
                'count': f"{v:,}",
                'percent': f"{pct:.1f}",
                'skew_class': skew_class,
                'skew_label': skew_label,
                'positive_count': f"{ls['positive']:,}",
                'negative_count': f"{ls['negative']:,}",
                'neutral_count': f"{ls['neutral']:,}",
                'positive_pct': f"{(ls['positive'] / lt) * 100:.0f}",
                'negative_pct': f"{(ls['negative'] / lt) * 100:.0f}",
                'neutral_pct': f"{(ls['neutral'] / lt) * 100:.0f}",
            })

    total = max(1, total_valid_comments)
    metrics = {
        'total_comments': f"{total_valid_comments:,}",
        'total_course_count': f"{len(course_stats)}",
        'overall_positive_count': f"{sentiment_counts['positive']:,}",
        'overall_negative_count': f"{sentiment_counts['negative']:,}",
        'overall_neutral_count': f"{sentiment_counts['neutral']:,}",
        'overall_positive_percent': f"{(sentiment_counts['positive'] / total) * 100:.1f}",
        'overall_negative_percent': f"{(sentiment_counts['negative'] / total) * 100:.1f}",
        'overall_neutral_percent': f"{(sentiment_counts['neutral'] / total) * 100:.1f}",
        'courses': course_stats,
        'course_list': [{'name': c['name'], 'count': c['total_comments']} for c in course_stats],
        'languages': lang_summary,
        'language_chips': [lang['name'] for lang in lang_summary],
        'top_issues': top_issues_list,
        'flat_buckets': flat_buckets,
    }
    
    return metrics

def build_dashboard():
    print("Analyzing full comments data...")
    metrics_full = analyze_dataset(SENTIMENT_FULL_DIR, SARVAM_FULL_DIR, GEMINI_DIR)
    
    # Bundle for Jinja
    context = {}
    context.update(metrics_full)  # Flatten so Jinja {{ total_comments }} works
    context['full_json'] = json.dumps(metrics_full)
    
    print("Rendering HTML template...")
    env = FileSystemLoader(TEMPLATE_DIR)
    jinja_env = Environment(loader=env)
    template = jinja_env.get_template(TEMPLATE_FILE)
    
    output_html = template.render(context)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_html)
    print(f"Dashboard successfully generated at: {OUTPUT_FILE}")

if __name__ == '__main__':
    build_dashboard()
