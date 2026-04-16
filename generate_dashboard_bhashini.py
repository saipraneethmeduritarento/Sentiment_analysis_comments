import os
import json
import csv
import glob
import re
import unicodedata
from jinja2 import Environment, FileSystemLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SENTIMENT_FULL_DIR = os.path.join(BASE_DIR, 'output', 'full_comments')
BHASHINI_FULL_DIR = os.path.join(BASE_DIR, 'output', 'language_detection', 'bhashini', 'full_comments')
GEMINI_NEG_DIR = os.path.join(BASE_DIR, 'gemini_analysis', 'negative')
GEMINI_POS_DIR = os.path.join(BASE_DIR, 'gemini_analysis', 'positive')
TEMPLATE_DIR = os.path.join(BASE_DIR, 'ui', 'template')
TEMPLATE_FILE = 'dashboard_template_v1.html'
OUTPUT_FILE = os.path.join(BASE_DIR, 'ui', 'output', 'sentiment_analysis_report.html')

LANG_MAP = {
    'hi': 'Hindi', 'en': 'English', 'ta': 'Tamil', 'te': 'Telugu',
    'mr': 'Marathi', 'bn': 'Bengali', 'gu': 'Gujarati', 'kn': 'Kannada',
    'ml': 'Malayalam', 'or': 'Odia', 'pa': 'Punjabi', 'as': 'Assamese',
    'ur': 'Urdu', 'hinglish': 'Hinglish', 'unknown': 'Hinglish'
}


def norm_comment(text):
    """Normalize comment text for cross-CSV matching: unicode, smart quotes, backslashes, whitespace."""
    text = unicodedata.normalize('NFKC', str(text))
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\\', '')
    return ' '.join(text.lower().split())


def load_category_summaries(gemini_dir: str) -> dict:
    """Load pre-generated per-course category summary JSONs.
    Returns: {course_id: summary_dict}
    """
    summaries = {}
    if not os.path.exists(gemini_dir):
        return summaries
    for summary_file in glob.glob(os.path.join(gemini_dir, '*_category_summary.json')):
        course_id = os.path.basename(summary_file).replace('_category_summary.json', '')
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                summaries[course_id] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load summary file {summary_file}: {e}")
    return summaries


def load_gemini_lookup(gemini_dir, category_key):
    """Load a Gemini analysis JSON directory into a lookup: course_id -> {norm_comment -> analysis_dict}"""
    lookup = {}
    if not os.path.exists(gemini_dir):
        return lookup
    for gfile in glob.glob(os.path.join(gemini_dir, '*.json')):
        if '_category_summary' in os.path.basename(gfile):
            continue
        course_id = os.path.splitext(os.path.basename(gfile))[0]
        try:
            with open(gfile, 'r', encoding='utf-8') as f:
                gdata = json.load(f)
            lookup[course_id] = {}
            for item in gdata:
                raw_c = str(item.get('comment', ''))
                norm_c = norm_comment(raw_c)
                ga = item.get('Gemini Analysis', {})
                if isinstance(ga, dict) and category_key in ga:
                    lookup[course_id][norm_c] = ga
        except Exception as e:
            print(f"Warning: Could not load gemini file {gfile}: {e}")
    return lookup


def analyze_dataset():
    flat_buckets = []
    course_stats = []
    issue_counts = {}
    strength_counts = {}
    sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
    language_counts = {}
    language_sentiment = {}
    total_valid_comments = 0

    # Load Gemini lookups (negative uses "Issue Category", positive uses "Strength Category")
    gemini_neg_lookup = load_gemini_lookup(GEMINI_NEG_DIR, 'Issue Category')
    gemini_pos_lookup = load_gemini_lookup(GEMINI_POS_DIR, 'Strength Category')

    # Load pre-generated per-course category summaries
    neg_category_summaries = load_category_summaries(GEMINI_NEG_DIR)
    pos_category_summaries = load_category_summaries(GEMINI_POS_DIR)

    comment_sentiment_map = {}

    csv_files = glob.glob(os.path.join(SENTIMENT_FULL_DIR, '*.csv'))
    
    for csv_file in csv_files:
        course_id = os.path.splitext(os.path.basename(csv_file))[0]
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        course_name = ""
        c_pos, c_neg, c_neu = 0, 0, 0
        c_issues = {}
        c_pos_strengths = {}
        
        for item in rows:
            if not course_name:
                course_name = item.get('content_name', course_id)
            
            comment_text = str(item.get('comment', ''))
            total_valid_comments += 1
            sentiment = str(item.get('predicted sentiment', 'neutral')).strip().lower()
            
            norm_key = norm_comment(comment_text)

            # Pick appropriate Gemini analysis based on sentiment
            if sentiment == 'negative':
                gemini_analysis = gemini_neg_lookup.get(course_id, {}).get(norm_key, {})
                issue = gemini_analysis.get('Issue Category', 'Other Issue') if isinstance(gemini_analysis, dict) else 'Other Issue'
                strength = ''
            elif sentiment == 'positive':
                gemini_analysis = gemini_pos_lookup.get(course_id, {}).get(norm_key, {})
                strength = gemini_analysis.get('Strength Category', '') if isinstance(gemini_analysis, dict) else ''
                issue = 'Other Issue'
            else:
                gemini_analysis = {}
                issue = 'Other Issue'
                strength = ''

            if sentiment == 'positive':
                c_pos += 1
                sentiment_counts['positive'] += 1
                if strength and strength != '':
                    strength_counts[strength] = strength_counts.get(strength, 0) + 1
                    c_pos_strengths[strength] = c_pos_strengths.get(strength, 0) + 1
            elif sentiment == 'negative':
                c_neg += 1
                sentiment_counts['negative'] += 1
                if issue and issue != 'Other Issue':
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1
                    c_issues[issue] = c_issues.get(issue, 0) + 1
            else:
                c_neu += 1
                sentiment_counts['neutral'] += 1

            if course_id not in comment_sentiment_map:
                comment_sentiment_map[course_id] = {}
            comment_sentiment_map[course_id][norm_key] = {
                'sentiment': sentiment,
                'issue': issue,
                'strength': strength,
                'analysis': gemini_analysis,
                'date': item.get('comment_date', '')
            }
                
        # Language detection join
        c_langs_counts = {}
        course_lang_buckets = {}
        course_lang_sentiment = {}  # lang_name -> {positive, negative, neutral}
        csv_path = os.path.join(BHASHINI_FULL_DIR, f"{course_id}.csv")
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    comment_text = str(row.get('comment', ''))
                    lang_code = str(row.get('predicted_language', 'hinglish')).strip().lower()
                    lang_name = LANG_MAP.get(lang_code, lang_code.title())

                    norm_csv_comment = norm_comment(comment_text)
                    comment_data = comment_sentiment_map.get(course_id, {}).get(norm_csv_comment)
                    if comment_data is None:
                        # Comment text differs between pipelines — skip to avoid sentiment count drift
                        continue

                    c_langs_counts[lang_code] = c_langs_counts.get(lang_code, 0) + 1
                    sent = comment_data['sentiment']

                    if lang_name not in language_sentiment:
                        language_sentiment[lang_name] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    language_sentiment[lang_name][sent] += 1

                    if lang_name not in course_lang_sentiment:
                        course_lang_sentiment[lang_name] = {'positive': 0, 'negative': 0, 'neutral': 0}
                    course_lang_sentiment[lang_name][sent] += 1

                    c_date = comment_data.get('date', row.get('comment_date', ''))
                    c_lang = lang_name
                    if lang_name not in course_lang_buckets:
                        course_lang_buckets[lang_name] = {
                            'pos': 0, 'neg': 0, 'neu': 0, 'issues': {}, 'strengths': {},
                            'negative_comments': [], 'positive_comments': [], 'neutral_comments': []
                        }

                    if sent == 'positive':
                        course_lang_buckets[lang_name]['pos'] += 1
                        strength_cat = comment_data.get('strength', '')
                        if strength_cat:
                            course_lang_buckets[lang_name]['strengths'][strength_cat] = \
                                course_lang_buckets[lang_name]['strengths'].get(strength_cat, 0) + 1
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
                        if issue and issue != 'Other Issue':
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
            if b_total == 0:
                continue
            sorted_issues = sorted(b['issues'].items(), key=lambda x: x[1], reverse=True)
            sorted_strengths = sorted(b['strengths'].items(), key=lambda x: x[1], reverse=True)

            def dedup_comments(comments):
                out, seen = [], set()
                for nc in sorted(comments, key=lambda x: x.get('date', ''), reverse=True):
                    if nc['comment'] not in seen:
                        seen.add(nc['comment'])
                        out.append(nc)
                return out

            flat_buckets.append({
                'course': course_name,
                'course_id': course_id,
                'lang': lang_name,
                'pos': b['pos'],
                'neg': b['neg'],
                'neu': b['neu'],
                'issues': [{'name': ik, 'count': iv} for ik, iv in sorted_issues],
                'strengths': [{'name': sk, 'count': sv} for sk, sv in sorted_strengths],
                'negative_comments': dedup_comments(b['negative_comments']),
                'positive_comments': dedup_comments(b['positive_comments']),
                'neutral_comments': dedup_comments(b['neutral_comments'])
            })

        # Determine course-level languages (top 3)
        sorted_langs = sorted(c_langs_counts.items(), key=lambda x: x[1], reverse=True)[:3]
        top_langs = [k.upper() for k, v in sorted_langs]
        
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
        sorted_c_strengths = sorted(c_pos_strengths.items(), key=lambda x: x[1], reverse=True)[:5]

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
            'course_lang_sentiment': {
                lang: {'positive_count': v['positive'], 'negative_count': v['negative'], 'neutral_count': v['neutral']}
                for lang, v in course_lang_sentiment.items()
            },
            'issues': [{'name': k, 'count': v} for k, v in sorted_c_issues],
            'strengths': [{'name': k, 'count': v} for k, v in sorted_c_strengths],
            'neg_category_summary': neg_category_summaries.get(course_id, {}),
            'pos_category_summary': pos_category_summaries.get(course_id, {}),
            'sort_key': total_c
        })

    # Sorting
    course_stats.sort(key=lambda x: x['sort_key'], reverse=True)
    
    # Process overall issues
    top_issues_list = []
    if issue_counts:
        total_issues = sum(issue_counts.values()) or 1
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for k, v in sorted_issues:
            pct = (v / total_issues) * 100
            top_issues_list.append({'name': k, 'count': f"{v:,}", 'percent': f"{pct:.1f}"})

    # Process overall strengths
    top_strengths_list = []
    if strength_counts:
        total_strengths = sum(strength_counts.values()) or 1
        sorted_strengths = sorted(strength_counts.items(), key=lambda x: x[1], reverse=True)[:15]
        for k, v in sorted_strengths:
            pct = (v / total_strengths) * 100
            top_strengths_list.append({'name': k, 'count': f"{v:,}", 'percent': f"{pct:.1f}"})

    # Language summary with per-language sentiment
    lang_summary = []
    if language_counts:
        total_langs = sum(language_counts.values()) or 1
        
        CUSTOM_LANG_ORDER = {
            'English': 1, 'Hindi': 2, 'Telugu': 3, 'Marathi': 4, 'Tamil': 5,
            'Urdu': 6, 'Odia': 7, 'Hinglish': 8, 'Error': 9
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
        'top_strengths': top_strengths_list,
        'flat_buckets': flat_buckets,
        'neg_category_summaries': neg_category_summaries,
        'pos_category_summaries': pos_category_summaries,
    }
    
    return metrics

def build_dashboard():
    print("Analyzing full comments data...")
    metrics = analyze_dataset()

    context = {}
    context.update(metrics)
    context['full_json'] = json.dumps(metrics)
    
    print("Rendering HTML template...")
    env = FileSystemLoader(TEMPLATE_DIR)
    jinja_env = Environment(loader=env)
    jinja_env.filters['regex_search'] = lambda s, pattern: bool(re.search(pattern, s, re.IGNORECASE))
    template = jinja_env.get_template(TEMPLATE_FILE)
    
    output_html = template.render(context)
    
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(output_html)
    print(f"Dashboard successfully generated at: {OUTPUT_FILE}")

if __name__ == '__main__':
    build_dashboard()
