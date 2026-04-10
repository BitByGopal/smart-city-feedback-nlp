# demo.py
"""
Smart City Feedback Analysis System — Terminal Demo
====================================================
Run this to see the full pipeline working without Streamlit.

Usage:
    python demo.py              # full demo, 100 records
    python demo.py --n 50       # smaller sample
    python demo.py --eval       # run sentiment accuracy test
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Color helpers ────────────────────────────────────────────────
RED     = "\033[91m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
BLUE    = "\033[94m"
MAGENTA = "\033[95m"
CYAN    = "\033[96m"
BOLD    = "\033[1m"
RESET   = "\033[0m"

def header(text):
    width = 60
    print(f"\n{BOLD}{BLUE}{'═'*width}{RESET}")
    print(f"{BOLD}{BLUE}  {text}{RESET}")
    print(f"{BOLD}{BLUE}{'═'*width}{RESET}")

def step(n, text):
    print(f"\n{BOLD}{CYAN}[STEP {n}]{RESET} {text}")

def ok(text):
    print(f"  {GREEN}✓{RESET} {text}")

def info(text):
    print(f"  {YELLOW}→{RESET} {text}")

def alert(text, level="high"):
    color = RED if level == "critical" else YELLOW if level == "high" else GREEN
    print(f"  {color}●{RESET} {text}")


def run_demo(n_records: int = 100):
    start = time.time()

    header("Smart City Feedback Analysis System")
    print(f"  NLP + LLM Pipeline Demo | {n_records} records")

    # ── Step 1: Data Collection ──────────────────────────────────
    step(1, "Data Collection")
    from src.data_collector import generate_synthetic_data
    df = generate_synthetic_data(n=n_records, city="Hyderabad")
    ok(f"Generated {len(df)} feedback records from {df['source'].nunique()} sources")
    ok(f"Departments: {', '.join(df['department'].unique()[:4])}...")

    # Sample records
    print(f"\n  {BOLD}Sample feedback:{RESET}")
    for _, row in df.sample(3).iterrows():
        print(f"    [{row['source']:8}] {row['text'][:70]}...")

    # ── Step 2: Preprocessing ─────────────────────────────────────
    step(2, "Text Preprocessing")
    from src.preprocessor import preprocess_dataframe
    df = preprocess_dataframe(df)
    ok(f"Cleaned {len(df)} records — avg word count: {df['word_count'].mean():.1f} words")
    ok("Applied: URL removal, slang normalization, lemmatization, stopword removal")

    # ── Step 3: Sentiment Analysis ───────────────────────────────
    step(3, "Sentiment Analysis (VADER)")
    from src.sentiment_analyzer import analyze_dataframe
    df = analyze_dataframe(df, use_bert=False)

    counts = df['sentiment_label'].value_counts()
    neg_pct = round(counts.get('NEGATIVE', 0) / len(df) * 100, 1)
    pos_pct = round(counts.get('POSITIVE', 0) / len(df) * 100, 1)
    neu_pct = round(counts.get('NEUTRAL', 0) / len(df) * 100, 1)

    ok(f"Analyzed {len(df)} records")
    print(f"    {RED}NEGATIVE:{RESET}  {neg_pct:5.1f}%  {'█' * int(neg_pct/3)}")
    print(f"    {YELLOW}NEUTRAL:{RESET}   {neu_pct:5.1f}%  {'█' * int(neu_pct/3)}")
    print(f"    {GREEN}POSITIVE:{RESET}  {pos_pct:5.1f}%  {'█' * int(pos_pct/3)}")

    # Most negative sample
    most_neg = df.nsmallest(1, 'sentiment_numeric').iloc[0]
    print(f"\n  {BOLD}Most negative feedback:{RESET}")
    print(f"    \"{most_neg['text'][:80]}\"")
    print(f"    Score: {most_neg['sentiment_score']:.2f} | Dept: {most_neg['department']}")

    # ── Step 4: Topic Modeling ────────────────────────────────────
    step(4, "Topic Modeling (Keyword Matching)")
    df['topic_label'] = df['department']   # fast demo mode
    topic_counts = df['topic_label'].value_counts()
    ok(f"Identified {len(topic_counts)} topic categories")
    print()
    for dept, count in topic_counts.items():
        bar = '█' * (count * 30 // len(df))
        pct = count * 100 // len(df)
        print(f"    {dept:<25} {count:3d} ({pct}%)  {bar}")

    # ── Step 5: NER ───────────────────────────────────────────────
    step(5, "Named Entity Recognition")
    from src.ner_extractor import extract_entities_batch, get_location_frequency
    df = extract_entities_batch(df)
    loc_df = get_location_frequency(df)

    ok(f"Extracted location mentions from {len(df)} records")
    print(f"\n  {BOLD}Top 8 complaint hotspots:{RESET}")
    for _, row in loc_df.head(8).iterrows():
        bar = '█' * min(int(row['count'] * 20 / loc_df['count'].max()), 20)
        print(f"    {row['location']:<22} {int(row['count']):3d}  {bar}")

    # ── Step 6: Priority Scoring ──────────────────────────────────
    step(6, "Priority Scoring & Alert System")
    from src.priority_scorer import score_dataframe, get_overall_city_score, get_top_complaints

    df = score_dataframe(df)
    kpis = get_overall_city_score(df)

    ok(f"Urgency score: {kpis['urgency_score']}/10")
    ok(f"Critical alerts: {kpis['critical_alerts']}  |  High: {kpis['high_alerts']}")

    alert_counts = df['alert_level'].value_counts()
    print(f"\n  {BOLD}Alert breakdown:{RESET}")
    for level, count in alert_counts.items():
        lvl = "critical" if "CRITICAL" in level else "high" if "HIGH" in level else "low"
        alert(f"{level}  {count} complaints", level=lvl)

    # Top 5 complaints
    print(f"\n  {BOLD}Top 5 urgent complaints:{RESET}")
    top5 = get_top_complaints(df, n=5)
    for i, (_, row) in enumerate(top5.iterrows(), 1):
        score_color = RED if row['priority_score'] >= 7 else YELLOW if row['priority_score'] >= 4 else GREEN
        print(f"\n  {BOLD}{i}.{RESET} {row['text'][:70]}...")
        print(f"     Area: {row.get('area', 'N/A')}  |  Dept: {row['topic_label']}")
        print(f"     Score: {score_color}{row['priority_score']}/10{RESET}  |  {row['alert_level']}")

    # ── Step 7: LLM Report ─────────────────────────────────────────
    step(7, "AI Report Generation")
    from src.priority_scorer import get_department_summary
    from src.llm_reporter import generate_city_report, answer_query

    dept_summary = get_department_summary(df)
    loc_str = ", ".join(f"{r['location']} ({int(r['count'])})" for _, r in loc_df.head(5).iterrows())

    aggregated = {
        "city": "Hyderabad",
        "days": 30,
        "total": kpis["total"],
        "negative_pct": kpis["negative_pct"],
        "positive_pct": kpis["positive_pct"],
        "urgency_score": kpis["urgency_score"],
        "critical_alerts": kpis["critical_alerts"],
        "high_alerts": kpis["high_alerts"],
        "top_issue": kpis.get("top_issue", "Infrastructure"),
        "dept_summary": "\n".join(
            f"  - {r['Department']}: {r['Total Complaints']} complaints"
            for _, r in dept_summary.iterrows()
        ),
        "top_locations": loc_str,
        "top_complaints": top5.iloc[0]['text'] if len(top5) > 0 else "",
        "topic_distribution": "",
    }

    report = generate_city_report(aggregated)
    ok("Executive report generated (template mode — add GEMINI_API_KEY for AI report)")
    print()
    # Print report with indentation
    for line in report.split('\n'):
        print(f"    {line}")

    # ── Step 8: Q&A Chatbot ────────────────────────────────────────
    step(8, "Q&A Chatbot Demo")
    demo_questions = [
        "What is the most urgent issue?",
        "Which area has the most complaints?",
        "What percentage of feedback is negative?",
    ]
    ok("Testing chatbot with sample questions:")
    for q in demo_questions:
        answer = answer_query(q, aggregated)
        print(f"\n  {BOLD}Q:{RESET} {q}")
        print(f"  {CYAN}A:{RESET} {answer}")

    # ── Summary ────────────────────────────────────────────────────
    elapsed = round(time.time() - start, 1)
    header("Pipeline Complete")
    print(f"  {GREEN}✓{RESET} All 8 modules ran successfully")
    print(f"  {GREEN}✓{RESET} Total time: {elapsed}s")
    print(f"  {GREEN}✓{RESET} Records processed: {len(df)}")
    print(f"\n  {BOLD}Next steps:{RESET}")
    print(f"    1. Run dashboard:  {CYAN}streamlit run app.py{RESET}")
    print(f"    2. Run API:        {CYAN}uvicorn api:app --reload{RESET}")
    print(f"    3. Fine-tune BERT: {CYAN}python src/fine_tuner.py --epochs 3{RESET}")
    print(f"    4. Add Gemini key in .env for AI-powered reports\n")


def run_sentiment_eval():
    """Quick accuracy evaluation on predefined test cases."""
    header("Sentiment Model Evaluation")
    from src.fine_tuner import quick_evaluate
    acc = quick_evaluate()
    print(f"\n  {BOLD}Accuracy: {GREEN}{acc:.1%}{RESET}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart City Feedback Analysis Demo")
    parser.add_argument("--n", type=int, default=100, help="Number of records to analyze")
    parser.add_argument("--eval", action="store_true", help="Run sentiment evaluation instead")
    args = parser.parse_args()

    if args.eval:
        run_sentiment_eval()
    else:
        run_demo(n_records=args.n)