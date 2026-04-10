# src/priority_scorer.py
"""
Priority Scoring & Alert System
Calculates urgency scores for complaints and triggers department alerts.
"""

import pandas as pd
import numpy as np
from typing import Dict, List

# Urgency keywords with weights
URGENCY_KEYWORDS = {
    "critical": 3.0,
    "emergency": 3.0,
    "dangerous": 2.5,
    "accident": 2.5,
    "injured": 2.5,
    "death": 3.0,
    "fire": 3.0,
    "flood": 2.5,
    "broken": 1.5,
    "not working": 1.5,
    "no water": 2.0,
    "no electricity": 2.0,
    "power cut": 1.8,
    "months": 2.0,
    "weeks": 1.5,
    "days": 1.0,
    "please": 0.5,
    "urgently": 2.0,
    "immediately": 2.0,
    "worst": 1.5,
    "terrible": 1.5,
    "horrible": 1.5,
    "unacceptable": 1.5,
    "disgusting": 1.5,
    "overflowing": 2.0,
    "stagnant": 1.5,
    "disease": 2.5,
    "infection": 2.0,
    "contaminated": 2.0,
}

DEPARTMENT_PRIORITY = {
    "Roads & Infrastructure": 0.8,
    "Water Supply": 0.9,
    "Sanitation": 0.7,
    "Electricity": 0.8,
    "Public Transport": 0.6,
    "Parks & Recreation": 0.4,
    "Public Safety": 1.0,
    "General Services": 0.5,
    "Miscellaneous": 0.3,
}


def calculate_text_urgency(text: str) -> float:
    """Calculate urgency score from text keywords (0-5 scale)."""
    text_lower = text.lower()
    score = 0.0
    for keyword, weight in URGENCY_KEYWORDS.items():
        if keyword in text_lower:
            score += weight
    return min(score, 5.0)


def calculate_record_priority(row: pd.Series) -> float:
    """
    Calculate overall priority score for a single feedback record (0-10).
    
    Factors:
    1. Text urgency keywords (0-5)
    2. Sentiment negativity (0-3)
    3. Social media engagement (0-1)
    4. Department base priority (0-1)
    """
    # Factor 1: Text urgency
    text_urgency = calculate_text_urgency(str(row.get("text", "")))

    # Factor 2: Sentiment (negative = higher priority)
    sentiment_label = str(row.get("sentiment_label", "NEUTRAL"))
    sentiment_score = float(row.get("sentiment_score", 0.5))
    if sentiment_label == "NEGATIVE":
        sentiment_factor = 3.0 * sentiment_score
    elif sentiment_label == "NEUTRAL":
        sentiment_factor = 1.0
    else:
        sentiment_factor = 0.0

    # Factor 3: Engagement (normalize to 0-1)
    likes = int(row.get("likes", 0))
    retweets = int(row.get("retweets", 0))
    engagement = min((likes + retweets * 2) / 200, 1.0)

    # Factor 4: Department base priority
    dept = str(row.get("topic_label", row.get("department", "General Services")))
    dept_weight = DEPARTMENT_PRIORITY.get(dept, 0.5)

    # Weighted sum normalized to 0-10
    raw_score = (text_urgency * 0.4 + sentiment_factor * 0.35 + engagement * 0.1 + dept_weight * 0.15) * 10 / 5
    return round(min(raw_score, 10.0), 2)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add priority_score and alert_level columns to DataFrame."""
    print("📊 Calculating priority scores...")
    df["priority_score"] = df.apply(calculate_record_priority, axis=1)

    # Alert level based on score
    def get_alert_level(score: float) -> str:
        if score >= 7.5:
            return "🔴 CRITICAL"
        elif score >= 5.0:
            return "🟠 HIGH"
        elif score >= 2.5:
            return "🟡 MEDIUM"
        else:
            return "🟢 LOW"

    df["alert_level"] = df["priority_score"].apply(get_alert_level)
    print(f"✅ Scoring complete.\n{df['alert_level'].value_counts()}")
    return df


def get_top_complaints(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Get top N most urgent complaints."""
    return (
        df.nlargest(n, "priority_score")[
            ["text", "area", "topic_label", "sentiment_label", "priority_score", "alert_level"]
        ]
        .reset_index(drop=True)
    )


def get_department_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize complaints by department with aggregate stats.
    Used for the dashboard overview cards.
    """
    dept_col = "topic_label" if "topic_label" in df.columns else "department"
    sentiment_col = "sentiment_label" if "sentiment_label" in df.columns else None

    agg_dict = {
        "text": "count",
        "priority_score": "mean",
    }
    if sentiment_col:
        pass  # handled separately

    summary = df.groupby(dept_col).agg(
        total_complaints=("text", "count"),
        avg_priority=("priority_score", "mean") if "priority_score" in df.columns else ("text", "count"),
    ).reset_index()

    summary.columns = ["Department", "Total Complaints", "Avg Priority"]
    summary["Avg Priority"] = summary["Avg Priority"].round(2)

    # Add negative sentiment %
    if sentiment_col:
        neg_counts = df[df[sentiment_col] == "NEGATIVE"].groupby(dept_col).size().reset_index(name="neg_count")
        summary = summary.merge(neg_counts, left_on="Department", right_on=dept_col, how="left").drop(
            columns=[dept_col], errors="ignore"
        )
        summary["neg_count"] = summary["neg_count"].fillna(0)
        summary["Negative %"] = (summary["neg_count"] / summary["Total Complaints"] * 100).round(1)
    else:
        summary["Negative %"] = 0.0

    summary = summary.sort_values("Total Complaints", ascending=False)
    return summary


def get_overall_city_score(df: pd.DataFrame) -> Dict:
    """
    Compute overall city health metrics.
    Returns dict of KPI values for the dashboard.
    """
    total = len(df)
    if total == 0:
        return {"total": 0, "urgency": 0, "negative_pct": 0, "top_issue": "N/A"}

    neg_pct = round((df.get("sentiment_label", pd.Series()) == "NEGATIVE").sum() / total * 100, 1)
    pos_pct = round((df.get("sentiment_label", pd.Series()) == "POSITIVE").sum() / total * 100, 1)

    avg_urgency = round(df["priority_score"].mean(), 1) if "priority_score" in df.columns else 5.0
    urgency_10 = round(avg_urgency, 1)

    dept_col = "topic_label" if "topic_label" in df.columns else "department"
    top_issue = df[dept_col].mode()[0] if dept_col in df.columns and len(df) > 0 else "N/A"

    critical_count = (df.get("alert_level", pd.Series()) == "🔴 CRITICAL").sum()
    high_count = (df.get("alert_level", pd.Series()) == "🟠 HIGH").sum()

    return {
        "total": total,
        "urgency_score": urgency_10,
        "negative_pct": neg_pct,
        "positive_pct": pos_pct,
        "top_issue": top_issue,
        "critical_alerts": int(critical_count),
        "high_alerts": int(high_count),
    }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_collector import generate_synthetic_data
    from src.preprocessor import preprocess_dataframe
    from src.sentiment_analyzer import analyze_dataframe

    df = generate_synthetic_data(n=100)
    df = preprocess_dataframe(df)
    df = analyze_dataframe(df, use_bert=False)
    df["topic_label"] = df["department"]  # mock topic labels
    df = score_dataframe(df)

    print("\nTop 5 urgent complaints:")
    print(get_top_complaints(df, n=5).to_string())
    print("\nCity KPIs:")
    print(get_overall_city_score(df))