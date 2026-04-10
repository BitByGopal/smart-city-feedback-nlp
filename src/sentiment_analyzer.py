# src/sentiment_analyzer.py
"""
Sentiment Analysis Module
Primary:  HuggingFace BERT (cardiffnlp/twitter-roberta-base-sentiment-latest)
Fallback: VADER (rule-based, no GPU needed)
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict

# ─────────────────────────────────────────────
# BERT-BASED SENTIMENT  (primary)
# ─────────────────────────────────────────────

_sentiment_pipeline = None

def _load_bert_model():
    global _sentiment_pipeline
    if _sentiment_pipeline is None:
        try:
            from transformers import pipeline
            print("⏳ Loading BERT sentiment model (first time takes ~30s)...")
            _sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                return_all_scores=True,
                device=-1,  # CPU; change to 0 for GPU
            )
            print("✅ BERT sentiment model loaded.")
        except Exception as e:
            print(f"⚠️  BERT load failed ({e}). Falling back to VADER.")
            _sentiment_pipeline = None
    return _sentiment_pipeline


def _bert_predict(texts: List[str]) -> List[Dict]:
    model = _load_bert_model()
    if model is None:
        return []

    results = []
    # Process in batches of 32
    batch_size = 32
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            batch_results = model(batch, truncation=True, max_length=128)
            for scores in batch_results:
                label_map = {"negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE"}
                best = max(scores, key=lambda x: x["score"])
                results.append({
                    "label": label_map.get(best["label"].lower(), best["label"].upper()),
                    "score": round(best["score"], 4),
                    "scores": {
                        label_map.get(s["label"].lower(), s["label"]): round(s["score"], 4)
                        for s in scores
                    },
                })
        except Exception as e:
            print(f"Batch error: {e}")
            # Add neutral fallback for this batch
            for _ in batch:
                results.append({"label": "NEUTRAL", "score": 0.5, "scores": {}})

    return results


# ─────────────────────────────────────────────
# VADER FALLBACK  (no model download needed)
# ─────────────────────────────────────────────

def _vader_predict(texts: List[str]) -> List[Dict]:
    try:
        import nltk
        for attempt in ["vader_lexicon"]:
            try:
                nltk.data.find(f"sentiment/{attempt}.zip")
            except LookupError:
                try:
                    nltk.download(attempt, quiet=True)
                except Exception:
                    pass

        from nltk.sentiment.vader import SentimentIntensityAnalyzer
        sia = SentimentIntensityAnalyzer()
        results = []
        for text in texts:
            scores = sia.polarity_scores(text)
            compound = scores["compound"]

            if compound >= 0.05:
                label = "POSITIVE"
                score = round((compound + 1) / 2, 4)
            elif compound <= -0.05:
                label = "NEGATIVE"
                score = round((1 - compound) / 2, 4)
            else:
                label = "NEUTRAL"
                score = round(1 - abs(compound), 4)

            results.append({
                "label": label,
                "score": score,
                "scores": {
                    "POSITIVE": round(scores["pos"], 4),
                    "NEUTRAL": round(scores["neu"], 4),
                    "NEGATIVE": round(scores["neg"], 4),
                },
            })
        return results

    except Exception as e:
        print(f"VADER unavailable ({e}). Using keyword fallback.")
        return _keyword_sentiment(texts)


# Zero-dependency keyword sentiment — works 100% offline
_POS_WORDS = {"good","great","excellent","amazing","beautiful","wonderful",
              "improved","clean","fast","safe","happy","restored","fixed",
              "appreciate","thank","love","best","perfect","efficient","helpful"}
_NEG_WORDS = {"bad","terrible","horrible","worst","broken","damage","damaged",
              "pothole","overflowing","dirty","contaminated","dangerous","accident",
              "delay","cut","irregular","missing","stolen","blocked","flooded",
              "neglected","ignored","unacceptable","disgusting","disease","sick",
              "outage","stagnant","leaking","burst","collapsed","no water",
              "no electricity","power cut","not working","not cleaned"}

def _keyword_sentiment(texts: List[str]) -> List[Dict]:
    results = []
    for text in texts:
        t = text.lower()
        pos = sum(1 for w in _POS_WORDS if w in t)
        neg = sum(1 for w in _NEG_WORDS if w in t)
        if neg > pos:
            label, score = "NEGATIVE", round(0.5 + min(neg * 0.1, 0.45), 3)
        elif pos > neg:
            label, score = "POSITIVE", round(0.5 + min(pos * 0.1, 0.45), 3)
        else:
            label, score = "NEUTRAL", 0.5
        results.append({"label": label, "score": score,
                        "scores": {"POSITIVE": 0.0, "NEUTRAL": 0.0, "NEGATIVE": 0.0}})
    return results


# ─────────────────────────────────────────────
# MAIN SENTIMENT FUNCTION
# ─────────────────────────────────────────────

def analyze_sentiment(texts: List[str], use_bert: bool = True) -> List[Dict]:
    """
    Analyze sentiment for a list of texts.
    Returns list of {label, score, scores} dicts.
    """
    if not texts:
        return []

    if use_bert:
        results = _bert_predict(texts)
        if results:
            return results

    # Fallback to VADER
    return _vader_predict(texts)


def analyze_dataframe(df: pd.DataFrame, text_col: str = "sentiment_text", use_bert: bool = True) -> pd.DataFrame:
    """
    Add sentiment columns to a DataFrame.
    Adds: sentiment_label, sentiment_score, sentiment_color
    """
    texts = df[text_col].fillna("").tolist()
    print(f"🔍 Analyzing sentiment for {len(texts)} records...")

    results = analyze_sentiment(texts, use_bert=use_bert)

    df["sentiment_label"] = [r["label"] for r in results]
    df["sentiment_score"] = [r["score"] for r in results]

    # Color mapping for visualizations
    color_map = {"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"}
    df["sentiment_color"] = df["sentiment_label"].map(color_map)

    # Numeric score: -1 (negative) to +1 (positive)
    def to_numeric(row):
        if row["sentiment_label"] == "POSITIVE":
            return row["sentiment_score"]
        elif row["sentiment_label"] == "NEGATIVE":
            return -row["sentiment_score"]
        return 0.0

    df["sentiment_numeric"] = df.apply(to_numeric, axis=1)

    print("✅ Sentiment analysis complete.")
    print(df["sentiment_label"].value_counts())
    return df


def get_urgency_score(df: pd.DataFrame) -> float:
    """
    Calculate overall urgency score (0-10) for city officials.
    Based on negative sentiment ratio + engagement metrics.
    """
    total = len(df)
    if total == 0:
        return 0.0

    negative_ratio = (df["sentiment_label"] == "NEGATIVE").sum() / total

    # Engagement weight (high likes/retweets = more urgent)
    avg_engagement = df[["likes", "retweets"]].sum(axis=1).mean()
    engagement_weight = min(avg_engagement / 100, 1.0)  # cap at 1.0

    urgency = (negative_ratio * 7) + (engagement_weight * 3)
    return round(min(urgency, 10.0), 1)


if __name__ == "__main__":
    test_texts = [
        "The roads in Banjara Hills are full of potholes! My car got damaged.",
        "New park in Jubilee Hills looks beautiful. Great work by GHMC!",
        "Water supply in our area is okay but could be better.",
        "Power cut for 6 hours and nobody from electricity board responded.",
        "Bus service has improved a lot recently. Very happy.",
    ]

    results = analyze_sentiment(test_texts, use_bert=False)  # VADER for quick test
    for text, result in zip(test_texts, results):
        print(f"{result['label']:8} ({result['score']:.2f}) | {text[:60]}")