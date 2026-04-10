# src/topic_modeler.py
"""
Topic Modeling Module
Primary:  BERTopic (transformer-based, best quality)
Fallback: LDA with Gensim (lightweight)
"""

import os
import re
import pickle
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict


# Department keyword mapping for auto-labeling topics
DEPARTMENT_KEYWORDS = {
    "Roads & Infrastructure": [
        "road", "pothole", "traffic", "footpath", "bridge", "construction",
        "repair", "signal", "divider", "pavement", "highway", "street",
    ],
    "Water Supply": [
        "water", "pipe", "supply", "tank", "drainage", "sewage", "leak",
        "pressure", "drinking", "contamination", "flood", "pump",
    ],
    "Sanitation": [
        "garbage", "waste", "dustbin", "trash", "clean", "drain", "stink",
        "smell", "dump", "litter", "sweeping", "sanitation", "hygiene",
    ],
    "Electricity": [
        "power", "light", "electricity", "transformer", "voltage", "wire",
        "streetlight", "electric", "outage", "cut", "meter",
    ],
    "Public Transport": [
        "bus", "auto", "metro", "train", "transport", "route", "stop",
        "delay", "fare", "ticket", "driver", "overcrowd",
    ],
    "Parks & Recreation": [
        "park", "garden", "playground", "bench", "tree", "green",
        "recreation", "gym", "children", "plant", "lake",
    ],
    "Public Safety": [
        "police", "crime", "theft", "safety", "accident", "dangerous",
        "speed", "patrol", "security", "cctv", "harassment",
    ],
}


def map_keywords_to_department(keywords: List[str]) -> str:
    """Map a list of topic keywords to the best-matching department."""
    keyword_text = " ".join(keywords).lower()
    scores = {}
    for dept, dept_keywords in DEPARTMENT_KEYWORDS.items():
        score = sum(1 for kw in dept_keywords if kw in keyword_text)
        scores[dept] = score

    best_dept = max(scores, key=scores.get)
    return best_dept if scores[best_dept] > 0 else "General Services"


# ─────────────────────────────────────────────
# BERTOPIC  (primary - best quality)
# ─────────────────────────────────────────────

def run_bertopic(documents: List[str], n_topics: int = 8) -> Tuple[object, pd.DataFrame, List[int]]:
    """
    Run BERTopic on documents.
    Returns: (model, topic_info_df, topic_assignments)
    """
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer

        print("⏳ Running BERTopic (first run downloads embedding model ~80MB)...")

        embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

        topic_model = BERTopic(
            embedding_model=embedding_model,
            nr_topics=n_topics,
            min_topic_size=max(5, len(documents) // 20),
            verbose=False,
        )

        topics, probs = topic_model.fit_transform(documents)
        topic_info = topic_model.get_topic_info()

        # Add department labels to topics
        def get_dept(row):
            if row["Topic"] == -1:
                return "Miscellaneous"
            kw_list = [kw for kw, _ in topic_model.get_topic(row["Topic"])]
            return map_keywords_to_department(kw_list)

        topic_info["Department"] = topic_info.apply(get_dept, axis=1)

        print(f"✅ BERTopic found {len(topic_info) - 1} topics.")

        # Save model
        os.makedirs("models", exist_ok=True)
        with open("models/topic_model.pkl", "wb") as f:
            pickle.dump(topic_model, f)

        return topic_model, topic_info, topics

    except Exception as e:
        print(f"⚠️  BERTopic failed ({e}). Falling back to LDA.")
        return run_lda(documents, n_topics)


# ─────────────────────────────────────────────
# LDA  (fallback)
# ─────────────────────────────────────────────

def run_lda(documents: List[str], n_topics: int = 8) -> Tuple[object, pd.DataFrame, List[int]]:
    """
    Run LDA topic modeling with Gensim.
    Fallback when BERTopic is unavailable.
    """
    try:
        import gensim
        import gensim.corpora as corpora
        from gensim.models import LdaModel

        print("⏳ Running LDA topic modeling...")

        tokenized = [doc.split() for doc in documents]
        tokenized = [t for t in tokenized if len(t) > 2]

        dictionary = corpora.Dictionary(tokenized)
        dictionary.filter_extremes(no_below=2, no_above=0.9)
        corpus = [dictionary.doc2bow(text) for text in tokenized]

        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=n_topics,
            random_state=42,
            alpha="auto",
            passes=10,
        )

        # Get dominant topic for each doc
        topic_assignments = []
        for bow in corpus:
            topic_dist = lda_model.get_document_topics(bow)
            if topic_dist:
                dominant = max(topic_dist, key=lambda x: x[1])[0]
            else:
                dominant = -1
            topic_assignments.append(dominant)

        # Pad if needed
        while len(topic_assignments) < len(documents):
            topic_assignments.append(-1)

        # Build topic_info DataFrame
        rows = []
        for topic_id in range(n_topics):
            keywords = [kw for kw, _ in lda_model.show_topic(topic_id, topn=10)]
            dept = map_keywords_to_department(keywords)
            count = topic_assignments.count(topic_id)
            rows.append({
                "Topic": topic_id,
                "Count": count,
                "Name": f"Topic_{topic_id}",
                "Keywords": ", ".join(keywords[:5]),
                "Department": dept,
            })

        # Add outliers row
        rows.insert(0, {"Topic": -1, "Count": topic_assignments.count(-1),
                        "Name": "Outliers", "Keywords": "", "Department": "Miscellaneous"})

        topic_info = pd.DataFrame(rows)
        print(f"✅ LDA found {n_topics} topics.")
        return lda_model, topic_info, topic_assignments

    except Exception as e:
        print(f"⚠️  LDA also failed: {e}. Using keyword-based fallback.")
        return _keyword_fallback(documents)


# ─────────────────────────────────────────────
# KEYWORD FALLBACK  (always works)
# ─────────────────────────────────────────────

def _keyword_fallback(documents: List[str]) -> Tuple[None, pd.DataFrame, List[int]]:
    """Zero-dependency keyword matching fallback."""
    assignments = []
    dept_list = list(DEPARTMENT_KEYWORDS.keys())

    for doc in documents:
        doc_lower = doc.lower()
        scores = {
            dept: sum(1 for kw in kws if kw in doc_lower)
            for dept, kws in DEPARTMENT_KEYWORDS.items()
        }
        best = max(scores, key=scores.get)
        idx = dept_list.index(best) if scores[best] > 0 else -1
        assignments.append(idx)

    # Build topic_info
    rows = []
    for i, dept in enumerate(dept_list):
        count = assignments.count(i)
        rows.append({
            "Topic": i,
            "Count": count,
            "Name": dept,
            "Keywords": ", ".join(DEPARTMENT_KEYWORDS[dept][:5]),
            "Department": dept,
        })
    rows.insert(0, {"Topic": -1, "Count": assignments.count(-1),
                    "Name": "Outliers", "Keywords": "", "Department": "Miscellaneous"})

    return None, pd.DataFrame(rows), assignments


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def extract_topics(
    df: pd.DataFrame,
    text_col: str = "cleaned_text",
    n_topics: int = 8,
    method: str = "auto",
) -> pd.DataFrame:
    """
    Run topic modeling on dataframe. Adds topic_id and topic_label columns.
    method: 'bertopic' | 'lda' | 'auto' (tries BERTopic first)
    """
    documents = df[text_col].fillna("").tolist()
    documents = [d for d in documents if len(d.split()) > 3]

    if len(documents) < 10:
        print("⚠️  Too few documents for topic modeling. Using keyword matching.")
        _, topic_info, assignments = _keyword_fallback(documents)
    elif method == "lda":
        _, topic_info, assignments = run_lda(documents, n_topics)
    elif method == "bertopic":
        _, topic_info, assignments = run_bertopic(documents, n_topics)
    else:  # auto
        _, topic_info, assignments = run_bertopic(documents, n_topics)

    # Assign back to dataframe (handle length mismatch)
    valid_docs = df[df[text_col].fillna("").apply(lambda x: len(x.split()) > 3)].index
    assignment_series = pd.Series(assignments, index=valid_docs[:len(assignments)])
    df["topic_id"] = df.index.map(assignment_series).fillna(-1).astype(int)

    # Map topic_id to department label
    topic_to_dept = dict(zip(topic_info["Topic"], topic_info["Department"]))
    df["topic_label"] = df["topic_id"].map(topic_to_dept).fillna("General Services")

    print(f"\nTopic distribution:\n{df['topic_label'].value_counts()}")
    return df, topic_info


def get_topic_keywords(topic_info: pd.DataFrame, topic_id: int) -> List[str]:
    """Get keywords for a specific topic."""
    row = topic_info[topic_info["Topic"] == topic_id]
    if row.empty:
        return []
    keywords_str = row.iloc[0].get("Keywords", "")
    return [kw.strip() for kw in keywords_str.split(",") if kw.strip()]


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_collector import generate_synthetic_data
    from src.preprocessor import preprocess_dataframe

    df = generate_synthetic_data(n=100)
    df = preprocess_dataframe(df)
    df, topic_info = extract_topics(df, method="auto")
    print("\nTopic Info:")
    print(topic_info[["Topic", "Count", "Department", "Keywords"]].to_string())