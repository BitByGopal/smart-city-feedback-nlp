# src/ner_extractor.py
"""
Named Entity Recognition Module
Extracts: Locations (GPE, LOC), Organizations (ORG), Facilities (FAC)
Primary:  spaCy transformer model
Fallback: spaCy small model + regex
"""

import re
import pandas as pd
from typing import Dict, List
from collections import Counter

# ─────────────────────────────────────────────
# LOAD spaCy MODEL
# ─────────────────────────────────────────────

_nlp = None

def _load_spacy():
    global _nlp
    if _nlp is None:
        try:
            import spacy
            # Try transformer model first (best accuracy)
            try:
                _nlp = spacy.load("en_core_web_trf")
                print("✅ spaCy transformer model loaded.")
            except OSError:
                # Try small model
                try:
                    _nlp = spacy.load("en_core_web_sm")
                    print("✅ spaCy small model loaded.")
                except OSError:
                    # Auto download small model
                    import subprocess
                    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"], check=True)
                    _nlp = spacy.load("en_core_web_sm")
                    print("✅ spaCy small model downloaded and loaded.")
        except Exception as e:
            print(f"⚠️  spaCy load failed: {e}. Using regex fallback.")
            _nlp = None
    return _nlp


# ─────────────────────────────────────────────
# KNOWN CITY AREAS (enhance NER for Indian cities)
# ─────────────────────────────────────────────

KNOWN_AREAS = {
    # Hyderabad
    "banjara hills", "jubilee hills", "madhapur", "gachibowli", "hitech city",
    "kukatpally", "ameerpet", "begumpet", "secunderabad", "kondapur",
    "miyapur", "lb nagar", "dilsukhnagar", "mehdipatnam", "tolichowki",
    "uppal", "malakpet", "nampally", "abids", "koti", "somajiguda",
    "punjagutta", "sr nagar", "erra manzil", "masab tank",
    # Mumbai
    "andheri", "bandra", "dadar", "kurla", "thane", "borivali", "malad",
    "goregaon", "kandivali", "churchgate", "colaba", "worli", "lower parel",
    # Bangalore
    "koramangala", "indiranagar", "jayanagar", "btm layout", "hsr layout",
    "whitefield", "electronic city", "hebbal", "marathahalli", "jp nagar",
    # Delhi
    "connaught place", "saket", "dwarka", "rohini", "janakpuri", "pitampura",
    "lajpat nagar", "karol bagh", "nehru place", "noida", "gurugram",
    # NYC (for 311 dataset)
    "manhattan", "brooklyn", "queens", "bronx", "staten island",
    "harlem", "midtown", "downtown", "flushing", "jamaica", "astoria",
}


def extract_entities_spacy(text: str, nlp) -> Dict:
    """Extract entities using spaCy NLP model."""
    doc = nlp(text)
    entities = {
        "locations": [],
        "organizations": [],
        "facilities": [],
        "all_entities": [],
    }

    for ent in doc.ents:
        entity_text = ent.text.strip()
        if len(entity_text) < 2:
            continue

        entities["all_entities"].append({"text": entity_text, "label": ent.label_})

        if ent.label_ in ["GPE", "LOC"]:
            entities["locations"].append(entity_text)
        elif ent.label_ == "ORG":
            entities["organizations"].append(entity_text)
        elif ent.label_ == "FAC":
            entities["facilities"].append(entity_text)

    return entities


def extract_entities_regex(text: str) -> Dict:
    """
    Regex-based entity extraction fallback.
    Matches known area names + capitalized location patterns.
    """
    text_lower = text.lower()
    entities = {
        "locations": [],
        "organizations": [],
        "facilities": [],
        "all_entities": [],
    }

    # Match known city areas
    for area in KNOWN_AREAS:
        if area in text_lower:
            proper = area.title()
            entities["locations"].append(proper)
            entities["all_entities"].append({"text": proper, "label": "GPE"})

    # Match pattern: "near <Location>" or "in <Location>"
    location_patterns = [
        r"(?:near|in|at|from|to)\s+([A-Z][a-zA-Z\s]{2,20})",
        r"([A-Z][a-zA-Z]+\s+(?:Road|Street|Nagar|Colony|Hills|City|Park|Area|Zone|District))",
    ]
    for pattern in location_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            match = match.strip()
            if len(match) > 3:
                entities["locations"].append(match)
                entities["all_entities"].append({"text": match, "label": "LOC"})

    # Deduplicate
    entities["locations"] = list(dict.fromkeys(entities["locations"]))
    return entities


# ─────────────────────────────────────────────
# MAIN EXTRACT FUNCTION
# ─────────────────────────────────────────────

def extract_entities(text: str) -> Dict:
    """Extract named entities from a single text."""
    nlp = _load_spacy()
    if nlp:
        return extract_entities_spacy(text, nlp)
    return extract_entities_regex(text)


def extract_entities_batch(df: pd.DataFrame, text_col: str = "text") -> pd.DataFrame:
    """
    Run NER on entire DataFrame.
    Adds: extracted_locations, extracted_orgs columns.
    """
    print(f"🔍 Running NER on {len(df)} records...")
    nlp = _load_spacy()

    locations_list = []
    orgs_list = []

    for text in df[text_col].fillna("").tolist():
        if nlp:
            ents = extract_entities_spacy(text, nlp)
        else:
            ents = extract_entities_regex(text)

        locations_list.append(", ".join(ents["locations"][:3]) if ents["locations"] else "")
        orgs_list.append(", ".join(ents["organizations"][:2]) if ents["organizations"] else "")

    df["extracted_locations"] = locations_list
    df["extracted_orgs"] = orgs_list

    # Fill missing locations with the area column if available
    if "area" in df.columns:
        df["extracted_locations"] = df.apply(
            lambda row: row["extracted_locations"] if row["extracted_locations"]
            else row.get("area", ""),
            axis=1,
        )

    print("✅ NER complete.")
    return df


def get_location_frequency(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get frequency of mentioned locations.
    Returns DataFrame: location, count, avg_sentiment
    """
    all_locations = []
    for locs in df["extracted_locations"].fillna("").tolist():
        for loc in locs.split(","):
            loc = loc.strip()
            if loc and len(loc) > 2:
                all_locations.append(loc)

    if not all_locations:
        # Fall back to area column
        all_locations = df.get("area", pd.Series()).fillna("Unknown").tolist()

    location_counts = Counter(all_locations)
    location_df = pd.DataFrame(
        [(loc, count) for loc, count in location_counts.most_common(20)],
        columns=["location", "count"],
    )

    # Add average sentiment if available
    if "sentiment_numeric" in df.columns and "extracted_locations" in df.columns:
        loc_sentiments = {}
        for _, row in df.iterrows():
            locs = str(row.get("extracted_locations", "")).split(",")
            for loc in locs:
                loc = loc.strip()
                if loc:
                    if loc not in loc_sentiments:
                        loc_sentiments[loc] = []
                    loc_sentiments[loc].append(row.get("sentiment_numeric", 0))

        location_df["avg_sentiment"] = location_df["location"].map(
            lambda loc: round(sum(loc_sentiments.get(loc, [0])) / max(len(loc_sentiments.get(loc, [1])), 1), 3)
        )
    else:
        location_df["avg_sentiment"] = 0.0

    return location_df


if __name__ == "__main__":
    test_texts = [
        "Huge pothole on the road near Banjara Hills flyover. Please fix it!",
        "GHMC has not cleaned the Madhapur market area for a week. Very dirty.",
        "Water supply in Kondapur has been irregular since Tuesday.",
        "Street light near Gachibowli stadium is not working for 3 weeks.",
    ]

    for text in test_texts:
        ents = extract_entities(text)
        print(f"Text: {text[:50]}...")
        print(f"  Locations: {ents['locations']}")
        print(f"  Organizations: {ents['organizations']}")
        print()