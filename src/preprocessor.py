# src/preprocessor.py
"""
Text Preprocessing Pipeline
Handles cleaning, tokenization, stopword removal, normalization
"""

import re
import string
import pandas as pd
import nltk

# Download NLTK resources on first run (graceful fallback if offline)
_BUILTIN_STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "yourself","yourselves","he","him","his","himself","she","her","hers",
    "herself","it","its","itself","they","them","their","theirs","themselves",
    "what","which","who","whom","this","that","these","those","am","is","are",
    "was","were","be","been","being","have","has","had","having","do","does",
    "did","doing","a","an","the","and","but","if","or","because","as","until",
    "while","of","at","by","for","with","about","against","between","into",
    "through","during","before","after","above","below","to","from","up","down",
    "in","out","on","off","over","under","again","further","then","once","here",
    "there","when","where","why","how","all","both","each","few","more","most",
    "other","some","such","no","nor","not","only","own","same","so","than",
    "too","very","s","t","can","will","just","don","should","now","d","ll",
    "m","o","re","ve","y","ain","aren","couldn","didn","doesn","hadn","hasn",
    "haven","isn","ma","mightn","mustn","needn","shan","shouldn","wasn",
    "weren","won","wouldn","also","us","its"
}

def _setup_nltk():
    for pkg in ["stopwords", "punkt", "wordnet", "averaged_perceptron_tagger", "punkt_tab"]:
        try:
            nltk.data.find(f"tokenizers/{pkg}" if "punkt" in pkg else f"corpora/{pkg}")
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

_setup_nltk()

try:
    from nltk.corpus import stopwords as _sw
    STOP_WORDS = set(_sw.words("english"))
except Exception:
    STOP_WORDS = _BUILTIN_STOPWORDS

try:
    from nltk.stem import WordNetLemmatizer
    LEMMATIZER = WordNetLemmatizer()
    _HAS_LEMMATIZER = True
except Exception:
    _HAS_LEMMATIZER = False

try:
    from nltk.tokenize import word_tokenize as _wt
    _HAS_TOKENIZER = True
except Exception:
    _HAS_TOKENIZER = False

# City-specific abbreviations and slang normalization
CITY_SLANG = {
    r"\bbmc\b": "municipal corporation",
    r"\bnmc\b": "municipal corporation",
    r"\bghmc\b": "greater hyderabad municipal corporation",
    r"\bbbmp\b": "municipal corporation",
    r"\bmcd\b": "municipal corporation delhi",
    r"\bpwd\b": "public works department",
    r"\bescom\b": "electricity board",
    r"\btsrtc\b": "transport corporation",
    r"\bbescom\b": "electricity board",
    r"\bnot wrking\b": "not working",
    r"\bplz\b": "please",
    r"\bpls\b": "please",
    r"\bfx\b": "fix",
    r"\bu\b": "you",
    r"\br\b": "are",
    r"\bwont\b": "will not",
    r"\bcant\b": "cannot",
    r"\bdont\b": "do not",
    r"\bisnt\b": "is not",
    r"\bhasnt\b": "has not",
}


def normalize_slang(text: str) -> str:
    text = text.lower()
    for pattern, replacement in CITY_SLANG.items():
        text = re.sub(pattern, replacement, text)
    return text


def clean_text(text: str, for_sentiment: bool = False) -> str:
    """
    Full text cleaning pipeline.
    
    Args:
        text: raw input text
        for_sentiment: if True, keep more context (for BERT models)
                       if False, aggressive clean (for topic modeling)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # Normalize city-specific slang
    text = normalize_slang(text)

    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # Remove @mentions (keep for context awareness)
    text = re.sub(r"@\w+", "", text)

    # Remove hashtags (keep text, remove symbol)
    text = re.sub(r"#(\w+)", r"\1", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", "", text)

    # Remove emojis and special unicode
    text = text.encode("ascii", "ignore").decode("ascii")

    if for_sentiment:
        # For sentiment: keep punctuation, just normalize spacing
        text = re.sub(r"\s+", " ", text).strip()
        return text

    # For topic modeling: aggressive clean
    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize and lemmatize
    if _HAS_TOKENIZER:
        try:
            tokens = _wt(text)
        except Exception:
            tokens = text.split()
    else:
        tokens = text.split()

    if _HAS_LEMMATIZER:
        try:
            tokens = [LEMMATIZER.lemmatize(t) for t in tokens
                      if t not in STOP_WORDS and len(t) > 2]
        except Exception:
            tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]
    else:
        tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 2]

    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply preprocessing to the full dataframe.
    Adds cleaned_text and clean_for_sentiment columns.
    """
    print(f"⚙️  Preprocessing {len(df)} records...")

    # Clean for topic modeling (aggressive)
    df["cleaned_text"] = df["text"].apply(lambda x: clean_text(x, for_sentiment=False))

    # Clean for sentiment (minimal, preserve context)
    df["sentiment_text"] = df["text"].apply(lambda x: clean_text(x, for_sentiment=True))

    # Word count feature
    df["word_count"] = df["cleaned_text"].apply(lambda x: len(x.split()))

    # Remove empty rows after cleaning
    df = df[df["cleaned_text"].str.strip().str.len() > 5].reset_index(drop=True)

    # Parse timestamp
    try:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["date"] = df["timestamp"].dt.date
        df["month"] = df["timestamp"].dt.to_period("M").astype(str)
        df["hour"] = df["timestamp"].dt.hour
        df["weekday"] = df["timestamp"].dt.day_name()
    except Exception:
        df["date"] = pd.Timestamp.now().date()
        df["month"] = "2024-01"
        df["hour"] = 12
        df["weekday"] = "Monday"

    print(f"✅ Preprocessing complete. {len(df)} valid records.")
    return df


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.data_collector import generate_synthetic_data

    df = generate_synthetic_data(n=50)
    df = preprocess_dataframe(df)
    print(df[["text", "cleaned_text", "sentiment_text", "word_count"]].head(5))