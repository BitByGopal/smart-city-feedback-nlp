# 🏙️ Smart City Public Feedback Analysis System

<div align="center">

**An end-to-end Deep NLP + LLM pipeline that analyzes citizen feedback to help Smart City officials make data-driven decisions.**

[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![HuggingFace](https://img.shields.io/badge/🤗-Transformers-FFD21E?style=for-the-badge)](https://huggingface.co)
[![Gemini](https://img.shields.io/badge/Gemini-1.5_Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=for-the-badge)](LICENSE)

[🚀 Live Demo](https://smart-city-feedback-nlp-gollagopal.streamlit.app/) • [📖 How It Works](#-how-it-works) • [⚡ Quick Start](#-quick-start)

</div>

---

## 🎯 What This System Does

Citizens report problems — potholes, water supply failures, garbage, power cuts — across social media, surveys, and government portals. This system:

1. **Collects** feedback from Twitter, Reddit, CSV uploads, and open datasets
2. **Analyzes sentiment** using BERT — detecting negative, positive, neutral tone
3. **Extracts topics** — automatically categorizes issues by city department
4. **Identifies locations** — finds which areas have the most complaints using NER
5. **Scores urgency** — ranks complaints so officials know what to fix first
6. **Generates AI reports** — Gemini LLM writes executive summaries automatically
7. **Answers questions** — chatbot lets officials query the data in plain English

---

## ✨ Key Features

| Feature | Technology | Description |
|---|---|---|
| 🧠 Sentiment Analysis | RoBERTa (BERT) | Classifies feedback as Positive / Negative / Neutral |
| 📌 Topic Modeling | BERTopic + LDA | Groups complaints by department automatically |
| 🗺️ Named Entity Recognition | spaCy | Extracts city areas and organizations from text |
| ⚡ Priority Scoring | Custom Algorithm | Urgency score 0–10 for complaint triage |
| 🤖 LLM Reports | Gemini 1.5 Flash | AI-generated executive summaries |
| 💬 Q&A Chatbot | Gemini API | Ask questions about feedback in plain English |
| 📊 Interactive Dashboard | Streamlit + Plotly | 4-tab visual analytics dashboard |
| 🔌 REST API | FastAPI | 8 endpoints for system integration |
| 🌍 Multi-City Support | Built-in | Hyderabad, Bengaluru, Mumbai, Delhi, Chennai, NYC, Pune |

---

## 🛠️ Tech Stack

```
Dashboard      →  Streamlit + Plotly
NLP Models     →  HuggingFace Transformers (RoBERTa), spaCy, BERTopic
LLM            →  Google Gemini 1.5 Flash
Backend API    →  FastAPI + Uvicorn
Data Sources   →  Twitter API v2, Reddit PRAW, NYC 311 Open Data, Synthetic
Language       →  Python 3.11
```

---

## ⚡ Quick Start

### 1. Clone
```bash
git clone https://github.com/BitByGopal/smart-city-feedback-nlp.git
cd smart-city-feedback-nlp
```

### 2. Install
```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 3. Configure API keys (optional)
```bash
cp .env.example .env
# Add GEMINI_API_KEY=your_key  ← free at aistudio.google.com
```
> Works without any API keys using built-in synthetic data.

### 4. Run dashboard
```bash
python -m streamlit run app.py
# Open: http://localhost:8501
```

### 5. Terminal demo
```bash
python demo.py
```

### 6. REST API
```bash
python -m uvicorn api:app --reload
# Docs: http://localhost:8000/docs
```

---

## 📁 Project Structure

```
smart-city-feedback-nlp/
├── app.py                    # Streamlit dashboard
├── api.py                    # FastAPI REST backend
├── demo.py                   # Terminal demo
├── requirements.txt
├── .env.example
│
├── src/
│   ├── data_collector.py     # Multi-source + synthetic data
│   ├── preprocessor.py       # Text cleaning + normalization
│   ├── sentiment_analyzer.py # BERT sentiment (+ fallback)
│   ├── topic_modeler.py      # BERTopic + LDA
│   ├── ner_extractor.py      # Named Entity Recognition
│   ├── priority_scorer.py    # Urgency scoring + alerts
│   ├── llm_reporter.py       # Gemini report + chatbot
│   └── fine_tuner.py         # BERT fine-tuning script
│
├── notebooks/
│   └── EDA_and_Analysis.ipynb  # 36-cell analysis notebook
│
└── data/
    ├── raw/
    └── processed/
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API info |
| POST | `/analyze` | Sentiment + NER for single text |
| POST | `/batch` | Batch analysis (up to 100 texts) |
| POST | `/entities` | Named entity extraction |
| GET | `/urgency?text=...` | Urgency score 0–5 |
| POST | `/chat` | LLM Q&A chatbot |
| POST | `/report` | Generate executive report |
| POST | `/upload-csv` | Analyze uploaded CSV |

**Example request:**
```bash
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "Huge pothole near Banjara Hills. Very dangerous!"}'
```
```json
{
  "sentiment": {"label": "NEGATIVE", "score": 0.94},
  "entities": {"locations": ["Banjara Hills"]},
  "urgency_score": 3.5
}
```

---

## 🌍 Supported Cities

Hyderabad • Bengaluru • Mumbai • Delhi • Chennai • New York City • Pune • Any custom city

---

## 📊 How It Works

```
Citizen Feedback (Twitter / Reddit / CSV / Survey)
           ↓
   Preprocessing  (clean · tokenize · normalize)
           ↓
   BERT Sentiment  +  BERTopic / LDA  +  spaCy NER
           ↓
   Priority Scoring  →  Urgency 0–10, Alert levels
           ↓
   Gemini LLM  →  Executive Report + Q&A Chatbot
           ↓
   Streamlit Dashboard  +  FastAPI REST API
```

---

## 🚀 Deployment

### Streamlit Cloud (Free)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → set main file: `app.py`
4. Add secret: `GEMINI_API_KEY = "your_key"`
5. Click **Deploy** ✅

### HuggingFace Spaces
1. New Space → choose **Streamlit** SDK
2. Connect GitHub or upload files
3. Add `GEMINI_API_KEY` in Space secrets

---

## 🎓 About

**Project:** NLP with Deep Learning — University Project  
**Student:** Golla Gopal Yadav  
**University:** Lovely Professional University, B.Tech CSE AI & ML (2027)  
**GitHub:** [@BitByGopal](https://github.com/BitByGopal)  
**LinkedIn:** [golla-gopal](https://linkedin.com/in/golla-gopal)  

---

## 📄 License

MIT — free to use for academic and personal projects.

---

<div align="center">
Built with ❤️ by <a href="https://github.com/BitByGopal">Golla Gopal</a><br>
⭐ Star this repo if it helped you!
</div>
