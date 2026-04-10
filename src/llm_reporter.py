# src/llm_reporter.py
"""
LLM Report Generator & Chatbot Module
Primary:  Google Gemini 1.5 Flash (free tier, generous limits)
Fallback: Rule-based report template
"""

import os
import json
import pandas as pd
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()


# ─────────────────────────────────────────────
# GEMINI CLIENT
# ─────────────────────────────────────────────

_gemini_model = None

def _get_gemini_key():
    """Read key from .env first, then Streamlit Cloud secrets."""
    key = os.getenv("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        return None

def _load_gemini():
    global _gemini_model
    if _gemini_model is None:
        api_key = _get_gemini_key()
        if not api_key:
            print("⚠️  No GEMINI_API_KEY found. Using rule-based report fallback.")
            return None
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            _gemini_model = genai.GenerativeModel(
                "gemini-1.5-flash",
                generation_config={"temperature": 0.3, "max_output_tokens": 1500},
            )
            print("✅ Gemini model loaded.")
        except Exception as e:
            print(f"⚠️  Gemini load failed: {e}")
            _gemini_model = None
    return _gemini_model


def _call_gemini(prompt: str) -> str:
    model = _load_gemini()
    if model is None:
        return None
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"Gemini API error: {e}")
        return None


# ─────────────────────────────────────────────
# REPORT GENERATION
# ─────────────────────────────────────────────

def _build_report_context(data: Dict) -> str:
    """Build a structured context string from aggregated data."""
    return f"""
    CITY FEEDBACK ANALYSIS REPORT DATA
    ===================================
    City: {data.get('city', 'City')}
    Analysis Period: Last {data.get('days', 30)} days
    Total Feedback Records: {data.get('total', 0)}
    
    SENTIMENT BREAKDOWN:
    - Negative: {data.get('negative_pct', 0)}%
    - Positive: {data.get('positive_pct', 0)}%
    - Neutral: {100 - data.get('negative_pct', 0) - data.get('positive_pct', 0)}%
    
    URGENCY SCORE: {data.get('urgency_score', 5.0)}/10
    Critical Alerts: {data.get('critical_alerts', 0)}
    High Priority Alerts: {data.get('high_alerts', 0)}
    
    TOP DEPARTMENT COMPLAINTS:
    {data.get('dept_summary', 'Not available')}
    
    TOP LOCATIONS MENTIONED:
    {data.get('top_locations', 'Not available')}
    
    SAMPLE CRITICAL COMPLAINTS:
    {data.get('top_complaints', 'Not available')}
    
    TOPIC BREAKDOWN:
    {data.get('topic_distribution', 'Not available')}
    """


def generate_city_report(data: Dict) -> str:
    """
    Generate an executive summary report using Gemini LLM.
    Falls back to rule-based template if API key not set.
    """
    context = _build_report_context(data)

    prompt = f"""
    You are an AI Smart City Analyst preparing an executive report for municipal officials.
    
    {context}
    
    Generate a professional executive report with the following sections:
    
    ## Executive Summary
    (2-3 sentence overview of the city's feedback situation)
    
    ## Key Findings
    (3-5 bullet points of the most important insights)
    
    ## Priority Action Items
    (Top 3 issues requiring IMMEDIATE attention, with specific departments tagged)
    
    ## Department-wise Analysis
    (Brief analysis for each major department with complaint counts)
    
    ## Trend & Risk Assessment
    (Identify patterns, risks, and areas of concern)
    
    ## Recommended Actions
    (Specific, actionable recommendations for city officials)
    
    Keep the report professional, data-driven, and under 600 words.
    Use the actual numbers from the data. Be specific about locations and issues.
    """

    result = _call_gemini(prompt)
    if result:
        return result

    # Rule-based fallback
    return _generate_template_report(data)


def answer_query(question: str, data: Dict) -> str:
    """
    Answer a natural language question about the feedback data.
    Powers the chatbot tab in the dashboard.
    """
    context = _build_report_context(data)

    prompt = f"""
    You are a Smart City AI Assistant. You have access to citizen feedback analysis data.
    
    DATA CONTEXT:
    {context}
    
    CITY OFFICIAL'S QUESTION: {question}
    
    Instructions:
    - Answer based ONLY on the provided data context
    - Be specific and cite actual numbers when available
    - If the question cannot be answered from the data, say so clearly
    - Keep the answer concise (2-4 sentences max)
    - If suggesting actions, be specific about which department should handle it
    """

    result = _call_gemini(prompt)
    if result:
        return result

    # Fallback: keyword-based simple answers
    return _simple_keyword_answer(question, data)


def generate_department_brief(department: str, dept_data: Dict) -> str:
    """Generate a brief for a specific department."""
    prompt = f"""
    You are an AI analyst. Generate a 3-sentence brief for the {department} department 
    based on this citizen feedback summary:
    
    - Total complaints: {dept_data.get('total', 0)}
    - Negative sentiment: {dept_data.get('negative_pct', 0)}%
    - Average urgency: {dept_data.get('avg_priority', 5)}/10
    - Key areas mentioned: {dept_data.get('locations', 'Various')}
    - Top complaint keywords: {dept_data.get('keywords', 'Not specified')}
    
    Be specific, professional, and mention the urgency level.
    """
    result = _call_gemini(prompt)
    return result or f"The {department} department has {dept_data.get('total', 0)} complaints with {dept_data.get('negative_pct', 0)}% negative sentiment. Immediate attention is recommended."


# ─────────────────────────────────────────────
# RULE-BASED FALLBACKS (work without API key)
# ─────────────────────────────────────────────

def _generate_template_report(data: Dict) -> str:
    """Rule-based report when Gemini is not available."""
    city = data.get("city", "the city")
    total = data.get("total", 0)
    neg_pct = data.get("negative_pct", 0)
    pos_pct = data.get("positive_pct", 0)
    urgency = data.get("urgency_score", 5.0)
    top_issue = data.get("top_issue", "Infrastructure")
    critical = data.get("critical_alerts", 0)
    high = data.get("high_alerts", 0)

    urgency_level = "HIGH" if urgency >= 7 else "MODERATE" if urgency >= 4 else "LOW"

    return f"""## Executive Summary

Analysis of **{total} citizen feedback records** for {city} reveals an urgency level of **{urgency_level} ({urgency}/10)**. 
{neg_pct}% of feedback is negative, while {pos_pct}% reflects citizen satisfaction. 
The primary concern is **{top_issue}**, requiring immediate departmental attention.

## Key Findings

- 📊 **{total}** total feedback records analyzed across all departments
- ⚠️ **{neg_pct}%** negative sentiment — {urgency_level.lower()} urgency for city administration
- 🔴 **{critical} critical** and **{high} high-priority** alerts require immediate action
- 🏆 Top concern: **{top_issue}** — highest complaint volume across all areas
- 📈 Positive sentiment at {pos_pct}% indicates some citizen satisfaction areas

## Priority Action Items

1. **{top_issue} Department** — Address the highest volume of complaints immediately
2. **Critical Alert Resolution** — {critical} critical complaints need same-day response
3. **High Priority Backlog** — {high} high-priority issues need resolution within 48 hours

## Recommended Actions

1. Deploy rapid response teams to top complaint areas
2. Set up a real-time feedback monitoring system
3. Establish 24-hour helpline for critical infrastructure issues
4. Conduct weekly department-wise complaint review meetings
5. Publish monthly resolution status reports for citizen transparency

---
*Report generated by Smart City Feedback AI System*
"""


def _simple_keyword_answer(question: str, data: Dict) -> str:
    """Simple keyword-based Q&A fallback."""
    q = question.lower()
    total = data.get("total", 0)
    neg_pct = data.get("negative_pct", 0)
    urgency = data.get("urgency_score", 5.0)
    top_issue = data.get("top_issue", "Infrastructure")

    if any(w in q for w in ["how many", "total", "count", "number"]):
        return f"There are **{total}** total feedback records in the current analysis period."
    elif any(w in q for w in ["negative", "complaint", "worst"]):
        return f"**{neg_pct}%** of feedback is negative. The top complaint area is **{top_issue}**."
    elif any(w in q for w in ["urgent", "critical", "priority", "score"]):
        return f"The overall urgency score is **{urgency}/10**. There are {data.get('critical_alerts', 0)} critical and {data.get('high_alerts', 0)} high-priority alerts."
    elif any(w in q for w in ["positive", "good", "best", "happy"]):
        return f"**{data.get('positive_pct', 0)}%** of feedback is positive. Citizens are most satisfied in areas where complaints are low."
    elif any(w in q for w in ["area", "location", "where", "zone"]):
        return f"Top locations mentioned: {data.get('top_locations', 'Various areas across the city')}."
    else:
        return f"Based on the current data of {total} records: urgency score is {urgency}/10, {neg_pct}% negative sentiment, with **{top_issue}** as the primary concern."


if __name__ == "__main__":
    # Test with mock data
    mock_data = {
        "city": "Hyderabad",
        "days": 30,
        "total": 450,
        "negative_pct": 58.3,
        "positive_pct": 22.1,
        "urgency_score": 6.8,
        "critical_alerts": 23,
        "high_alerts": 87,
        "top_issue": "Roads & Infrastructure",
        "dept_summary": "Roads: 145, Water: 98, Sanitation: 87, Electricity: 65",
        "top_locations": "Banjara Hills (45), Madhapur (38), Kukatpally (31)",
        "top_complaints": "Pothole on main road, Water supply irregular, Garbage not collected",
        "topic_distribution": "Roads 32%, Water 22%, Sanitation 19%, Electricity 14%, Others 13%",
    }

    print("=== Testing Template Report (no API key) ===\n")
    report = generate_city_report(mock_data)
    print(report)

    print("\n=== Testing Q&A Fallback ===\n")
    questions = [
        "How many complaints are there?",
        "What is the urgency score?",
        "Which area has the most negative feedback?",
    ]
    for q in questions:
        print(f"Q: {q}")
        print(f"A: {answer_query(q, mock_data)}\n")