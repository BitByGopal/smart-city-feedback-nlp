# app.py
"""
Smart City Public Feedback Analysis System
Main Streamlit Dashboard
Run: streamlit run app.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="SmartCity Feedback AI",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: white; }
    .metric-card { background: #f8f9fa; border-radius: 10px; padding: 1rem; border-left: 4px solid #0066cc; }
    .alert-critical { background: #fff0f0; border-left: 4px solid #e74c3c; padding: 0.5rem 1rem; border-radius: 5px; }
    .alert-high { background: #fff8f0; border-left: 4px solid #e67e22; padding: 0.5rem 1rem; border-radius: 5px; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
    div[data-testid="metric-container"] { background-color: #f0f4ff; border-radius: 10px; padding: 0.75rem; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# IMPORTS (with error handling)
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def import_modules():
    from src.data_collector import load_data
    from src.preprocessor import preprocess_dataframe
    from src.sentiment_analyzer import analyze_dataframe
    from src.topic_modeler import extract_topics
    from src.ner_extractor import extract_entities_batch, get_location_frequency
    from src.priority_scorer import score_dataframe, get_top_complaints, get_department_summary, get_overall_city_score
    from src.llm_reporter import generate_city_report, answer_query
    return (load_data, preprocess_dataframe, analyze_dataframe, extract_topics,
            extract_entities_batch, get_location_frequency, score_dataframe,
            get_top_complaints, get_department_summary, get_overall_city_score,
            generate_city_report, answer_query)

(load_data, preprocess_dataframe, analyze_dataframe, extract_topics,
 extract_entities_batch, get_location_frequency, score_dataframe,
 get_top_complaints, get_department_summary, get_overall_city_score,
 generate_city_report, answer_query) = import_modules()


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/city.png", width=64)
    st.title("⚙️ Configuration")
    st.divider()

    city_name = st.text_input("🏙️ City Name", value="Hyderabad")

    data_source = st.selectbox(
        "📡 Data Source",
        ["Synthetic Data (Demo)", "Upload CSV", "NYC 311 Open Data"],
    )

    n_records = st.slider("📊 Sample Size", min_value=100, max_value=1000, value=300, step=50)

    use_bert = st.toggle("🤖 Use BERT Sentiment", value=False,
                         help="Enable for higher accuracy. Disable for faster processing.")

    topic_method = st.selectbox("📌 Topic Method", ["auto (BERTopic)", "lda", "keywords"])
    method_map = {"auto (BERTopic)": "auto", "lda": "lda", "keywords": "keywords"}

    st.divider()

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload CSV (must have 'text' column)", type=["csv"])

    run_btn = st.button("🚀 Analyze Feedback", type="primary", use_container_width=True)

    st.divider()
    st.caption("💡 Tip: Start with Synthetic Data to explore the dashboard.")
    st.caption("🔑 Add GEMINI_API_KEY in .env for AI-generated reports.")


# ─────────────────────────────────────────────
# DATA PIPELINE
# ─────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Running full NLP pipeline...", ttl=300)
def run_pipeline(source: str, city: str, n: int, use_bert: bool, method: str, csv_path: str = None):
    # Note: no st.spinner inside cached functions — causes ScriptRunContext error on cloud
    if source == "Upload CSV" and csv_path:
        df = load_data(source="csv", city=city, csv_path=csv_path)
    elif source == "NYC 311 Open Data":
        df = load_data(source="nyc311", city="New York City", n=n)
    else:
        df = load_data(source="synthetic", city=city, n=n)

    df = preprocess_dataframe(df)
    df = analyze_dataframe(df, use_bert=use_bert)
    df, topic_info = extract_topics(df, method=method)
    df = extract_entities_batch(df)
    df = score_dataframe(df)

    return df, topic_info


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────

st.markdown('<p class="main-header">🏙️ Smart City Public Feedback Analysis System</p>', unsafe_allow_html=True)
st.caption(f"Powered by Deep NLP + LLMs | {datetime.now().strftime('%d %b %Y, %I:%M %p')}")

# Load or run pipeline
if "df" not in st.session_state or run_btn:
    csv_path = None
    if uploaded_file:
        csv_path = f"/tmp/upload_{uploaded_file.name}"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    df, topic_info = run_pipeline(
        source=data_source,
        city=city_name,
        n=n_records,
        use_bert=use_bert,
        method=method_map.get(topic_method, "auto"),
        csv_path=csv_path,
    )
    st.session_state["df"] = df
    st.session_state["topic_info"] = topic_info
    st.session_state["city"] = city_name

df = st.session_state.get("df", None)
topic_info = st.session_state.get("topic_info", pd.DataFrame())

if df is None:
    st.info("👈 Configure settings and click **Analyze Feedback** to begin.")
    st.stop()

# Pre-compute KPIs
kpis = get_overall_city_score(df)
dept_summary = get_department_summary(df)
location_freq = get_location_frequency(df)
top_complaints_df = get_top_complaints(df, n=10)

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview",
    "🔍 Sentiment Analysis",
    "📌 Topic Modeling",
    "🤖 AI Report + Chatbot",
])


# ═══════════════════════ TAB 1: OVERVIEW ═══════════════════════
with tab1:
    # KPI Metrics Row
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("📬 Total Feedback", f"{kpis['total']:,}", help="Total records analyzed")
    col2.metric("😟 Negative Sentiment", f"{kpis['negative_pct']}%",
                delta=f"-{round(kpis['negative_pct'] - 50, 1)}%" if kpis['negative_pct'] > 50 else None,
                delta_color="inverse")
    col3.metric("😊 Positive Sentiment", f"{kpis['positive_pct']}%")
    col4.metric("⚡ Urgency Score", f"{kpis['urgency_score']}/10",
                delta="High" if kpis['urgency_score'] > 6 else "Moderate")
    col5.metric("🔴 Critical Alerts", f"{kpis['critical_alerts']}", delta_color="inverse")

    st.divider()

    col_left, col_right = st.columns([1.5, 1])

    with col_left:
        # Sentiment trend over time
        st.subheader("📈 Sentiment Trend Over Time")
        if "date" in df.columns and "sentiment_label" in df.columns:
            trend = df.groupby(["date", "sentiment_label"]).size().reset_index(name="count")
            fig = px.line(
                trend, x="date", y="count", color="sentiment_label",
                color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"},
                labels={"count": "Feedback Count", "date": "Date", "sentiment_label": "Sentiment"},
            )
            fig.update_layout(height=300, margin=dict(t=20, b=20), legend_title="")
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Sentiment donut
        st.subheader("🥧 Sentiment Distribution")
        sentiment_counts = df["sentiment_label"].value_counts()
        fig_donut = go.Figure(data=[go.Pie(
            labels=sentiment_counts.index,
            values=sentiment_counts.values,
            hole=0.5,
            marker_colors=["#e74c3c", "#95a5a6", "#2ecc71"],
        )])
        fig_donut.update_layout(height=300, margin=dict(t=10, b=10),
                                 showlegend=True, legend_title="")
        st.plotly_chart(fig_donut, use_container_width=True)

    # Department bar chart
    st.subheader("🏛️ Complaints by Department")
    dept_col = "topic_label" if "topic_label" in df.columns else "department"
    dept_counts = df[dept_col].value_counts().reset_index()
    dept_counts.columns = ["Department", "Count"]
    fig_bar = px.bar(
        dept_counts, x="Count", y="Department", orientation="h",
        color="Count", color_continuous_scale="Blues",
        labels={"Count": "Number of Complaints"},
    )
    fig_bar.update_layout(height=350, margin=dict(t=20, b=20), coloraxis_showscale=False)
    st.plotly_chart(fig_bar, use_container_width=True)

    # Top complaints table
    st.subheader("🚨 Top Priority Complaints")
    display_cols = ["text", "area", "topic_label", "sentiment_label", "priority_score", "alert_level"]
    display_cols = [c for c in display_cols if c in df.columns]
    st.dataframe(
        top_complaints_df.head(10),
        use_container_width=True,
        column_config={
            "text": st.column_config.TextColumn("Feedback", width="large"),
            "priority_score": st.column_config.ProgressColumn("Priority", min_value=0, max_value=10),
        },
    )


# ═══════════════════════ TAB 2: SENTIMENT ═══════════════════════
with tab2:
    st.subheader("🔍 Deep Sentiment Analysis")

    col_a, col_b = st.columns(2)

    with col_a:
        # Sentiment by department
        st.markdown("#### Sentiment by Department")
        dept_col = "topic_label" if "topic_label" in df.columns else "department"
        dept_sent = df.groupby([dept_col, "sentiment_label"]).size().reset_index(name="count")
        fig_dept_sent = px.bar(
            dept_sent, x="count", y=dept_col, color="sentiment_label",
            orientation="h", barmode="stack",
            color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"},
        )
        fig_dept_sent.update_layout(height=350, margin=dict(t=10, b=10), legend_title="")
        st.plotly_chart(fig_dept_sent, use_container_width=True)

    with col_b:
        # Sentiment by hour of day
        st.markdown("#### Feedback Volume by Hour")
        if "hour" in df.columns:
            hourly = df.groupby(["hour", "sentiment_label"]).size().reset_index(name="count")
            fig_hour = px.bar(
                hourly, x="hour", y="count", color="sentiment_label",
                color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c", "NEUTRAL": "#95a5a6"},
                labels={"hour": "Hour of Day", "count": "Count"},
            )
            fig_hour.update_layout(height=350, margin=dict(t=10, b=10), legend_title="")
            st.plotly_chart(fig_hour, use_container_width=True)

    # Location heatmap
    st.markdown("#### 📍 Top Complaint Locations")
    if not location_freq.empty:
        fig_loc = px.bar(
            location_freq.head(15), x="location", y="count",
            color="count", color_continuous_scale="Reds",
            labels={"count": "Mentions", "location": "Area"},
        )
        fig_loc.update_layout(height=300, margin=dict(t=10, b=60), coloraxis_showscale=False)
        st.plotly_chart(fig_loc, use_container_width=True)

    # Raw data explorer
    st.markdown("#### 🔎 Explore Data")
    filter_sentiment = st.multiselect(
        "Filter by sentiment", ["NEGATIVE", "POSITIVE", "NEUTRAL"],
        default=["NEGATIVE", "POSITIVE", "NEUTRAL"]
    )
    filtered_df = df[df["sentiment_label"].isin(filter_sentiment)]
    show_cols = ["text", "area", "sentiment_label", "sentiment_score", "priority_score"]
    show_cols = [c for c in show_cols if c in filtered_df.columns]
    st.dataframe(filtered_df[show_cols].head(50), use_container_width=True)


# ═══════════════════════ TAB 3: TOPIC MODELING ═══════════════════════
with tab3:
    st.subheader("📌 Topic & Issue Analysis")

    col_t1, col_t2 = st.columns([1, 1])

    with col_t1:
        # Topic distribution pie
        st.markdown("#### Topic Distribution")
        dept_col = "topic_label" if "topic_label" in df.columns else "department"
        topic_counts = df[dept_col].value_counts().reset_index()
        topic_counts.columns = ["Topic", "Count"]
        fig_pie = px.pie(
            topic_counts, names="Topic", values="Count",
            color_discrete_sequence=px.colors.qualitative.Set3,
        )
        fig_pie.update_layout(height=380, margin=dict(t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_t2:
        # Department summary table
        st.markdown("#### Department Summary")
        st.dataframe(dept_summary, use_container_width=True, height=380)

    # Topic over time
    st.markdown("#### 📅 Topic Trends Over Time")
    if "date" in df.columns:
        topic_time = df.groupby(["date", dept_col]).size().reset_index(name="count")
        top_topics = df[dept_col].value_counts().head(5).index.tolist()
        topic_time_filtered = topic_time[topic_time[dept_col].isin(top_topics)]
        fig_trend = px.line(
            topic_time_filtered, x="date", y="count", color=dept_col,
            labels={"count": "Complaints", "date": "Date"},
            color_discrete_sequence=px.colors.qualitative.Set2,
        )
        fig_trend.update_layout(height=300, margin=dict(t=10, b=20), legend_title="")
        st.plotly_chart(fig_trend, use_container_width=True)

    # Alert level breakdown
    st.markdown("#### 🚦 Alert Level Distribution")
    if "alert_level" in df.columns:
        alert_counts = df["alert_level"].value_counts().reset_index()
        alert_counts.columns = ["Alert Level", "Count"]
        fig_alert = px.bar(
            alert_counts, x="Alert Level", y="Count",
            color="Alert Level",
            color_discrete_map={
                "🔴 CRITICAL": "#e74c3c",
                "🟠 HIGH": "#e67e22",
                "🟡 MEDIUM": "#f1c40f",
                "🟢 LOW": "#2ecc71",
            },
        )
        fig_alert.update_layout(height=280, margin=dict(t=10, b=20), showlegend=False)
        st.plotly_chart(fig_alert, use_container_width=True)


# ═══════════════════════ TAB 4: LLM REPORT + CHATBOT ═══════════════════════
with tab4:
    st.subheader("🤖 AI-Powered Insights")

    # Prepare aggregated data for LLM
    top_locations_str = ", ".join(
        f"{row['location']} ({row['count']})"
        for _, row in location_freq.head(5).iterrows()
    ) if not location_freq.empty else "Various areas"

    dept_summary_str = "\n".join(
        f"- {row['Department']}: {row['Total Complaints']} complaints ({row['Negative %']}% negative)"
        for _, row in dept_summary.iterrows()
    ) if not dept_summary.empty else "Not available"

    top_complaints_str = "\n".join(
        f"- {row['text'][:80]}..."
        for _, row in top_complaints_df.head(3).iterrows()
    ) if not top_complaints_df.empty else "Not available"

    aggregated_data = {
        "city": st.session_state.get("city", "City"),
        "days": 30,
        "total": kpis["total"],
        "negative_pct": kpis["negative_pct"],
        "positive_pct": kpis["positive_pct"],
        "urgency_score": kpis["urgency_score"],
        "critical_alerts": kpis["critical_alerts"],
        "high_alerts": kpis["high_alerts"],
        "top_issue": kpis.get("top_issue", "Infrastructure"),
        "dept_summary": dept_summary_str,
        "top_locations": top_locations_str,
        "top_complaints": top_complaints_str,
        "topic_distribution": dept_summary_str,
    }

    col_r1, col_r2 = st.columns([2, 1])

    with col_r1:
        st.markdown("### 📋 Executive Report")
        st.caption("AI-generated report for city officials. Add GEMINI_API_KEY in .env for full LLM report.")

        if st.button("🔄 Generate AI Report", type="primary"):
            with st.spinner("🤖 Generating report with AI..."):
                report = generate_city_report(aggregated_data)
                st.session_state["report"] = report

        if "report" in st.session_state:
            st.markdown(st.session_state["report"])
            st.download_button(
                "⬇️ Download Report (MD)",
                data=st.session_state["report"],
                file_name=f"city_report_{datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown",
            )

    with col_r2:
        # Quick stats for reference
        st.markdown("### 📊 Quick Stats")
        st.info(f"""
        **City:** {aggregated_data['city']}
        **Records:** {aggregated_data['total']:,}
        **Urgency:** {aggregated_data['urgency_score']}/10
        **Negative:** {aggregated_data['negative_pct']}%
        **Top Issue:** {aggregated_data['top_issue']}
        **Critical Alerts:** {aggregated_data['critical_alerts']}
        """)

    st.divider()

    # Chatbot
    st.markdown("### 💬 Ask About the City Data")
    st.caption("Ask questions about citizen feedback data. Powered by Gemini AI (or rule-based fallback).")

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []

    # Quick question buttons
    st.markdown("**Quick questions:**")
    q_cols = st.columns(3)
    quick_qs = [
        "What is the most urgent issue?",
        "Which area has most complaints?",
        "What % is negative sentiment?",
    ]
    for i, (col, q) in enumerate(zip(q_cols, quick_qs)):
        if col.button(q, key=f"quick_{i}"):
            st.session_state["pending_q"] = q

    # Text input — handle quick-question buttons cleanly
    user_input = st.chat_input("Type your question here...")
    if "pending_q" in st.session_state and not user_input:
        user_input = st.session_state.pop("pending_q")

    if user_input:
        st.session_state["chat_history"].append({"role": "user", "content": user_input})
        with st.spinner("🤖 Thinking..."):
            response = answer_query(user_input, aggregated_data)
        st.session_state["chat_history"].append({"role": "assistant", "content": response})

    # Display chat history
    for msg in st.session_state["chat_history"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if st.session_state["chat_history"]:
        if st.button("🗑️ Clear Chat"):
            st.session_state["chat_history"] = []
            st.rerun()
