import os
import sys
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

import spacy
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# --- Backend imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.analyze_user_input import extract_user_features
from backend.scoring import compare_to_baseline

# --- Streamlit setup
st.set_page_config(page_title="🧠 Cognitive Wellness Screener", layout="wide")

# --- Local lightweight models for visuals
_nlp = spacy.load("en_core_web_sm")
_sent_embed = SentenceTransformer("all-MiniLM-L6-v2")
_vader = SentimentIntensityAnalyzer()

# --- OpenAI client
def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI()
    except Exception:
        return None

client = _openai_client()

# --- Helpers and friendly naming
FRIENDLY_NAMES = {
    "semantic_coherence": "Sentence Flow",
    "global_coherence_mean": "Topic Focus",
    "global_coherence_sd": "Topic Stability",
    "entity_consistency_score": "Idea Consistency",
    "temporal_stability_score": "Timeline Clarity",
    "mtld": "Word Diversity",
    "type_token_ratio": "Word Variety",
    "avg_clauses_per_sentence": "Sentence Complexity",
    "simple_sentence_ratio": "Simple Sentences",
    "fragment_rate": "Sequence Fragments",
    "repetition_ratio": "Repetitiveness",
    "fillers_per_100w": "Fillers (per 100 words)",
    "temporal_markers_per_100w": "Time References (per 100 words)",
    "emotion_entropy": "Emotion Range",
    "emotion_volatility": "Emotion Variability",
    "avg_sentiment": "Overall Tone",
    "sentiment_variance": "Tone Changes",
    "sentiment_range": "Emotional Range",
}

def nice_label(k):
    return FRIENDLY_NAMES.get(k, k.replace("_", " ").title())

def _sentences(text):
    return [s.text.strip() for s in _nlp(text).sents if s.text.strip()]

def zscore_map(results):
    out = {}
    for k, v in results.items():
        if isinstance(v, dict) and "z_score" in v:
            out[k] = v["z_score"]
    return out

# --- Simple visuals
def grade_badge(label, grade, score):
    color = {"A": "🟢", "B": "🟡", "C": "🟠", "D": "🔴"}.get(grade, "⚪️")
    st.markdown(f"### {color} **{label}: {grade}**  \nScore: {score:.1f}/100")

def group_bar_chart(title, your_score, mean_band=(70, 90)):
    fig, ax = plt.subplots(figsize=(4, 0.6))
    ax.barh([0], [100], color="#eee")
    ax.barh([0], [your_score], color="#4CAF50" if your_score >= 60 else "#FFC107")
    ax.axvspan(mean_band[0], mean_band[1], color="#90CAF9", alpha=0.3)
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Score")
    ax.set_title(title, fontsize=10)
    st.pyplot(fig, use_container_width=True)

def sentiment_trajectory(sentences):
    if not sentences:
        return None
    vals = [_vader.polarity_scores(s)["compound"] for s in sentences]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(range(1, len(vals) + 1), vals, marker="o", color="#4CAF50")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Valence Trajectory Across Sentences", fontsize=10)
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Valence")
    ax.grid(True, linewidth=0.3)
    return fig

def coherence_heatmap(sentences):
    if len(sentences) < 2:
        return None
    embs = _sent_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    sim = util.cos_sim(embs, embs).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(sim, vmin=-1, vmax=1, cmap="vlag", ax=ax, cbar=True)
    ax.set_title("Coherence Heatmap", fontsize=10)
    ax.set_xlabel("Sentence")
    ax.set_ylabel("Sentence")
    return fig

# --- GPT feedback
# --- GPT feedback
def coach_feedback(summary, zs):
    prompt = (
        "You are a warm, thoughtful writing coach giving personal feedback on a short piece of writing.\n\n"
        f"Cognitive Fluency Score: {summary.get('cognitive_fluency')}\n"
        f"Emotional Clarity Score: {summary.get('emotional_clarity')}\n\n"
        f"Feature deviations (z-scores): "
        f"{str({k: round(v, 2) for k, v in sorted(zs.items(), key=lambda x: -abs(x[1]))[:6]})}\n\n"
        "Write your feedback in a friendly and personal way, as if you're talking directly to the writer.\n"
        "Avoid numbers or technical terms. Focus on clarity, tone, and readability.\n"
        "Your response should sound human, gentle, and encouraging — not robotic.\n\n"
        "Structure your response as follows:\n"
        "1) A short greeting that feels personal and supportive (e.g., 'Hey, nice work on this entry!').\n"
        "2) One paragraph (2–3 sentences) on how the ideas connect and flow.\n"
        "3) One paragraph (2–3 sentences) on sentence structure and clarity — what reads smoothly, what could be improved.\n"
        "4) One paragraph (2–3 sentences) on emotional tone — how it comes across and how to make it stronger or more balanced.\n"
        "5) End with two simple and friendly suggestions for next time.\n"
        "Keep everything concise, positive, and easy to understand."
    )
    if not client:
        return (
            "AI feedback unavailable. Focus on connecting your ideas smoothly, varying your sentences, "
            "and making emotions come through naturally."
        )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a kind and empathetic writing coach offering clear, human feedback."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=350,
            temperature=0.7,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "AI coach temporarily unavailable. Try again later."


# --- App layout
st.title("🧠 Cognitive Wellness Screener")
st.markdown(
    "Reflect on your writing to explore how your **focus**, **structure**, and **emotional tone** compare with healthy patterns. "
    "This is for self-reflection, not diagnosis."
)

user_text = st.text_area("✍️ Write or paste your journal entry:", height=240, placeholder="Type a paragraph or two here...")

col_run, col_clear = st.columns([1, 1])
with col_run:
    run = st.button("Analyze", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.experimental_rerun()

if run and user_text.strip():
    with st.spinner("Analyzing your writing..."):
        feats = extract_user_features(user_text)
        results = compare_to_baseline(feats)
        summary = results.get("_summary", {})
        zs = zscore_map(results)

    st.markdown("## ✨ Summary Overview")

    col1, col2, col3 = st.columns(3)
    with col1:
        grade_badge("🎯 Focus", summary.get("focus_grade", "—"), summary.get("focus_score", 0))
        group_bar_chart("Focus", summary.get("focus_score", 0))
    with col2:
        grade_badge("🧩 Structure", summary.get("structure_grade", "—"), summary.get("structure_score", 0))
        group_bar_chart("Structure", summary.get("structure_score", 0))
    with col3:
        grade_badge("💬 Emotion", summary.get("emotion_grade", "—"), summary.get("emotion_score", 0))
        group_bar_chart("Emotion", summary.get("emotion_score", 0))

    st.markdown("#### Confidence")
    conf = summary.get("confidence", {})
    conf_label = conf.get("label", "Low")
    st.write(f"Confidence level: **{conf_label}** — longer texts (120–200+ words) increase accuracy.")

    st.divider()

    tab1, tab2, tab3, tab4 = st.tabs(["Insights", "Coherence", "Emotions", "AI Coach"])

    with tab1:
        st.subheader("Detailed Insights")
        rows = []
        for k, v in results.items():
            if not isinstance(v, dict) or "z_score" not in v:
                continue
            rows.append({
                "Feature": nice_label(k),
                "Your Value": None if v["value"] is None else round(v["value"], 4),
                "Healthy Mean": None if v["baseline_mean"] is None else round(v["baseline_mean"], 4),
                "Z-Score": round(v["z_score"], 2),
                "Verdict": v["verdict"],
            })
        df = pd.DataFrame(rows).sort_values("Feature")
        st.dataframe(df, use_container_width=True, height=300)
        st.caption("Z-Score = how far your feature differs from the healthy baseline (0 = typical).")

    with tab2:
        st.subheader("Sentence Flow & Coherence")
        sents = _sentences(user_text)
        fig_hm = coherence_heatmap(sents)
        if fig_hm:
            st.pyplot(fig_hm, use_container_width=True)
            st.caption("Brighter areas show stronger connections between sentences.")
        else:
            st.info("Write at least two sentences to visualize coherence.")

    with tab3:
        st.subheader("Emotional Tone & Variation")
        sents = _sentences(user_text)
        fig_sent = sentiment_trajectory(sents)
        if fig_sent:
            st.pyplot(fig_sent, use_container_width=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Top Emotion", feats.get("emotion_top_1", "—"))
        with colB:
            st.metric("Second Emotion", feats.get("emotion_top_2", "—"))
        with colC:
            st.metric("Emotion Mix", f"{feats.get('emotion_entropy', 0):.2f}")
        st.caption("The line shows how emotional tone (positive or negative) shifts throughout your text.")

    with tab4:
        st.subheader("Coach’s Notes")
        fb = coach_feedback(summary, zs)
        st.markdown(f"> {fb}")

    st.divider()
    st.caption("This app is for educational self-reflection only — not a medical assessment.")

