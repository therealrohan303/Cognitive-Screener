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

# import backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.analyze_user_input import extract_user_features
from backend.scoring import compare_to_baseline

# --- UI setup ---
st.set_page_config(page_title="🧠 Cognitive Wellness Screener", layout="wide")

# --- Optional model objects for visuals (separate from analyzers for plotting only) ---
_nlp = spacy.load("en_core_web_sm")
_sent_embed = SentenceTransformer("all-MiniLM-L6-v2")
_vader = SentimentIntensityAnalyzer()

# --- OpenAI client (optional) ---
def _openai_client():
    try:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            return None
        return OpenAI()
    except Exception:
        return None

client = _openai_client()

# --- Helpers ---
FRIENDLY_NAMES = {
    "type_token_ratio": "Lexical Richness (TTR)",
    "mtld": "Lexical Diversity (MTLD)",
    "avg_sentence_length": "Avg Sentence Length",
    "avg_clauses_per_sentence": "Clause Density",
    "simple_sentence_ratio": "Simple Sentence Ratio",
    "fragment_rate": "Fragment Rate",
    "repetition_ratio": "Repetition Ratio",
    "semantic_coherence": "Adjacent Coherence",
    "global_coherence_mean": "Global Coherence (Mean)",
    "global_coherence_sd": "Global Coherence (SD)",
    "entity_consistency_score": "Entity Consistency",
    "temporal_stability_score": "Temporal Stability",
    "temporal_marker_density": "Temporal Marker Density",
    "filler_rate": "Filler Rate",
    "avg_sentiment": "Avg Valence",
    "sentiment_variance": "Valence Variance",
    "sentiment_range": "Valence Range",
    "emotion_entropy": "Emotion Entropy",
    "emotion_volatility": "Emotion Volatility",
    "word_count": "Word Count",
    "sentence_count": "Sentence Count",
    "pronoun_count": "Pronouns",
    "first_person_pronouns": "1st-Person Pronouns",
}

FOCUS_FEATURES = [
    "semantic_coherence",
    "global_coherence_mean",
    "global_coherence_sd",
    "entity_consistency_score",
    "temporal_stability_score",
    "mtld",
    "type_token_ratio",
    "avg_clauses_per_sentence",
    "simple_sentence_ratio",
    "fragment_rate",
    "repetition_ratio",
    "emotion_entropy",
    "emotion_volatility",
    "avg_sentiment",
    "sentiment_variance",
]

def nice_label(key: str) -> str:
    return FRIENDLY_NAMES.get(key, key.replace("_", " ").title())

def zscore_map(results: dict) -> dict:
    return {k: v["z_score"] for k, v in results.items() if k != "_summary" and isinstance(v, dict) and "z_score" in v}

def _sentences(text: str):
    return [s.text.strip() for s in _nlp(text).sents if s.text.strip()]

def coherence_heatmap(sentences):
    if len(sentences) < 2:
        return None
    embs = _sent_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    sim = util.cos_sim(embs, embs).cpu().numpy()
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    sns.heatmap(sim, vmin=-1, vmax=1, cmap="vlag", ax=ax, cbar=True)
    ax.set_title("Coherence Heatmap (Sentence↔Sentence)")
    ax.set_xlabel("Sentence")
    ax.set_ylabel("Sentence")
    return fig

def sentiment_trajectory(sentences):
    if not sentences:
        return None
    vals = [_vader.polarity_scores(s)["compound"] for s in sentences]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(range(1, len(vals) + 1), vals, marker="o")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Valence Trajectory Across Sentences")
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Compound Valence")
    ax.grid(True, linewidth=0.3)
    return fig

def coach_feedback(zs: dict, summary: dict) -> str:
    prompt = (
        "You are a concise, kind cognitive writing coach. Use the data to give clear, practical guidance.\n\n"
        f"Composite scores:\n- Cognitive Fluency: {summary.get('cognitive_fluency')}\n"
        f"- Emotional Clarity: {summary.get('emotional_clarity')}\n\n"
        f"Top deviations (z-scores): { {k: round(v,2) for k,v in sorted(zs.items(), key=lambda x: -abs(x[1]))[:8]} }\n\n"
        "Write 4 short bullets:\n"
        "1) What stands out in topic flow/coherence.\n"
        "2) What stands out in structure/complexity.\n"
        "3) What stands out in emotional expression.\n"
        "4) Two concrete suggestions to improve next time.\n"
        "Avoid clinical terms; be supportive and specific."
    )
    if not client:
        return "AI coach unavailable (API key not set). Focus on any large positive/negative bars in the chart and try the two tips: (a) outline your point in 3 steps before writing; (b) add one sentence that names a feeling and why."
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a warm, precise writing coach focused on coherence, clarity, and emotional expression."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=300,
            temperature=0.5,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "AI coach temporarily unavailable. Try again later. Meanwhile: (1) Re-read and merge related sentences to tighten flow; (2) Add a clarifying sentence naming a feeling and a cause."

# --- App ---
st.title("🧠 Cognitive Wellness Screener")

st.markdown(
    "Analyze your writing for **coherence**, **linguistic complexity**, and **emotional clarity**. "
    "Results are educational and non-diagnostic."
)

user_text = st.text_area("✍️ Write or paste your journal/story entry:", height=220, placeholder="Write a paragraph or two…")

col_go, col_clear = st.columns([1, 1])
with col_go:
    run = st.button("Analyze", use_container_width=True)
with col_clear:
    clear = st.button("Clear", use_container_width=True)
if clear:
    st.experimental_rerun()

if run and user_text.strip():
    with st.spinner("Analyzing…"):
        feats = extract_user_features(user_text)
        results = compare_to_baseline(feats)
        summary = results.get("_summary", {})
        zs = zscore_map(results)

    # Overview
    st.subheader("Overview")
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        st.metric("Cognitive Fluency", f"{summary.get('cognitive_fluency', 0):.1f}/100")
        st.progress(min(max(summary.get('cognitive_fluency', 0)/100.0, 0.0), 1.0))
    with c2:
        st.metric("Emotional Clarity", f"{summary.get('emotional_clarity', 0):.1f}/100")
        st.progress(min(max(summary.get('emotional_clarity', 0)/100.0, 0.0), 1.0))
    with c3:
        flags = summary.get("flags", [])
        if flags:
            st.write("**Notable Deviations**")
            for f in flags[:6]:
                st.write(f"{f['level']} — {nice_label(f['feature'])} (z={f['z']:.2f})")
        else:
            st.success("No notable deviations vs. healthy baseline.")

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Details", "Coherence", "Emotions", "Coach’s Notes"])

    # Details
    with tab1:
        rows = []
        for k, v in results.items():
            if k == "_summary" or not isinstance(v, dict) or "z_score" not in v:
                continue
            rows.append({
                "Feature": nice_label(k),
                "Your Value": None if v["value"] is None else round(v["value"], 4),
                "Healthy Mean": None if v["baseline_mean"] is None else round(v["baseline_mean"], 4),
                "Z-Score": round(v["z_score"], 3),
                "Verdict": v["verdict"],
                "Key": k,
            })
        df = pd.DataFrame(rows).sort_values("Feature")
        st.dataframe(df.drop(columns=["Key"]), use_container_width=True, height=280)

        focus = {k: zs[k] for k in FOCUS_FEATURES if k in zs}
        if focus:
            st.markdown("**Key Metric Deviations**")
            fig, ax = plt.subplots(figsize=(11, 4))
            order = sorted(focus.items(), key=lambda x: -abs(x[1]))
            sns.barplot(x=[nice_label(k) for k, _ in order], y=[v for _, v in order], ax=ax, palette="coolwarm")
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("Z-Score vs. Healthy")
            ax.set_xlabel("")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig, use_container_width=True)

    # Coherence
    with tab2:
        sents = _sentences(user_text)
        fig_hm = coherence_heatmap(sents)
        if fig_hm:
            st.pyplot(fig_hm, use_container_width=True)
        else:
            st.info("Provide at least two sentences to view the coherence heatmap.")
        st.caption("Warmer colors indicate higher similarity; diagonal is self-similarity.")

    # Emotions
    with tab3:
        fig_val = sentiment_trajectory(sents if 'sents' in locals() else _sentences(user_text))
        if fig_val:
            st.pyplot(fig_val, use_container_width=True)
        emo_top1 = feats.get("emotion_top_1", "")
        emo_top2 = feats.get("emotion_top_2", "")
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Top Emotion", emo_top1 or "—")
        with colB:
            st.metric("Second Emotion", emo_top2 or "—")
        with colC:
            st.metric("Emotion Entropy", f"{feats.get('emotion_entropy', 0):.2f}")

    # Coach’s Notes
    with tab4:
        st.write("**Personalized Guidance**")
        fb = coach_feedback(zs, summary)
        st.markdown(f"> {fb}")

    st.divider()
    st.caption("This tool is for self-reflection only and does not diagnose or treat any condition.")

