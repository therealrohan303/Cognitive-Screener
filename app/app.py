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

# --- Imports from backend
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from backend.analyze_user_input import extract_user_features
from backend.scoring import compare_to_baseline

# --- UI setup
st.set_page_config(page_title="🧠 Cognitive Wellness Screener", layout="wide")

# --- Light-weight local models for visuals
_nlp = spacy.load("en_core_web_sm")
_sent_embed = SentenceTransformer("all-MiniLM-L6-v2")
_vader = SentimentIntensityAnalyzer()

# --- OpenAI client (optional)
def _openai_client():
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None
    try:
        return OpenAI()
    except Exception:
        return None

client = _openai_client()

# --- Friendly labels and feature groups
FRIENDLY_NAMES = {
    "type_token_ratio": "Lexical Richness (TTR)",
    "mtld": "Lexical Diversity (MTLD)",
    "avg_sentence_length": "Avg Sentence Length",
    "avg_clauses_per_sentence": "Clause Density",
    "simple_sentence_ratio": "Simple Sentence Ratio",
    "fragment_rate": "Fragment Rate",
    "repetition_ratio": "Repetition Ratio",
    "semantic_coherence": "Sentence-to-Sentence Flow",
    "global_coherence_mean": "Topic Focus (Mean)",
    "global_coherence_sd": "Topic Stability (Variation)",
    "entity_consistency_score": "Entity Consistency",
    "temporal_stability_score": "Temporal Stability",
    "temporal_markers_per_100w": "Temporal Markers /100 words",
    "fillers_per_100w": "Fillers /100 words",
    "avg_sentiment": "Avg Valence",
    "sentiment_variance": "Valence Variance",
    "sentiment_range": "Valence Range",
    "emotion_entropy": "Emotion Mix (Entropy)",
    "emotion_volatility": "Emotion Volatility",
    "word_count": "Word Count",
    "sentence_count": "Sentence Count",
    "pronoun_count": "Pronouns",
    "first_person_pronouns": "1st-Person Pronouns",
}

# hide legacy names from UI if present
HIDE_KEYS = {"temporal_marker_density", "filler_rate"}

GROUPS = {
    "Coherence": [
        "semantic_coherence",
        "global_coherence_mean",
        "global_coherence_sd",
        "entity_consistency_score",
        "temporal_stability_score",
    ],
    "Complexity": [
        "avg_clauses_per_sentence",
        "simple_sentence_ratio",
        "fragment_rate",
        "repetition_ratio",
    ],
    "Lexical": [
        "mtld",
        "type_token_ratio",
    ],
    "Emotion": [
        "emotion_entropy",
        "emotion_volatility",
        "avg_sentiment",
        "sentiment_variance",
    ],
}

FOCUS_FEATURES = sum(GROUPS.values(), [])  # flattened for strength/opportunity

# --- Small helpers
def nice_label(key: str) -> str:
    return FRIENDLY_NAMES.get(key, key.replace("_", " ").title())

def _sentences(text: str):
    return [s.text.strip() for s in _nlp(text).sents if s.text.strip()]

def zscore_map(results: dict) -> dict:
    out = {}
    for k, v in results.items():
        if k == "_summary" or k in HIDE_KEYS:
            continue
        if isinstance(v, dict) and "z_score" in v:
            out[k] = v["z_score"]
    return out

def confidence_pill(conf: dict):
    label = conf.get("label", "Low")
    val = conf.get("value", 0.4)
    meter = "🟢🟢🟢" if label == "High" else ("🟢🟢⚪️" if label == "Medium" else "🟢⚪️⚪️")
    st.metric("Confidence", f"{label} {meter}", help="Longer entries (≈120–200+ words) generally improve accuracy.")

def key_strength_opportunity(zs: dict):
    focus = {k: v for k, v in zs.items() if k in FOCUS_FEATURES and np.isfinite(v)}
    if not focus:
        return None, None
    best = max(focus.items(), key=lambda kv: kv[1])  # highest z
    worst = min(focus.items(), key=lambda kv: kv[1])  # lowest z
    return best, worst

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

def centroid_strip(sentences):
    if len(sentences) < 2:
        return None
    embs = _sent_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    centroid = np.mean(embs, axis=0)
    sims = [util.cos_sim(e, centroid).item() for e in embs]
    fig, ax = plt.subplots(figsize=(7, 1.6))
    ax.bar(range(1, len(sims) + 1), sims)
    ax.set_ylim(0, 1)
    ax.set_title("Topic Focus per Sentence (vs. document)")
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Sim.")
    return fig

def sentiment_trajectory(sentences):
    if not sentences:
        return None, None
    vals = [_vader.polarity_scores(s)["compound"] for s in sentences]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.plot(range(1, len(vals) + 1), vals, marker="o")
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_title("Valence Trajectory Across Sentences")
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Compound Valence")
    ax.grid(True, linewidth=0.3)
    return fig, vals

def emotion_ribbon(sentences):
    if not sentences:
        return None
    pos, neu, neg = [], [], []
    for s in sentences:
        scores = _vader.polarity_scores(s)
        pos.append(scores["pos"])
        neu.append(scores["neu"])
        neg.append(scores["neg"])
    x = np.arange(1, len(sentences) + 1)
    pos = np.array(pos); neu = np.array(neu); neg = np.array(neg)
    pos_cum = pos
    neu_cum = pos + neu
    fig, ax = plt.subplots(figsize=(7, 2.8))
    ax.fill_between(x, 0, pos_cum, alpha=0.5, label="Positive")
    ax.fill_between(x, pos_cum, neu_cum, alpha=0.5, label="Neutral")
    ax.fill_between(x, neu_cum, 1.0, alpha=0.5, label="Negative")
    ax.set_ylim(0, 1)
    ax.set_title("Emotion Mix Across Sentences (VADER)")
    ax.set_xlabel("Sentence Index")
    ax.set_ylabel("Proportion")
    ax.legend(ncol=3, frameon=False)
    return fig

def pre_gpt_summary(results: dict, summary: dict) -> str:
    flags = summary.get("flags", [])
    zs = zscore_map(results)
    best, worst = key_strength_opportunity(zs)
    bullets = []
    if best:
        k, v = best
        bullets.append(f"**Strength** — {nice_label(k)} (z={v:+.2f}): a notable positive.")
    if worst:
        k, v = worst
        label = nice_label(k)
        if "Coherence" in label or "Topic" in label or k in {"semantic_coherence","global_coherence_mean","global_coherence_sd"}:
            tip = "Try grouping related ideas into single sentences and use transitions to link them."
        elif k in {"mtld","type_token_ratio","repetition_ratio"}:
            tip = "Vary wording slightly—replace repeated words and add precise nouns/verbs."
        elif k in {"avg_clauses_per_sentence","simple_sentence_ratio","fragment_rate"}:
            tip = "Combine short sentences with conjunctions to add nuance without losing clarity."
        else:
            tip = "Add one sentence that names a feeling and what caused it."
        bullets.append(f"**Opportunity** — {label} (z={v:+.2f}): {tip}")
    conf = summary.get("confidence", {}).get("label", "Low")
    if conf == "Low":
        bullets.append("**Note** — Short entries reduce accuracy; ~120–200+ words improve confidence.")
    if not bullets:
        bullets.append("Balanced profile overall; keep your structure and clarity consistent.")
    return "• " + "\n• ".join(bullets)

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
        return "AI coach unavailable. Focus on the strength/opportunity above; try: (1) outline 3 points and write one paragraph per point, (2) add a sentence that names a feeling and a cause."
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
        return "AI coach temporarily unavailable. Try again later. Meanwhile: (1) merge related sentences to tighten flow; (2) add a sentence naming a feeling and why."

# --- App body
st.title("🧠 Cognitive Wellness Screener")
st.markdown("Analyze your writing for **coherence**, **linguistic complexity**, and **emotional clarity**. Results are educational and non-diagnostic.")

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

    # --- Overview
    st.subheader("Overview")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    with c1:
        cf = float(summary.get('cognitive_fluency', 0.0))
        st.metric("Cognitive Fluency", f"{cf:.1f}/100")
        st.progress(min(max(cf / 100.0, 0.0), 1.0))
    with c2:
        ec = float(summary.get('emotional_clarity', 0.0))
        st.metric("Emotional Clarity", f"{ec:.1f}/100")
        st.progress(min(max(ec / 100.0, 0.0), 1.0))
    with c3:
        confidence_pill(summary.get("confidence", {}))
    with c4:
        best, worst = key_strength_opportunity(zs)
        cols = st.columns(2)
        with cols[0]:
            if best:
                st.success(f"**Key Strength**: {nice_label(best[0])} (z={best[1]:+.2f})")
            else:
                st.info("Balanced profile.")
        with cols[1]:
            if worst:
                st.warning(f"**Key Opportunity**: {nice_label(worst[0])} (z={worst[1]:+.2f})")
            else:
                st.info("No notable opportunities flagged.")

    flags = summary.get("flags", [])
    if flags:
        st.markdown("**Notable Deviations**")
        for f in flags[:6]:
            st.write(f"{f['level']} — {nice_label(f['feature'])} (z={f['z']:+.2f})")
    else:
        st.success("No notable deviations vs. healthy baseline.")

    # --- Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Details", "Coherence", "Emotions", "Coach’s Notes"])

    # Details
    with tab1:
        rows = []
        for k, v in results.items():
            if k == "_summary" or k in HIDE_KEYS or not isinstance(v, dict) or "z_score" not in v:
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
        st.dataframe(df.drop(columns=["Key"]), use_container_width=True, height=300)

        st.markdown("**Grouped Metric Deviations**")
        for group_name, keys in GROUPS.items():
            group_vals = [(k, zs[k]) for k in keys if k in zs and np.isfinite(zs[k])]
            if not group_vals:
                continue
            fig, ax = plt.subplots(figsize=(10, 3))
            order = sorted(group_vals, key=lambda x: -abs(x[1]))
            sns.barplot(x=[nice_label(k) for k, _ in order], y=[v for _, v in order], ax=ax, palette="coolwarm")
            ax.axhline(0, color="black", linestyle="--", linewidth=1)
            ax.set_ylabel("Z-Score vs. Healthy")
            ax.set_xlabel("")
            ax.set_title(group_name)
            plt.xticks(rotation=25, ha="right")
            st.pyplot(fig, use_container_width=True)

    # Coherence
    with tab2:
        sents = _sentences(user_text)
        fig_hm = coherence_heatmap(sents)
        if fig_hm:
            st.pyplot(fig_hm, use_container_width=True)
            fig_strip = centroid_strip(sents)
            if fig_strip:
                st.pyplot(fig_strip, use_container_width=True)
            st.caption("Heatmap: brighter off-diagonal blocks = stronger links between nearby sentences. Strip: higher bars mean each sentence aligns well with the overall topic.")
        else:
            st.info("Provide at least two sentences to view coherence visuals.")

    # Emotions
    with tab3:
        sents = _sentences(user_text)
        fig_val, _vals = sentiment_trajectory(sents)
        if fig_val:
            st.pyplot(fig_val, use_container_width=True)
        fig_ribbon = emotion_ribbon(sents)
        if fig_ribbon:
            st.pyplot(fig_ribbon, use_container_width=True)
        colA, colB, colC = st.columns(3)
        with colA:
            st.metric("Top Emotion", feats.get("emotion_top_1", "") or "—")
        with colB:
            st.metric("Second Emotion", feats.get("emotion_top_2", "") or "—")
        with colC:
            st.metric("Emotion Entropy", f"{feats.get('emotion_entropy', 0):.2f}")

    # Coach’s Notes
    with tab4:
        st.write("**Quick Takeaways (no AI):**")
        st.markdown(pre_gpt_summary(results, summary))
        st.write("")
        st.write("**Coach’s Notes (AI):**")
        fb = coach_feedback(zs, summary)
        st.markdown(f"> {fb}")

    st.divider()
    st.caption("This tool is for self-reflection only and does not diagnose or treat any condition.")

