import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import openai
from openai import OpenAI
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.analyze_user_input import extract_user_features
from backend.scoring import compare_to_baseline

# Theme and layout config
st.set_page_config(page_title="🧠 Cognitive Wellness Screener", layout="wide")
st.markdown("""
<style>
body {
    background-color: #f4f4f4;
}
section.main > div {
    background-color: #ffffff;
    border-radius: 12px;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}
</style>
""", unsafe_allow_html=True)

client = OpenAI()

# GPT Feedback Function
def gpt_feedback(z_scores: dict) -> str:
    prompt = f"""
You are a cognitive health assistant. A user has written a journal entry, and these are the z-scores for cognitive-linguistic features compared to healthy controls:

{z_scores}

1. Explain what these scores suggest about their language use.
2. Mention any potential early signs of cognitive changes.
3. Give a warm and helpful suggestion to improve their clarity, emotional expression, or coherence.

Avoid clinical jargon. Make it sound insightful and kind.
"""

    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a friendly and perceptive cognitive health assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=250,
        temperature=0.7
    )

    return response.choices[0].message.content

# Title and Intro
st.title("🧠 Cognitive Wellness Screener")
st.markdown("""
This tool analyzes your journal or story entry for **cognitive-linguistic patterns**—like coherence, word richness, emotional clarity, and syntactic complexity. It provides:
- 📊 Feature breakdown and comparisons
- ⚠️ Flags for concerning patterns
- 🧠 AI-powered wellness feedback
""")

user_text = st.text_area("✍️ Write or paste your journal/story entry below:", height=250)

run_analysis = st.button("🔍 Analyze My Writing")

if run_analysis and user_text.strip():
    with st.spinner("Analyzing your text..."):
        user_features = extract_user_features(user_text)
        results = compare_to_baseline(user_features)

        st.subheader("📊 Detailed Cognitive Feature Analysis")
        feature_table = pd.DataFrame([
            {
                'Feature': feat.replace('_', ' ').title(),
                'Your Value': round(info['value'], 3),
                'Baseline Mean': round(info['baseline_mean'], 3),
                'Z-Score': round(info['z_score'], 2),
                'Verdict': info['verdict']
            } for feat, info in results.items()
        ])
        st.dataframe(feature_table.style.highlight_max(axis=0, color='lightgreen'))

        # Flag system
        flags = [feat for feat, info in results.items() if abs(info['z_score']) > 2]
        if flags:
            st.warning(f"⚠️ Notable deviations detected in: {', '.join([f.replace('_', ' ').title() for f in flags])}")
        else:
            st.success("✅ Your writing aligns well with typical healthy patterns.")

        # Visual: Z-Score Barplot
        st.subheader("📈 Z-Score Comparison")
        z_df = pd.DataFrame.from_dict(results, orient='index')
        z_df['Feature'] = z_df.index.str.replace('_', ' ').str.title()
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.barplot(data=z_df, x='Feature', y='z_score', palette='coolwarm', ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel("Z-Score")
        plt.title("Your Cognitive Feature Scores Compared to Healthy Baseline")
        st.pyplot(fig)

        # Visual: Radar Chart 
        
        # Extract z-scores dictionary for plotting
        z_scores = {k: v['z_score'] for k, v in results.items()}
        st.subheader("🕸️ Feature Radar Chart (Key Metrics)")

        radar_features = ['Type Token Ratio', 'Semantic Coherence', 'Avg Sentiment', 'Sentiment Variance']
        radar_values = [z_scores[feat.lower().replace(' ', '_')] for feat in radar_features]

        # Close the loop
        radar_values += radar_values[:1]  # Repeat the first value at the end
        angles = np.linspace(0, 2 * np.pi, len(radar_features), endpoint=False).tolist()
        angles += angles[:1]

        fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'}, figsize=(6, 6))
        ax2.plot(angles, radar_values, color='blue', linewidth=2)
        ax2.fill(angles, radar_values, color='blue', alpha=0.25)
        ax2.set_thetagrids(np.degrees(angles[:-1]), radar_features)  # ✅ Only use original labels
        ax2.set_title("Z-Score Radar View", size=15)
        ax2.grid(True)
        st.pyplot(fig2)

        # GPT Feedback
        st.subheader("🧠 AI-Powered Cognitive Feedback")
        feedback = gpt_feedback({feat: info['z_score'] for feat, info in results.items()})
        st.markdown(f"""
        <div style='background-color: #1e1e1e; border-left: 5px solid #007acc; padding: 1rem;'>
        <strong>🗣️ Insight:</strong> {feedback}
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.caption("Disclaimer: This tool is for educational and self-awareness purposes only. It does not diagnose or treat any condition.")

