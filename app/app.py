import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import openai
from openai import OpenAI
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.analyze_user_input import extract_user_features
from backend.scoring import compare_to_baseline

client = OpenAI()  # Initializes with your OPENAI_API_KEY from env var

def gpt_feedback(z_scores: dict) -> str:
    prompt = f"""You are a cognitive health assistant. A user has written a journal entry and here are their cognitive-linguistic z-scores compared to healthy writing:

{z_scores}

Please generate a 2–3 sentence gentle and insightful feedback summary. Include a suggestion for improvement if needed, but do not alarm the user.
"""

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a friendly cognitive health assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=150,
        temperature=0.7
    )

    return response.choices[0].message.content


st.set_page_config(page_title="🧠 Cognitive Wellness Screener", layout="wide")

st.title("🧠 Cognitive Wellness Screener")
st.markdown("""
This tool analyzes your writing for **cognitive-linguistic patterns** — including coherence, lexical richness, syntactic complexity, and emotional clarity.
It compares your text to healthy writing patterns and gives personalized feedback using AI.
""")

user_text = st.text_area("✍️ Write or paste your journal/story entry below:", height=250)

run_analysis = st.button("Analyze")

if run_analysis and user_text.strip():
    with st.spinner("Analyzing your text..."):
        user_features = extract_user_features(user_text)
        results = compare_to_baseline(user_features)

        # Display raw feature values and verdicts
        st.subheader("📊 Cognitive Feature Analysis")
        feature_table = pd.DataFrame([
            {
                'Feature': feat,
                'Your Value': f"{info['value']:.3f}",
                'Baseline Mean': f"{info['baseline_mean']:.3f}",
                'Z-Score': f"{info['z_score']:.2f}",
                'Verdict': info['verdict']
            } for feat, info in results.items()
        ])
        st.dataframe(feature_table)

        # Visual: Bar plot of z-scores
        st.subheader("📈 Visual Comparison to Healthy Baselines")
        z_scores = {k: v['z_score'] for k, v in results.items()}
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.barplot(x=list(z_scores.keys()), y=list(z_scores.values()), palette='coolwarm', ax=ax)
        ax.axhline(0, color='black', linestyle='--')
        plt.xticks(rotation=45)
        plt.ylabel("Z-Score")
        st.pyplot(fig)
        st.subheader("🧠 AI Feedback Summary")
        feedback = gpt_feedback(z_scores)
        st.markdown(f"🗣️ *{feedback}*")




