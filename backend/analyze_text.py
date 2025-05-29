import os
import pandas as pd
import numpy as np
import spacy
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()

# Directories
BASE_DIR = 'data/simulated_baselines'
LABELS = ['healthy', 'impaired']

def read_samples():
    all_data = []
    for label in LABELS:
        folder = os.path.join(BASE_DIR, label)
        for filename in os.listdir(folder):
            if filename.endswith('.txt'):
                filepath = os.path.join(folder, filename)
                with open(filepath, 'r') as f:
                    text = f.read().strip()
                    all_data.append({
                        'filename': filename,
                        'label': label,
                        'text': text
                    })
    return all_data

def extract_features(entry):
    text = entry['text']
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    # Lexical features
    ttr = len(set(tokens)) / len(tokens) if tokens else 0
    word_count = len(tokens)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0

    # POS counts
    pronouns = len([token for token in doc if token.pos_ == 'PRON'])
    nouns = len([token for token in doc if token.pos_ == 'NOUN'])
    verbs = len([token for token in doc if token.pos_ == 'VERB'])
    adjectives = len([token for token in doc if token.pos_ == 'ADJ'])

    # Semantic coherence
    embeddings = sent_model.encode(sentences)
    sim_scores = [
        util.cos_sim(embeddings[i], embeddings[i + 1]).item()
        for i in range(len(embeddings) - 1)
    ]
    semantic_coherence = np.mean(sim_scores) if sim_scores else 0

    # Sentiment features
    sentiments = [sentiment_analyzer.polarity_scores(sent)["compound"] for sent in sentences]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    sentiment_variance = np.std(sentiments) if sentiments else 0

    return {
        'filename': entry['filename'],
        'label': entry['label'],
        'word_count': word_count,
        'sentence_count': sentence_count,
        'ttr': ttr,
        'avg_sentence_length': avg_sentence_length,
        'pronoun_count': pronouns,
        'noun_count': nouns,
        'verb_count': verbs,
        'adjective_count': adjectives,
        'semantic_coherence': semantic_coherence,
        'avg_sentiment': avg_sentiment,
        'sentiment_variance': sentiment_variance
    }

def main():
    entries = read_samples()
    features = [extract_features(entry) for entry in entries]
    df = pd.DataFrame(features)
    df.to_csv('data/feature_dataset.csv', index=False)
    print("✅ feature_dataset.csv created with", len(df), "entries")

if __name__ == "__main__":
    main()
