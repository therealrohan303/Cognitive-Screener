import spacy
import numpy as np
from nltk.tokenize import sent_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util

# Load models
nlp = spacy.load("en_core_web_sm")
sentiment_analyzer = SentimentIntensityAnalyzer()
sent_model = SentenceTransformer('all-MiniLM-L6-v2')

def extract_user_features(text):
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if token.is_alpha]
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    # Lexical + syntactic
    ttr = len(set(tokens)) / len(tokens) if tokens else 0
    word_count = len(tokens)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0

    pronouns = len([token for token in doc if token.pos_ == 'PRON'])
    nouns = len([token for token in doc if token.pos_ == 'NOUN'])
    verbs = len([token for token in doc if token.pos_ == 'VERB'])
    adjectives = len([token for token in doc if token.pos_ == 'ADJ'])

    # Semantic coherence
    if len(sentences) > 1:
        embeddings = sent_model.encode(sentences)
        sim_scores = [
            util.cos_sim(embeddings[i], embeddings[i + 1]).item()
            for i in range(len(embeddings) - 1)
        ]
        semantic_coherence = np.mean(sim_scores)
    else:
        semantic_coherence = 0

    # Sentiment
    sentiments = [sentiment_analyzer.polarity_scores(sent)["compound"] for sent in sentences]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    sentiment_variance = np.std(sentiments) if sentiments else 0

    return {
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

