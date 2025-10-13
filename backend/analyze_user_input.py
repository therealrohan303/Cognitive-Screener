import numpy as np
import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from collections import Counter

# Load models and download resources
nlp = spacy.load("en_core_web_sm")
nltk.download('punkt')
sent_model = SentenceTransformer('all-MiniLM-L6-v2')
sentiment_analyzer = SentimentIntensityAnalyzer()

def extract_user_features(text: str) -> dict:
    doc = nlp(text)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha]
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    # Lexical & Structural Features
    ttr = len(set(tokens)) / len(tokens) if tokens else 0
    word_count = len(tokens)
    sentence_count = len(sentences)
    avg_sentence_length = word_count / sentence_count if sentence_count else 0

    # MTLD (proxy for lexical diversity)
    mtld = len(tokens) / ttr if ttr else 0

    # POS Features
    pos_counts = Counter([token.pos_ for token in doc])
    pronoun_count = pos_counts.get('PRON', 0)
    noun_count = pos_counts.get('NOUN', 0)
    verb_count = pos_counts.get('VERB', 0)
    adj_count = pos_counts.get('ADJ', 0)

    # Indefinite References
    indefinite_refs = sum(1 for token in tokens if token in ['thing', 'stuff', 'something', 'anything', 'everything', 'it', 'that'])

    # Syntactic Complexity
    clause_count = sum(1 for token in doc if token.dep_ == 'mark')
    avg_clauses_per_sentence = clause_count / sentence_count if sentence_count else 0
    parse_tree_depth = max([len([tok for tok in sent]) for sent in doc.sents]) if sentence_count else 0

    # Semantic Coherence
    semantic_coherence = 0
    if len(sentences) > 1:
        embeddings = sent_model.encode(sentences)
        sim_scores = [util.cos_sim(embeddings[i], embeddings[i + 1]).item() for i in range(len(embeddings) - 1)]
        semantic_coherence = np.mean(sim_scores) if sim_scores else 0

    # Sentiment
    sentiments = [sentiment_analyzer.polarity_scores(sent)["compound"] for sent in sentences]
    avg_sentiment = np.mean(sentiments) if sentiments else 0
    sentiment_variance = np.std(sentiments) if sentiments else 0
    sentiment_range = np.ptp(sentiments) if sentiments else 0

    # Repetition
    repetition_ratio = 1 - (len(set(tokens)) / len(tokens)) if tokens else 0

    # First-Person Pronouns
    first_person_count = sum(1 for token in doc if token.lower_ in ['i', 'me', 'my', 'mine'])

    return {
        'word_count': word_count,
        'sentence_count': sentence_count,
        'type_token_ratio': ttr,
        'mtld': mtld,
        'avg_sentence_length': avg_sentence_length,
        'pronoun_count': pronoun_count,
        'first_person_pronouns': first_person_count,
        'noun_count': noun_count,
        'verb_count': verb_count,
        'adjective_count': adj_count,
        'indefinite_reference_count': indefinite_refs,
        'avg_clauses_per_sentence': avg_clauses_per_sentence,
        'parse_tree_depth': parse_tree_depth,
        'semantic_coherence': semantic_coherence,
        'repetition_ratio': repetition_ratio,
        'avg_sentiment': avg_sentiment,
        'sentiment_variance': sentiment_variance,
        'sentiment_range': sentiment_range,
    }

