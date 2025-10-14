import re
import numpy as np
import spacy
import nltk
from collections import Counter
from lexicalrichness import LexicalRichness
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

nlp = spacy.load("en_core_web_sm")
nltk.download("punkt")

sent_embed = SentenceTransformer("all-MiniLM-L6-v2")

sentiment_analyzer = SentimentIntensityAnalyzer()

_emotion_model_id = "SamLowe/roberta-base-go_emotions"
_emotion_tokenizer = AutoTokenizer.from_pretrained(_emotion_model_id)
_emotion_model = AutoModelForSequenceClassification.from_pretrained(_emotion_model_id)
_emotion_pipe = pipeline(
    "text-classification",
    model=_emotion_model,
    tokenizer=_emotion_tokenizer,
    return_all_scores=True,
    truncation=True,
)

_FILLERS = {"like", "you know", "i mean", "kind of", "sort of", "and then"}
_TEMPORAL_MARKERS = {
    "yesterday","today","tomorrow","last night","last week","last month","last year",
    "this morning","this evening","this week","this month","this year",
    "next week","next month","next year","ago","earlier","later","now","then"
}

_MAX_SENTENCES = 120  # safety cap for very long inputs

def _sentence_list(doc):
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    return sents[:_MAX_SENTENCES]

def _tokens(doc):
    return [t.lemma_.lower() for t in doc if t.is_alpha]

def _lexical_diversity(tokens):
    wc = len(tokens)
    ttr = (len(set(tokens)) / wc) if wc else 0.0
    mtld = LexicalRichness(" ".join(tokens)).mtld() if wc >= 50 else 0.0
    return ttr, mtld

def _syntax_metrics(doc, sentences, tokens):
    wc = len(tokens)
    sc = len(sentences)
    avg_sentence_length = (wc / sc) if sc else 0.0
    clause_count = sum(1 for t in doc if t.dep_ == "mark")
    avg_clauses_per_sentence = (clause_count / sc) if sc else 0.0
    simple_sentence_ratio = (sum(1 for s in doc.sents if not any(t.dep_ == "mark" for t in s)) / sc) if sc else 0.0
    fragment_rate = (sum(1 for s in doc.sents if (len([t for t in s if t.is_alpha]) < 5) or not any(t.pos_ == "VERB" for t in s)) / sc) if sc else 0.0
    return avg_sentence_length, avg_clauses_per_sentence, simple_sentence_ratio, fragment_rate

def _repetition(tokens):
    wc = len(tokens)
    return (1 - (len(set(tokens)) / wc)) if wc else 0.0

def _semantic_coherence(sentences):
    if len(sentences) < 2:
        return 0.0, 0.0, 0.0
    embs = sent_embed.encode(sentences, convert_to_numpy=True, normalize_embeddings=True)
    adj = [util.cos_sim(embs[i], embs[i + 1]).item() for i in range(len(embs) - 1)]
    doc_centroid = np.mean(embs, axis=0)
    glob = [util.cos_sim(x, doc_centroid).item() for x in embs]
    return float(np.mean(adj)), float(np.mean(glob)), float(np.std(glob))

def _entities_and_temporal(doc, sentences):
    sc = len(sentences)
    if sc == 0:
        return 0.0, 0.0, 0.0

    by_sent_ents = []
    for s in doc.sents:
        ents = {f"{e.label_}:{e.text.lower()}" for e in s.ents}
        by_sent_ents.append(ents)
    continuity = []
    for i in range(1, len(by_sent_ents)):
        prev, cur = by_sent_ents[i - 1], by_sent_ents[i]
        if not prev and not cur:
            continuity.append(1.0)
        else:
            inter = len(prev & cur)
            union = len(prev | cur) if (prev or cur) else 1
            continuity.append(inter / union)
    entity_consistency_score = float(np.mean(continuity)) if continuity else 0.0

    temporal_hits = []
    for s in sentences:
        s_low = s.lower()
        hits = {m for m in _TEMPORAL_MARKERS if m in s_low}
        temporal_hits.append(hits)
    switches = sum(1 for i in range(1, sc) if temporal_hits[i] != temporal_hits[i - 1])
    temporal_stability_score = 1.0 - (switches / (sc - 1)) if sc > 1 else 1.0
    temporal_density = float(sum(len(h) for h in temporal_hits) / sc)
    return entity_consistency_score, temporal_stability_score, temporal_density

def _filler_rate(text, sc):
    t = text.lower()
    count = 0
    for f in _FILLERS:
        count += len(re.findall(rf"\b{re.escape(f)}\b", t))
    return (count / sc) if sc else 0.0

def _emotion_metrics(sentences):
    if not sentences:
        return {"emotion_top_1": "", "emotion_top_2": "", "emotion_volatility": 0.0, "emotion_entropy": 0.0}
    per_sent_probs = []
    labels_ref = None
    for s in sentences:
        out = _emotion_pipe(s)  # list of list of {label, score}
        scores = out[0]
        labels = [d["label"] for d in scores]
        probs = np.array([d["score"] for d in scores], dtype=float)
        probs = probs / (probs.sum() + 1e-8)
        if labels_ref is None:
            labels_ref = labels
        per_sent_probs.append(probs)
    M = np.stack(per_sent_probs, axis=0)
    mean_probs = M.mean(axis=0)
    top_idx = np.argsort(mean_probs)[::-1]
    emotion_top_1 = labels_ref[top_idx[0]] if len(top_idx) > 0 else ""
    emotion_top_2 = labels_ref[top_idx[1]] if len(top_idx) > 1 else ""
    emotion_volatility = float(np.mean(np.std(M, axis=0)))
    entropy = -np.sum(mean_probs * np.log(mean_probs + 1e-12))
    max_entropy = np.log(len(mean_probs) + 1e-12)
    emotion_entropy = float(entropy / (max_entropy + 1e-12))
    return {
        "emotion_top_1": emotion_top_1,
        "emotion_top_2": emotion_top_2,
        "emotion_volatility": emotion_volatility,
        "emotion_entropy": emotion_entropy
    }

def extract_user_features(text: str) -> dict:
    doc = nlp(text)
    sentences = _sentence_list(doc)
    tokens = _tokens(doc)

    word_count = len(tokens)
    sentence_count = len(sentences)

    ttr, mtld = _lexical_diversity(tokens)
    avg_sentence_length, avg_clauses_per_sentence, simple_sentence_ratio, fragment_rate = _syntax_metrics(doc, sentences, tokens)
    repetition_ratio = _repetition(tokens)

    adj_coh, glob_coh_mean, glob_coh_sd = _semantic_coherence(sentences)
    entity_consistency_score, temporal_stability_score, temporal_density = _entities_and_temporal(doc, sentences)
    filler_rate = _filler_rate(text, sentence_count)

    sentiments = [sentiment_analyzer.polarity_scores(s)["compound"] for s in sentences] if sentences else []
    avg_sentiment = float(np.mean(sentiments)) if sentiments else 0.0
    sentiment_variance = float(np.std(sentiments)) if sentiments else 0.0
    sentiment_range = float(np.ptp(sentiments)) if sentiments else 0.0

    emo = _emotion_metrics(sentences)

    pos_counts = Counter([t.pos_ for t in doc])
    pronoun_count = pos_counts.get("PRON", 0)
    first_person_pronouns = sum(1 for t in doc if t.lower_ in {"i", "me", "my", "mine"})

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "type_token_ratio": ttr,
        "mtld": mtld,
        "avg_sentence_length": avg_sentence_length,
        "avg_clauses_per_sentence": avg_clauses_per_sentence,
        "simple_sentence_ratio": simple_sentence_ratio,
        "fragment_rate": fragment_rate,
        "repetition_ratio": repetition_ratio,
        "semantic_coherence": adj_coh,
        "global_coherence_mean": glob_coh_mean,
        "global_coherence_sd": glob_coh_sd,
        "entity_consistency_score": entity_consistency_score,
        "temporal_stability_score": temporal_stability_score,
        "temporal_marker_density": temporal_density,
        "filler_rate": filler_rate,
        "avg_sentiment": avg_sentiment,
        "sentiment_variance": sentiment_variance,
        "sentiment_range": sentiment_range,
        "emotion_top_1": emo["emotion_top_1"],
        "emotion_top_2": emo["emotion_top_2"],
        "emotion_volatility": emo["emotion_volatility"],
        "emotion_entropy": emo["emotion_entropy"],
        "pronoun_count": pronoun_count,
        "first_person_pronouns": first_person_pronouns
    }

