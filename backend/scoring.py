import pandas as pd
import numpy as np

BASELINE_PATH = "data/feature_dataset.csv"

FEATURE_COLUMNS = [
    "word_count",
    "sentence_count",
    "type_token_ratio",
    "mtld",
    "avg_sentence_length",
    "avg_clauses_per_sentence",
    "simple_sentence_ratio",
    "fragment_rate",
    "repetition_ratio",
    "semantic_coherence",
    "global_coherence_mean",
    "global_coherence_sd",
    "entity_consistency_score",
    "temporal_stability_score",
    "temporal_markers_per_100w",
    "fillers_per_100w",
    "avg_sentiment",
    "sentiment_variance",
    "sentiment_range",
    "emotion_volatility",
    "emotion_entropy",
    "pronoun_count",
    "first_person_pronouns",
]

COGNITIVE_WEIGHTS = {
    "semantic_coherence":       0.18,
    "global_coherence_mean":    0.14,
    "global_coherence_sd":      0.06,
    "entity_consistency_score": 0.08,
    "temporal_stability_score": 0.06,
    "type_token_ratio":         0.07,
    "mtld":                     0.07,
    "avg_clauses_per_sentence": 0.08,
    "simple_sentence_ratio":    0.06,
    "fragment_rate":            0.06,
    "repetition_ratio":         0.06,
    "fillers_per_100w":         0.04,
    "temporal_markers_per_100w":0.04,
}

EMOTION_WEIGHTS = {
    "emotion_volatility":  0.34,
    "emotion_entropy":     0.28,
    "sentiment_variance":  0.18,
    "sentiment_range":     0.12,
    "avg_sentiment":       0.08,
}

def _robust_z(user_value: float, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty or user_value is None or not np.isfinite(user_value):
        return 0.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    med = float(s.median())
    mad = float((s - med).abs().median())
    robust_std = 1.4826 * mad if mad > 0 else 0.0
    denom = robust_std if robust_std > 1e-6 else (std if std > 1e-6 else 1.0)
    z = (float(user_value) - mean) / denom
    return float(np.clip(z, -3.0, 3.0))

def _verdict_from_z(z: float, confidence: str) -> str:
    az = abs(z)
    if confidence == "Low" and az > 1.0:
        return "Preliminary signal"
    if az <= 1.0:
        return "Typical"
    if az <= 2.0:
        return "Mild deviation"
    if az <= 3.0:
        return "Moderate deviation"
    return "Strong deviation"

def _composite_from_abs_z(abs_z_map: dict, weights: dict) -> float:
    total = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if k in abs_z_map and np.isfinite(abs_z_map[k]):
            total += w * abs_z_map[k]
            wsum += w
    if wsum == 0:
        return 100.0
    mean_abs_z = total / wsum
    score = 100.0 * float(np.exp(-mean_abs_z / 2.0))
    return float(np.clip(score, 0.0, 100.0))

def _confidence_from_tokens(token_count: int) -> dict:
    if token_count is None or token_count < 50:
        return {"label": "Low", "value": 0.4}
    if token_count < 120:
        return {"label": "Medium", "value": 0.7}
    return {"label": "High", "value": 1.0}

def compare_to_baseline(user_features: dict) -> dict:
    df = pd.read_csv(BASELINE_PATH)
    healthy_df = df[df["label"] == "healthy"] if "label" in df.columns else df.copy()
    if healthy_df.empty:
        healthy_df = df.copy()

    token_count = user_features.get("token_count", None)
    confidence = _confidence_from_tokens(token_count)

    results = {}
    abs_z_for_composites = {}

    for feature in FEATURE_COLUMNS:
        if feature not in user_features or feature not in healthy_df.columns:
            continue
        series = healthy_df[feature]
        user_value = user_features[feature]

        if user_value is None or not np.isfinite(user_value):
            results[feature] = {
                "value": None,
                "baseline_mean": None,
                "z_score": 0.0,
                "verdict": "N/A",
            }
            continue

        if pd.api.types.is_numeric_dtype(series):
            z = _robust_z(user_value, series)
            verdict = _verdict_from_z(z, confidence["label"])
            results[feature] = {
                "value": float(user_value),
                "baseline_mean": float(pd.to_numeric(series, errors="coerce").dropna().mean())
                    if not series.empty else None,
                "z_score": float(z),
                "verdict": verdict,
            }
            abs_z_for_composites[feature] = abs(float(z))
        else:
            results[feature] = {
                "value": user_value,
                "baseline_mean": None,
                "z_score": 0.0,
                "verdict": "N/A",
            }

    cognitive_abs_z = {k: v for k, v in abs_z_for_composites.items() if k in COGNITIVE_WEIGHTS}
    emotional_abs_z = {k: v for k, v in abs_z_for_composites.items() if k in EMOTION_WEIGHTS}

    cognitive_fluency = _composite_from_abs_z(cognitive_abs_z, COGNITIVE_WEIGHTS)
    emotional_clarity = _composite_from_abs_z(emotional_abs_z, EMOTION_WEIGHTS)

    flags = []
    for k, v in results.items():
        z = v.get("z_score", 0.0)
        if not np.isfinite(z):
            continue
        az = abs(z)
        if az > 3.0:
            level = "🔴 Strong"
        elif az > 2.0:
            level = "🚩 Moderate"
        elif az > 1.5:
            level = "⚠️ Minor"
        else:
            continue
        if confidence["label"] == "Low":
            level = "⚠️ Preliminary"
        flags.append({"feature": k, "level": level, "z": float(z)})

    summary = {
        "cognitive_fluency": round(float(cognitive_fluency), 1),
        "emotional_clarity": round(float(emotional_clarity), 1),
        "confidence": confidence,
        "flags": sorted(flags, key=lambda x: {"🔴 Strong":3, "🚩 Moderate":2, "⚠️ Minor":1, "⚠️ Preliminary":0}.get(x["level"], 0), reverse=True),
    }

    return {**results, "_summary": summary}

