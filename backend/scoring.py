import pandas as pd
import numpy as np

BASELINE_PATH = "data/feature_dataset.csv"

# All numeric features we can score (from updated analyzers)
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
    "temporal_marker_density",
    "filler_rate",
    "avg_sentiment",
    "sentiment_variance",
    "sentiment_range",
    "emotion_volatility",
    "emotion_entropy",
    "pronoun_count",
    "first_person_pronouns",
]

# Composites are built from a weighted mean absolute z-deviation from healthy baseline.
# Lower deviation => higher score. Weights sum to 1.0 within each composite.
COGNITIVE_WEIGHTS = {
    "semantic_coherence":            0.18,
    "global_coherence_mean":         0.14,
    "global_coherence_sd":           0.06,  # we expect *low* SD; using |z| handles both sides
    "entity_consistency_score":      0.08,
    "temporal_stability_score":      0.06,
    "type_token_ratio":              0.07,
    "mtld":                          0.07,
    "avg_clauses_per_sentence":      0.08,
    "simple_sentence_ratio":         0.06,
    "fragment_rate":                 0.06,
    "repetition_ratio":              0.06,
    "filler_rate":                   0.04,
    "temporal_marker_density":       0.04,
}

EMOTION_WEIGHTS = {
    "emotion_volatility":            0.34,
    "emotion_entropy":               0.28,
    "sentiment_variance":            0.18,
    "sentiment_range":               0.12,
    "avg_sentiment":                 0.08,
}

def _robust_z(user_value: float, series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 0.0
    mean = float(s.mean())
    std = float(s.std(ddof=0))
    med = float(s.median())
    mad = float((s - med).abs().median())
    robust_std = 1.4826 * mad if mad > 0 else 0.0
    denom = robust_std if robust_std > 1e-6 else (std if std > 1e-6 else 1.0)
    return (float(user_value) - mean) / denom

def _verdict_from_z(z: float) -> str:
    az = abs(z)
    if az <= 1.0:
        return "Typical"
    if az <= 2.0:
        return "Mild deviation"
    if az <= 3.0:
        return "Moderate deviation"
    return "Strong deviation"

def _composite_from_abs_z(abs_z_map: dict, weights: dict) -> float:
    # Weighted mean of |z|; map to 0..100 via exp decay (smooth, bounded)
    total = 0.0
    wsum = 0.0
    for k, w in weights.items():
        if k in abs_z_map and np.isfinite(abs_z_map[k]):
            total += w * abs_z_map[k]
            wsum += w
    if wsum == 0:
        return 100.0
    mean_abs_z = total / wsum
    # Decay: score = 100 * exp(-mean_abs_z / 2).  |z|≈0 → ~100; |z|≈2 → ~36.8; |z|≈1 → ~60.7
    score = 100.0 * float(np.exp(-mean_abs_z / 2.0))
    # Clip to 0..100 for safety
    return float(np.clip(score, 0.0, 100.0))

def compare_to_baseline(user_features: dict) -> dict:
    df = pd.read_csv(BASELINE_PATH)
    if "label" in df.columns:
        healthy_df = df[df["label"] == "healthy"]
        if healthy_df.empty:
            healthy_df = df.copy()
    else:
        healthy_df = df.copy()

    results = {}
    abs_z_for_composites = {}

    for feature in FEATURE_COLUMNS:
        if feature not in user_features or feature not in healthy_df.columns:
            continue

        series = healthy_df[feature]
        user_value = user_features[feature]
        if pd.api.types.is_numeric_dtype(series):
            z = _robust_z(user_value, series)
            verdict = _verdict_from_z(z)
            results[feature] = {
                "value": float(user_value) if user_value is not None else None,
                "baseline_mean": float(pd.to_numeric(series, errors="coerce").dropna().mean()) if not series.empty else None,
                "z_score": float(z),
                "verdict": verdict,
            }
            abs_z_for_composites[feature] = abs(float(z))
        else:
            # non-numeric baseline column
            results[feature] = {
                "value": user_value,
                "baseline_mean": None,
                "z_score": 0.0,
                "verdict": "N/A",
            }

    # Build composites from available pieces
    cognitive_abs_z = {k: v for k, v in abs_z_for_composites.items() if k in COGNITIVE_WEIGHTS}
    emotional_abs_z = {k: v for k, v in abs_z_for_composites.items() if k in EMOTION_WEIGHTS}

    cognitive_fluency = _composite_from_abs_z(cognitive_abs_z, COGNITIVE_WEIGHTS)
    emotional_clarity = _composite_from_abs_z(emotional_abs_z, EMOTION_WEIGHTS)

    # Flags (tiered)
    flags = []
    for k, v in results.items():
        z = v.get("z_score", 0.0)
        az = abs(z)
        if az > 3.0:
            flags.append({"feature": k, "level": "🔴 Strong", "z": float(z)})
        elif az > 2.0:
            flags.append({"feature": k, "level": "🚩 Moderate", "z": float(z)})
        elif az > 1.5:
            flags.append({"feature": k, "level": "⚠️ Minor", "z": float(z)})

    summary = {
        "cognitive_fluency": round(float(cognitive_fluency), 1),
        "emotional_clarity": round(float(emotional_clarity), 1),
        "flags": sorted(flags, key=lambda x: {"🔴 Strong":3, "🚩 Moderate":2, "⚠️ Minor":1}.get(x["level"], 0), reverse=True),
    }

    return {**results, "_summary": summary}

