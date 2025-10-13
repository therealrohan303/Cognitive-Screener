import pandas as pd
import numpy as np

# Path to your precomputed baseline dataset
BASELINE_PATH = 'data/feature_dataset.csv'

# Features to evaluate
FEATURE_COLUMNS = [
    'word_count',
    'sentence_count',
    'type_token_ratio',
    'mtld',
    'avg_sentence_length',
    'pronoun_count',
    'first_person_pronouns',
    'noun_count',
    'verb_count',
    'adjective_count',
    'indefinite_reference_count',
    'avg_clauses_per_sentence',
    'parse_tree_depth',
    'semantic_coherence',
    'repetition_ratio',
    'avg_sentiment',
    'sentiment_variance',
    'sentiment_range',
]

def compare_to_baseline(user_features: dict) -> dict:
    # Load and filter the baseline dataset
    df = pd.read_csv(BASELINE_PATH)
    healthy_df = df[df['label'] == 'healthy']

    results = {}
    for feature in FEATURE_COLUMNS:
        if feature not in user_features:
            continue

        baseline_mean = healthy_df[feature].mean()
        baseline_std = healthy_df[feature].std()
        user_value = user_features[feature]

        # Handle cases where std is 0 to avoid division by zero
        if baseline_std == 0:
            z_score = 0
        else:
            z_score = (user_value - baseline_mean) / baseline_std

        # Verdict based on z-score
        if abs(z_score) <= 1:
            verdict = "Typical"
        elif z_score < -1:
            verdict = "Below Average"
        else:
            verdict = "Above Average"

        results[feature] = {
            'value': user_value,
            'baseline_mean': baseline_mean,
            'z_score': z_score,
            'verdict': verdict
        }

    return results

