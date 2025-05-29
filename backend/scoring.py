import pandas as pd

# Load only once at module level
df = pd.read_csv('data/feature_dataset.csv')
healthy_df = df[df['label'] == 'healthy']

# Define which features to include in comparison
FEATURES = [
    'word_count', 'sentence_count', 'ttr', 'avg_sentence_length',
    'pronoun_count', 'noun_count', 'verb_count', 'adjective_count',
    'semantic_coherence', 'avg_sentiment', 'sentiment_variance'
]

# Compute healthy baselines (mean and std)
BASELINES = {
    feature: {
        'mean': healthy_df[feature].mean(),
        'std': healthy_df[feature].std()
    } for feature in FEATURES
}

def compare_to_baseline(user_features: dict):
    """
    Compare user feature dictionary to healthy baselines.
    Returns a dict of z-scores and interpretations.
    """
    results = {}
    for feature in FEATURES:
        mean = BASELINES[feature]['mean']
        std = BASELINES[feature]['std']
        user_value = user_features.get(feature, 0)

        # Handle division-by-zero
        if std == 0:
            z_score = 0
        else:
            z_score = (user_value - mean) / std

        interpretation = interpret_z_score(z_score)
        results[feature] = {
            'value': user_value,
            'baseline_mean': mean,
            'z_score': z_score,
            'verdict': interpretation
        }

    return results

def interpret_z_score(z):
    if abs(z) < 0.5:
        return "Normal"
    elif z >= 0.5:
        return "Above expected"
    elif z <= -0.5:
        return "Below expected"

