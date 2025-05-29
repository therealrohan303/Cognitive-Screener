from backend.analyze_user_input import extract_user_features
from backend.scoring import compare_to_baseline

sample_text = """Last night, I couldn’t sleep well. I kept thinking about my family trip to Italy. The warm breeze, the sunsets, the pasta—it all came back. I wish I could go back and feel that peace again."""

features = extract_user_features(sample_text)
results = compare_to_baseline(features)

for feature, info in results.items():
    print(f"{feature}: {info['verdict']} (z = {info['z_score']:.2f})")

