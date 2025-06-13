import pandas as pd
from src.utils.dataset import load_mmlu, stratified_sample

# Load original dataset
df = load_mmlu('data/raw/mmlu_test.csv')
print(f'Original dataset size: {len(df)}')

# Sample 2%
sampled = stratified_sample(df, frac=0.02, seed=42)
print(f'Sampled dataset size: {len(sampled)}')

# Save sampled dataset
sampled.to_csv('data/raw/mmlu_test_sampled_0.02.csv', index=False)
print('Saved sampled dataset') 