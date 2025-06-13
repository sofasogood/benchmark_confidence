"""Make dummy predictions and save them to a CSV file."""
import numpy as np
import pandas as pd
import os
from src.utils.dataset import load_mmlu         

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# Load the sampled dataset
df = load_mmlu("data/raw/mmlu_test_sampled_0.02.csv")      # choices parsed into lists
print(f"Loaded sampled dataset with {len(df)} questions")

rng = np.random.default_rng(seed=0)
rand = [rng.integers(0, len(opts)) for opts in df["choices"]]

df_out = pd.DataFrame({
    "is_correct": pd.Series(rand == df["answer"]).astype(int)
})
output_path = "data/predictions/dummy_preds_sampled.csv"
df_out.to_csv(output_path, index=False)
print(f"Saved dummy predictions to {output_path}")
print("Dummy accuracy:", df_out['is_correct'].mean()*100, "%")
