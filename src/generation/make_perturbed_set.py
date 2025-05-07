import pandas as pd
import random
import os
from src.utils.dataset import load_mmlu
from src.utils.perturb import inject_noise, shuffle_choices

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/perturbed", exist_ok=True)

random.seed(42)

# Load sampled dataset
src = load_mmlu("data/raw/mmlu_test_sampled_0.02.csv")
print(f"Loaded sampled dataset with {len(src)} questions")

# 2-A  Surface-noise version ----------------------------------------
noise = src.copy()
noise["question"] = noise["question"].apply(inject_noise)
noise_path = "data/perturbed/mmlu_noise.csv"
noise.to_csv(noise_path, index=False)

# 2-B  Distractor-shuffle version -----------------------------------
shuf_rows = []
for _, row in src.iterrows():
    new_choices, new_ans = shuffle_choices(row["choices"], row["answer"])
    shuf_rows.append({
        "question": row["question"],
        "subject" : row["subject"],
        "choices" : new_choices,
        "answer"  : new_ans,
    })
shuffle_path = "data/perturbed/mmlu_shuffle.csv"
pd.DataFrame(shuf_rows).to_csv(shuffle_path, index=False)

print(f"✓ Generated perturbed versions of sampled dataset:")
print(f"  - Noise version: {noise_path}")
print(f"  - Shuffle version: {shuffle_path}")
