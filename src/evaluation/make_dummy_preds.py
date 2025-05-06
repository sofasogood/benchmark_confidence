"""Make dummy predictions and save them to a CSV file."""
import numpy as np
import pandas as pd
from src.utils.dataset import load_mmlu         

df = load_mmlu("mmlu_test.csv")      # choices parsed into lists

rng   = np.random.default_rng(seed=0)
rand  = [rng.integers(0, len(opts)) for opts in df["choices"]]

df_out = pd.DataFrame({
    "is_correct": pd.Series(rand == df["answer"]).astype(int)
})
df_out.to_csv("dummy_preds.csv", index=False)
print("Dummy accuracy:", df_out['is_correct'].mean()*100, "%")
