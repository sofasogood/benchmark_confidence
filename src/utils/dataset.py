# dataset.py  (helper you can import anywhere)
import ast
import pandas as pd
import random, string, copy
from typing import List, Tuple


def load_mmlu(path: str = "mmlu_test.csv") -> pd.DataFrame:
    """Load and parse the MMLU dataset from CSV, converting choices from string to list."""
    df = pd.read_csv(path)
    df["choices"] = df["choices"].apply(ast.literal_eval)   # "[0,4,2,6]" → [0,4,2,6]
    return df

def stratified_sample(df, frac=0.2, seed: int = 42):
    """Stratified sample of the dataset."""
    return (
        df.groupby("subject", group_keys=False)
          .apply(lambda x: x.sample(frac=frac, random_state=seed))
          .reset_index(drop=True)
    )
