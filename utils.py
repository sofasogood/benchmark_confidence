# utils.py  (helper you can import anywhere)
import ast
import pandas as pd

def load_mmlu(path: str = "mmlu_test.csv") -> pd.DataFrame:
    """Load and parse the MMLU dataset from CSV, converting choices from string to list."""
    df = pd.read_csv(path)
    df["choices"] = df["choices"].apply(ast.literal_eval)   # "[0,4,2,6]" → [0,4,2,6]
    return df
