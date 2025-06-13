"""Download and save the MMLU test dataset to a CSV file."""
# save as download_mmlu.py
from datasets import load_dataset  # type: ignore
import pandas as pd  # type: ignore
import os

# Create data/raw directory if it doesn't exist
os.makedirs("data/raw", exist_ok=True)

ds = load_dataset("cais/mmlu", "all")["test"]    # 14 079 items
df = pd.DataFrame({
    "question":  ds["question"],
    "subject":   ds["subject"],
    "choices":   ds["choices"],
    "answer":    ds["answer"],
})
print(df.head()["choices"])
df.to_csv("data/raw/mmlu_test.csv", index=False)
print("Saved data/raw/mmlu_test.csv with", len(df), "rows")
