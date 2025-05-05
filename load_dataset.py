"""Download and save the MMLU test dataset to a CSV file."""
# save as download_mmlu.py
from datasets import load_dataset  # type: ignore
import pandas as pd  # type: ignore

ds = load_dataset("cais/mmlu", "all")["test"]    # 14 079 items
df = pd.DataFrame({
    "question":  ds["question"],
    "subject":   ds["subject"],
    "choices":   ds["choices"],
    "answer":    ds["answer"],
})
print(df.head()["choices"])
#df.to_csv("mmlu_test.csv", index=False)
print("Saved mmlu_test.csv with", len(df), "rows")
