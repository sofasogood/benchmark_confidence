"""Constructs.yml file for MMLU dataset."""
import json
from src.utils.dataset import load_mmlu 

subjects = sorted(load_mmlu()["subject"].unique())
json.dump(subjects, open("constructs.yml", "w", encoding="utf-8"), indent=2)

print("Wrote constructs.yml with", len(subjects), "subjects")
