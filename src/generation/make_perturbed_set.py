import pandas as pd
import random
from src.utils.dataset import load_mmlu
from src.utils.perturb import inject_noise, shuffle_choices

random.seed(42)

src = load_mmlu("mmlu_test.csv")             # choices already parsed

# 2-A  Surface-noise version ----------------------------------------
noise = src.copy()
noise["question"] = noise["question"].apply(inject_noise)
noise.to_csv("mmlu_noise.csv", index=False)

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
pd.DataFrame(shuf_rows).to_csv("mmlu_shuffle.csv", index=False)

print("✓ wrote mmlu_noise.csv  and  mmlu_shuffle.csv")
