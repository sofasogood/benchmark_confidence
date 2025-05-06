# robust_multi.py
import pandas as pd
import numpy as np
from robustness import rubric          # reuse the function

pairs = [
    ("gpt4_preds.csv",        "Original"),
    ("gpt4_paraphrase.csv",   "Paraphrased"),
    ("gpt4_noise.csv",        "Surface noise"),
    ("gpt4_shuffle.csv",      "Distractor shuffle"),
]

base = np.array(pd.read_csv(pairs[0][0])["is_correct"].values)
acc_base = np.mean(base) * 100

for path, name in pairs[1:]:
    pert = np.array(pd.read_csv(path)["is_correct"].values)
    delta = acc_base - np.mean(pert)*100
    print(f"{name:18}: drop {delta:5.2f} pp  →  score {rubric(delta)}/3")
