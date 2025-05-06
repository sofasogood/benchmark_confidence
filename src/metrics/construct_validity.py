# construct_validity.py  (minimal)
from itertools import combinations
import time
import json
import openai
import numpy as np
from src.utils.dataset import load_mmlu

print("Loading constructs and dataset...")
CONSTRUCTS = json.load(open("constructs.yml", encoding="utf-8"))
df = load_mmlu("mmlu_test_sampled_0.02.csv")
print(f"Loaded {len(df)} questions")

def cohens_kappa(y1, y2):
    """Calculate Cohen's Kappa between two raters."""
    labels = sorted(set(y1) | set(y2))
    n = len(labels)
    conf_mat = np.zeros((n, n))
    for a, b in zip(y1, y2):
        i, j = labels.index(a), labels.index(b)
        conf_mat[i, j] += 1
    
    po = np.sum(np.diag(conf_mat)) / np.sum(conf_mat)
    pe = np.sum(np.sum(conf_mat, axis=0) * np.sum(conf_mat, axis=1)) / (np.sum(conf_mat) ** 2)
    return (po - pe) / (1 - pe) if pe != 1 else 0

def llm_tag(q, temp):
    """Assign ONE subject id from this list: {list(CONSTRUCTS)}."""
    prompt = f"""Assign ONE subject id from this list: {list(CONSTRUCTS)}.
    Only return the id.\nQ: {q}"""
    r = openai.chat.completions.create(
        model="gpt-4o-mini", temperature=temp,
        messages=[{"role":"user","content":prompt}],
        max_tokens=5)
    return r.choices[0].message.content.strip()

print("Starting LLM tagging with 3 temperatures... start time:", start_time := time.time())
temps = [0.2,0.4,0.6]               # 3 "raters"
tags  = {t: df["question"].apply(lambda q, temp=t: llm_tag(q,temp)) for t in temps}
print("Finished LLM tagging... elapsed time:", time.time()- start_time)

print("Calculating Cohen's Kappa between raters...")
# pair-wise κ then average
kappas = [cohens_kappa(tags[a], tags[b]) for a,b in combinations(temps,2)]
kappa  = np.mean(kappas)

def rubric(k):
    """Convert Cohen's Kappa to BCR Construct Validity score (0-3)."""
    if k >= 0.8:
        return 3
    if k >= 0.6:
        return 2
    if k >= 0.4:
        return 1
    return 0

SCORE = rubric(kappa)
print(f"Cohen κ = {kappa:.3f} → BCR Construct Validity {SCORE} / 3")
