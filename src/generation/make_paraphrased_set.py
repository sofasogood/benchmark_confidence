"""
make_paraphrase_set.py
Generate a paraphrased version of every MMLU question using GPT-4o-mini.
Saves mmlu_paraphrase.csv with the same columns as mmlu_test.csv.
"""
import os
import time
import openai
from tqdm import tqdm
from src.utils.dataset import load_mmlu

openai.api_key = os.environ["OPENAI_API_KEY"]

SRC_PATH   = "mmlu_test.csv"
OUT_PATH   = "mmlu_paraphrase.csv"
MODEL      = "gpt-4o-mini"
TEMP       = 0.7       # higher → more diverse wording
MAX_RETRY  = 3

df = load_mmlu(SRC_PATH)[0:500]               # choices parsed into lists
paraphrased = []

def paraphrase(text: str) -> str:  # type: ignore
    """Paraphrase a question while preserving its meaning using GPT-4o-mini."""
    prompt = (
        "Paraphrase the following multiple-choice question stem. "
        "Keep meaning identical, don't add context, under 1 sentence longer.\n\n"
        f"Q: {text}"
    )
    for k in range(1, MAX_RETRY+1):
        try:
            r = openai.chat.completions.create(
                    model=MODEL,
                    temperature=TEMP,
                    messages=[{"role":"user","content":prompt}],
                    max_tokens=100)
            content = r.choices[0].message.content
            if content is None:
                raise ValueError("Empty response from API")
            return content.strip()
        except (openai.APIError, ValueError, StopIteration):
            if k == MAX_RETRY:
                raise
            time.sleep(2**k)

for q in tqdm(df["question"], desc="Paraphrasing"):
    paraphrased.append(paraphrase(q))

df_out = df.copy()
df_out["question"] = paraphrased
df_out.to_csv(OUT_PATH, index=False)
print("✓ saved", OUT_PATH)
