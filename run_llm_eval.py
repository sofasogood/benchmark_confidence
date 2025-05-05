"""Run LLM evaluation on MMLU test set and save predictions."""
import time
import os
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
from utils import load_mmlu

# ------------------------------------------------------------------ config
MODEL            = "gpt-4o-mini"     # or "gpt-4o" / "gpt-4-turbo" / etc.
TEMPERATURE      = 0                 # deterministic
MAX_RETRIES      = 3                 # simple exponential back-off
OUT_PATH         = "gpt4_preds.csv"
SAMPLE_FRACTION  = 0.05               # set to 0.2 for a cheap test run
SEED             = 42
# -------------------------------------------------------------------------

openai.api_key   = os.environ["OPENAI_API_KEY"]
rng              = np.random.default_rng(SEED)

df  = load_mmlu("mmlu_test.csv")
if SAMPLE_FRACTION < 1.0:
    df = df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)

pred_correct: list[int] = []

def query_llm(question: str, choices: list[str]) -> int | None:  # type: ignore
    """Return the choice index the model believes is correct (0-based)."""
    prompt = (
        "You are an expert test-solver. Choose the single best answer.\n\n"
        f"QUESTION:\n{question}\n\nOPTIONS:\n" +
        "\n".join(f"{i}) {opt}" for i, opt in enumerate(choices)) +
        "\n\nAnswer with the option *number* only (0-based)."
    )
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = openai.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4,
            )
            content = resp.choices[0].message.content
            if content is None:
                return None
            txt = content.strip()
            # quick clean-up: grab first integer in response
            choice_idx = int(next(filter(str.isdigit, txt)))
            return choice_idx
        except (openai.APIError, ValueError, StopIteration) as e:
            if attempt == MAX_RETRIES:
                print(" ✗ giving up:", e)
                return None
            sleep = 2 ** attempt
            print(f"Retry {attempt}/{MAX_RETRIES} in {sleep}s … ({e})")
            time.sleep(sleep)

# ---------------- main loop ------------------------------------------------
for row in tqdm(df.itertuples(), total=len(df), desc="LLM-eval"):
    assert isinstance(row.question, str)
    assert isinstance(row.choices, list)
    idx = query_llm(row.question, row.choices)
    pred_correct.append(int(idx == row.answer) if idx is not None else 0)

pd.DataFrame({"is_correct": pred_correct}).to_csv(OUT_PATH, index=False)
acc = np.mean(pred_correct) * 100
print(f"\nSaved {OUT_PATH}  —  accuracy {acc:.2f}% on {len(df)} items")
