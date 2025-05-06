"""Run LLM evaluation on MMLU test set and save predictions."""
import argparse
import time
import os
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
from src.utils.dataset import load_mmlu, stratified_sample

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="mmlu_test.csv")
parser.add_argument("--out",     default="gpt4_preds.csv")


# ------------------------------------------------------------------ config
MODEL            = "gpt-4o-mini"     # or "gpt-4o" / "gpt-4-turbo" / etc.
TEMPERATURE      = 0                 # deterministic
MAX_RETRIES      = 3                 # simple exponential back-off
SAMPLE_FRACTION  = 0.02 #0.7              # use full dataset
SEED             = 42
# -------------------------------------------------------------------------

openai.api_key   = os.environ["OPENAI_API_KEY"]
rng              = np.random.default_rng(SEED)
args = parser.parse_args()
df  = load_mmlu(args.dataset)

if SAMPLE_FRACTION < 1.0:
    df = stratified_sample(df, frac=SAMPLE_FRACTION, seed=SEED)
    # df = df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)
    # Save the sampled dataset
    sampled_path = args.dataset.replace('.csv', f'_sampled_{SAMPLE_FRACTION}.csv')
    df.to_csv(sampled_path, index=False)
    print(f"Saved sampled dataset to {sampled_path}")

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

pd.DataFrame({"is_correct": pred_correct}).to_csv(args.out, index=False)
acc = np.mean(pred_correct) * 100
print(f"\nSaved {args.out}  —  accuracy {acc:.2f}% on {len(df)} items")
