"""Run LLM evaluation on MMLU test set and save predictions."""
import argparse
import time
import os
import numpy as np
import pandas as pd
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from src.utils.dataset import load_mmlu, stratified_sample
import ast

load_dotenv(override=True)

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", default="data/raw/mmlu_test.csv")
parser.add_argument("--out",     default="data/predictions/gpt4_preds.csv")

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

# ------------------------------------------------------------------ config
MODEL            = "gpt-4.1"           # Updated from gpt-4o-mini
TEMPERATURE      = 0                 # deterministic
MAX_RETRIES      = 3                 # simple exponential back-off
SAMPLE_FRACTION  = 0.02 #0.7              # use full dataset
SEED             = 42
# -------------------------------------------------------------------------

# Debug prints for API key
print("API Key from env:", os.environ.get("OPENAI_API_KEY", "Not found"))
openai.api_key = os.environ["OPENAI_API_KEY"]
print("OpenAI API Key set to:", openai.api_key)

rng              = np.random.default_rng(SEED)
args = parser.parse_args()
df  = pd.read_csv(args.dataset)

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
    choices = ast.literal_eval(row.choices)  # Convert string representation of list to actual list
    idx = query_llm(row.question, choices)
    pred_correct.append(int(idx == row.answer) if idx is not None else 0)

pd.DataFrame({"is_correct": pred_correct}).to_csv(args.out, index=False)
acc = np.mean(pred_correct) * 100
print(f"\nSaved {args.out}  —  accuracy {acc:.2f}% on {len(df)} items")
