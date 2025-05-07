"""
make_paraphrase_set.py
Generate a paraphrased version of every MMLU question using GPT-4.
Saves mmlu_paraphrase_sampled.csv with the same columns as mmlu_test_sampled_0.02.csv.
"""
import os
import time
import openai
from tqdm import tqdm
from dotenv import load_dotenv
from src.utils.dataset import load_mmlu

# Load environment variables from .env file
load_dotenv()

# Create directories if they don't exist
os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/predictions", exist_ok=True)

openai.api_key = os.environ["OPENAI_API_KEY"]
print(openai.api_key)
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables")

SRC_PATH   = "data/raw/mmlu_test_sampled_0.02.csv"
OUT_PATH   = "data/predictions/mmlu_paraphrase_sampled.csv"
MODEL      = "gpt-4"
TEMP       = 0.7       # higher → more diverse wording
MAX_RETRY  = 3

# Load sampled dataset
df = load_mmlu(SRC_PATH)               # choices parsed into lists
print(f"Loaded sampled dataset with {len(df)} questions")
paraphrased = []

def paraphrase(text: str) -> str:  # type: ignore
    """Paraphrase a question while preserving its meaning using GPT-4."""
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
print(f"✓ Saved paraphrased version to {OUT_PATH}")
