# BCR Benchmark Power Demo

A Python implementation of the BCR (Benchmark Confidence Rubric) framework for evaluating LLM benchmarks. This framework assesses benchmarks across multiple dimensions to provide a comprehensive confidence score:

- **Robustness**: How well the benchmark maintains accuracy under perturbations (noise, paraphrasing, choice shuffling)
- **Coverage**: How well the benchmark covers different subject areas (measured by entropy of subject distribution)
- **Construct Validity**: How well the benchmark measures its intended constructs (measured by inter-rater agreement)
- **External Validity**: How well the benchmark generalizes across domains (comparing STEM vs non-STEM performance)
- **Difficulty & Discrimination**: How well the benchmark discriminates between different ability levels (measuring ceiling/floor effects)

Each dimension is scored on a scale of 0-3, providing a confidence level in the benchmark's reliability and validity. The final BCR score is the average across all dimensions, giving an overall confidence rating in the benchmark's quality.

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set your OpenAI API key:
```bash
export OPENAI_API_KEY='your-key-here'
```

## Usage

1. Generate perturbed datasets:
```bash
python make_perturbed_set.py  # Creates noise and shuffle variants
python make_paraphrased_set.py  # Creates paraphrased variant
```

2. Run LLM evaluation on each dataset:
```bash
python run_llm_eval.py --dataset="mmlu_test.csv" --out="gpt4_preds.csv"
python run_llm_eval.py --dataset="mmlu_noise.csv" --out="gpt4_noise.csv"
python run_llm_eval.py --dataset="mmlu_shuffle.csv" --out="gpt4_shuffle.csv"
python run_llm_eval.py --dataset="mmlu_paraphrase.csv" --out="gpt4_paraphrase.csv"
```

3. Calculate BCR scores:
```bash
python robustness.py  # Robustness score
python coverage.py    # Coverage score
python construct_validity.py  # Construct validity score
python external_validity.py   # External validity score
python difficulty_discrimination.py  # Difficulty & Discrimination score
```

## Project Structure

```
bcr/
├── data/                  # Data files
│   ├── raw/              # Original datasets
│   ├── perturbed/        # Generated perturbed datasets
│   └── predictions/      # Model predictions
├── src/                  # Source code
│   ├── generation/       # Dataset generation scripts
│   │   ├── make_perturbed_set.py
│   │   └── make_paraphrased_set.py
│   ├── evaluation/       # LLM evaluation scripts
│   │   └── run_llm_eval.py
│   ├── metrics/          # BCR metric calculations
│   │   ├── robustness.py
│   │   ├── coverage.py
│   │   ├── construct_validity.py
│   │   ├── external_validity.py
│   │   └── difficulty_discrimination.py
│   └── utils/           # Helper functions
│       ├── dataset.py
│       ├── perturb.py
│       └── constructs.py
├── config/              # Configuration files
│   └── constructs.yml
├── requirements.txt
└── README.md
```

## Scoring

Each dimension is scored on a scale of 0-3:
- 3: Excellent
- 2: Good
- 1: Fair
- 0: Poor

The final BCR score is the average across all dimensions. 