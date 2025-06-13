# BCR: Benchmark Confidence Rubric

A Python implementation of the BCR (Benchmark Confidence Rubric) framework for evaluating LLM benchmarks. This framework assesses benchmarks across multiple dimensions to provide a comprehensive confidence score:

- **Robustness**: How well the benchmark maintains accuracy under perturbations (noise, paraphrasing, choice shuffling)
- **Coverage**: How well the benchmark covers different subject areas (measured by entropy of subject distribution)
- **Construct Validity**: How well the benchmark measures its intended constructs (measured by inter-rater agreement)
- **External Validity**: How well the benchmark generalizes across domains (comparing STEM vs non-STEM performance)
- **Difficulty & Discrimination**: How well the benchmark discriminates between different ability levels (measuring ceiling/floor effects)
- **Power**: Statistical confidence in the results (measured by confidence interval width)

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

1. Generate stratified sample of the dataset:
```bash
python fix_sample.py  # Creates a 2% stratified sample of the dataset
```

2. Generate perturbed datasets:
```bash
python make_perturbed_set.py  # Creates noise and shuffle variants
python make_paraphrased_set.py  # Creates paraphrased variant
```

3. Run LLM evaluation on each dataset:
```bash
python run_llm_eval.py --dataset="data/raw/mmlu_test_sampled_0.02.csv" --out="data/predictions/gpt4_preds.csv"
python run_llm_eval.py --dataset="data/raw/mmlu_test_sampled_0.02.csv" --out="data/predictions/gpt4_noise.csv"
python run_llm_eval.py --dataset="data/raw/mmlu_test_sampled_0.02.csv" --out="data/predictions/gpt4_shuffle.csv"
python run_llm_eval.py --dataset="data/raw/mmlu_test_sampled_0.02.csv" --out="data/predictions/gpt4_paraphrase.csv"
```

4. Calculate BCR scores:
```bash
python run_metrics.py  # Runs all metrics and generates visualizations
```

Or run individual metrics:
```bash
python robustness.py  # Robustness score
python coverage.py    # Coverage score
python construct_validity.py  # Construct validity score
python external_validity.py   # External validity score
python difficulty_discrimination.py  # Difficulty & Discrimination score
python power_ci.py  # Power and confidence intervals
```

## Project Structure

```
bcr/
├── data/                  # Data files
│   ├── raw/              # Original and sampled datasets
│   ├── perturbed/        # Generated perturbed datasets
│   ├── predictions/      # Model predictions
│   └── metrics_results/  # Metric calculation results
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
│   │   ├── difficulty_discrimination.py
│   │   └── power_ci.py
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

### Scoring Criteria

- **Robustness**: Based on accuracy drop under perturbations
  - 3: Excellent (drop ≤ 2pp)
  - 2: Good (2pp < drop ≤ 5pp)
  - 1: Fair (5pp < drop ≤ 10pp)
  - 0: Poor (drop > 10pp)

- **Coverage**: Based on normalized entropy (H/Hmax)
  - 3: Excellent (H/Hmax ≥ 0.90)
  - 2: Good (0.75 ≤ H/Hmax < 0.90)
  - 1: Fair (0.50 ≤ H/Hmax < 0.75)
  - 0: Poor (H/Hmax < 0.50)

- **External Validity**: Based on STEM vs non-STEM gap
  - 3: Excellent (gap ≤ 2pp)
  - 2: Good (2pp < gap ≤ 5pp)
  - 1: Fair (5pp < gap ≤ 10pp)
  - 0: Poor (gap > 10pp)

- **Difficulty & Discrimination**: Based on ceiling/floor effects
  - 3: Excellent (< 5% ceiling/floor)
  - 2: Good (5-10% ceiling/floor)
  - 1: Fair (10-20% ceiling/floor)
  - 0: Poor (> 20% ceiling/floor)

- **Power**: Based on confidence interval width
  - 3: Excellent (CI width ≤ 2pp)
  - 2: Good (2pp < CI width ≤ 5pp)
  - 1: Fair (5pp < CI width ≤ 10pp)
  - 0: Poor (CI width > 10pp) 