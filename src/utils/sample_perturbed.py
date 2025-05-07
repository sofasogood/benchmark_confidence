import pandas as pd
import numpy as np

def main():
    # Load the sampled test set to get the indices
    sampled_df = pd.read_csv("data/raw/mmlu_test_sampled_0.02.csv")
    sampled_indices = set(sampled_df.index)
    
    # Process each perturbed dataset
    perturbed_files = [
        "data/perturbed/mmlu_noise.csv",
        "data/perturbed/mmlu_shuffle.csv",
        "data/perturbed/mmlu_paraphrase.csv"
    ]
    
    for file in perturbed_files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        # Sample the same indices as the test set
        sampled = df.iloc[list(sampled_indices)]
        # Save to predictions directory
        output_file = file.replace("perturbed", "predictions").replace("mmlu_", "gpt4_")
        sampled.to_csv(output_file, index=False)
        print(f"Saved sampled dataset to {output_file}")

if __name__ == "__main__":
    main() 