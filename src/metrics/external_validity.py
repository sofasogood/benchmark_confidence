"""Calculate BCR External Validity score based on accuracy drop."""
import pandas as pd
import numpy as np
import json
from src.utils.dataset import load_mmlu

STEM_SUBJECTS = {
    "astronomy","clinical_knowledge","college_biology","college_chemistry",
    "college_computer_science","college_mathematics","college_physics",
    "electrical_engineering","high_school_biology","high_school_chemistry",
    "high_school_computer_science","high_school_mathematics",
    "high_school_physics","high_school_statistics"
}

def main(pred_path="data/predictions/gpt4_preds.csv", data_path="data/raw/mmlu_test_sampled_0.02.csv"):
    # Load both files using pandas directly
    df = pd.read_csv(data_path)
    pred_df = pd.read_csv(pred_path)
    
    # Ensure we have the same number of rows
    if len(df) != len(pred_df):
        raise ValueError(f"Dataset size mismatch: {len(df)} questions vs {len(pred_df)} predictions")
    
    pred = pred_df["is_correct"].values

    mask_stem = np.array(df["subject"].isin(STEM_SUBJECTS).values)
    mask_non = ~mask_stem

    def acc(mask):
        """Calculate accuracy of a mask."""
        return pred[mask].mean() * 100      # percentage

    acc_stem = acc(mask_stem)
    acc_non = acc(mask_non)
    delta = abs(acc_stem - acc_non)

    def rubric(d):
        """Convert accuracy drop to BCR External Validity score (0-3)."""
        if d <= 2:   
            return 3
        if d <= 5:   
            return 2
        if d <= 10:  
            return 1
        return 0

    score = rubric(delta)
    print(f"STEM items     : {mask_stem.sum()}  •  Acc {acc_stem:5.2f} %")
    print(f"Non-STEM items : {mask_non.sum()}  •  Acc {acc_non:5.2f} %")
    print(f"Gap            : {delta:5.2f} pp")
    print(f"BCR Ext-Validity: {score} / 3")
    
    return {
        "stem_items": int(mask_stem.sum()),
        "non_stem_items": int(mask_non.sum()),
        "stem_accuracy": float(acc_stem),
        "non_stem_accuracy": float(acc_non),
        "accuracy_gap": float(delta),
        "score": int(score)
    }

if __name__ == "__main__":
    result = main()
    print(json.dumps(result, indent=2))
