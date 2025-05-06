"""Calculate BCR External Validity score based on accuracy drop."""
import pandas as pd
import numpy as np
from src.utils.dataset import load_mmlu

STEM_SUBJECTS = {
    "astronomy","clinical_knowledge","college_biology","college_chemistry",
    "college_computer_science","college_mathematics","college_physics",
    "electrical_engineering","high_school_biology","high_school_chemistry",
    "high_school_computer_science","high_school_mathematics",
    "high_school_physics","high_school_statistics"
}
PRED_PATH      = "gpt4_preds.csv"     # correctness for each row
DATA_PATH      = "mmlu_test_sampled_0.02.csv"

df    = load_mmlu(DATA_PATH)
pred  = pd.read_csv(PRED_PATH)["is_correct"].values

mask_stem = np.array(df["subject"].isin(STEM_SUBJECTS).values)
mask_non  = ~mask_stem

def acc(mask):
    """Calculate accuracy of a mask."""
    return pred[mask].mean() * 100      # percentage

acc_stem = acc(mask_stem)
acc_non  = acc(mask_non)
delta    = abs(acc_stem - acc_non)

def rubric(d):
    """Convert accuracy drop to BCR External Validity score (0-3)."""
    if d <= 2:   
        return 3
    if d <= 5:   
        return 2
    if d <= 10:  
        return 1
    return 0

print(f"STEM items     : {mask_stem.sum()}  •  Acc {acc_stem:5.2f} %")
print(f"Non-STEM items : {mask_non.sum()}  •  Acc {acc_non:5.2f} %")
print(f"Gap            : {delta:5.2f} pp")
print(f"BCR Ext-Validity: {rubric(delta)} / 3")
