# perturb.py
import random, string, copy
from typing import List, Tuple

def inject_noise(text: str, prob_space=0.15, prob_char=0.10) -> str:
    """Randomly double spaces and insert benign punctuation."""
    out = []
    for ch in text:
        out.append(ch)
        if ch == " " and random.random() < prob_space:
            out.append(" ")                       # double-space
        if ch.isalpha() and random.random() < prob_char:
            out.append(random.choice(",.;:-"))    # harmless char
    return "".join(out)

def shuffle_choices(choices: List[str], answer_idx: int) -> Tuple[List[str], int]:
    """Return new list + new index for the correct answer."""
    idxs = list(range(len(choices)))
    random.shuffle(idxs)
    shuffled = [choices[i] for i in idxs]
    new_answer = idxs.index(answer_idx)
    return shuffled, new_answer
