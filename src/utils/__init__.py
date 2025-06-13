"""Utility functions for the benchmark."""
from .dataset import load_mmlu
from .perturb import inject_noise, shuffle_choices

__all__ = ['load_mmlu', 'inject_noise', 'shuffle_choices'] 