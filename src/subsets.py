"""Labelled-subset selection shared by every label-efficiency condition.

A0 (frozen heads), A1 (caption-adapted heads) and A2 (caption + pseudo-caption
fine-tuning) must see exactly the same N labelled images for a given seed, so
all of them select through this one function.
"""
import pandas as pd
from sklearn.model_selection import train_test_split


def labelled_subset(labels: pd.DataFrame, n, seed: int, train_set: str = "official") -> pd.DataFrame:
    """EXP-stratified random subset of n training images (all of them if n is None or too big)."""
    tr = labels[labels["split"] == "train"]
    if train_set == "patient_level":
        tr = tr[~tr["patient_overlap"]]
    if n is None or n >= len(tr):
        return tr
    sub, _ = train_test_split(tr, train_size=n, stratify=tr["EXP"], random_state=seed)
    return sub.sort_values("image")
