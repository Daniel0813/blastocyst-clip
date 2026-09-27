"""Build the single clean label table for the Kromp 2023 blastocyst dataset.

Reads the four inconsistent source files in the dataset folder and writes
`labels.csv` next to them. Every later step reads only that file.

Usage:
    python src/build_labels.py [--data-dir data/blastocyst]

Encoding in the output (same codes as the source files, unified):
    EXP  0-4  = Gardner expansion grade 1-5
    ICM  0/1/2 = A/B/C, 3 = not defined (ND)
    TE   0/1/2 = A/B/C, 3 = not defined (ND)
    Missing gold labels stay empty (NA) and are excluded per grade when scoring.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SEED = 42
N_FOLDS = 5          # grouped CV folds, used to tune the linear heads
VAL_SPLITS = 40      # validation is built from small grouped folds (1/40 of train each)
VAL_FRACTION = 0.15
ND = 3

# Known data problems (see CLAUDE.md, "Known data problems")
DROP_IMAGES = {"838_02.png"}   # listed twice in train with conflicting EXP (4 vs 5)
GRADES = ["EXP", "ICM", "TE"]


def patient_id(image: str) -> int:
    """Filename prefix, e.g. '0175_05.png' -> 175, '420_02.png' -> 420."""
    return int(image.split("_")[0])


def to_code(value):
    """Map a source label to an integer code; 'ND' -> 3; blank -> NA."""
    if pd.isna(value):
        return pd.NA
    if str(value).strip().upper() == "ND":
        return ND
    return int(float(value))


def parse_agreement(desc):
    """'4/5' -> (0.8, 5, False); 'revised_cons' -> (NA, NA, True); 'ND'/blank -> all NA."""
    if pd.isna(desc) or str(desc).strip().upper() == "ND":
        return pd.NA, pd.NA, pd.NA
    s = str(desc).strip()
    if s == "revised_cons":
        return pd.NA, pd.NA, True
    m = re.fullmatch(r"(\d+)/(\d+)", s)
    if not m:
        raise ValueError(f"unexpected agreement value: {desc!r}")
    k, n = int(m.group(1)), int(m.group(2))
    return k / n, n, False


def load(data_dir: Path) -> pd.DataFrame:
    train = pd.read_csv(data_dir / "Gardner_train_silver.csv", sep=";")
    train = train.rename(columns={"EXP_silver": "EXP", "ICM_silver": "ICM", "TE_silver": "TE"})
    train["split"] = "train"

    test = pd.read_csv(data_dir / "Gardner_test_gold_onlyGardnerScores.csv", sep=";",
                       encoding="utf-8-sig").dropna(axis=1, how="all")
    test = test.rename(columns={"EXP_gold": "EXP", "ICM_gold": "ICM", "TE_gold": "TE"})
    test["split"] = "test"

    df = pd.concat([train, test], ignore_index=True)
    df["image"] = df.pop("Image").str.strip()      # raw names, never re-padded
    for g in GRADES:
        df[g] = df[g].map(to_code).astype("Int64")

    # Drop every row of a conflicting duplicate
    df = df[~df["image"].isin(DROP_IMAGES)].reset_index(drop=True)
    df["patient_id"] = df["image"].map(patient_id)

    # Per-image annotator agreement (test set only)
    gold = pd.read_excel(data_dir / "Gardner_test_gold.xlsx")
    gold["Image"] = gold["Image"].str.strip()
    for g in GRADES:
        parsed = gold[f"{g}_Agreement_desc"].map(parse_agreement)
        gold[f"agree_{g}"] = parsed.map(lambda t: t[0]).astype("Float64")
        gold[f"raters_{g}"] = parsed.map(lambda t: t[1]).astype("Int64")
        gold[f"revised_{g}"] = parsed.map(lambda t: t[2]).astype("boolean")
    agree_cols = [f"{p}_{g}" for g in GRADES for p in ("agree", "raters", "revised")]
    df = df.merge(gold[["Image"] + agree_cols].rename(columns={"Image": "image"}),
                  on="image", how="left")
    return df


def add_splits(df: pd.DataFrame) -> pd.DataFrame:
    tr = df["split"] == "train"
    X = np.zeros(tr.sum())
    y = df.loc[tr, "EXP"].to_numpy()
    groups = df.loc[tr, "patient_id"].to_numpy()

    # 5 grouped, EXP-stratified folds over the training set (for tuning linear heads)
    folds = np.full(tr.sum(), -1)
    for k, (_, idx) in enumerate(StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=SEED)
                                 .split(X, y, groups)):
        folds[idx] = k
    df["fold"] = pd.array([pd.NA] * len(df), dtype="Int64")
    df.loc[tr, "fold"] = folds

    # Fixed validation split (for fine-tuning runs): add small grouped, EXP-stratified
    # folds until at least VAL_FRACTION of train is covered. Patient groups vary in
    # size, so a fixed number of folds would undershoot.
    val = np.zeros(tr.sum(), dtype=bool)
    splitter = StratifiedGroupKFold(VAL_SPLITS, shuffle=True, random_state=SEED)
    for _, idx in splitter.split(X, y, groups):
        if val.mean() >= VAL_FRACTION:
            break
        val[idx] = True
    df["is_val"] = False
    df.loc[tr, "is_val"] = val

    # Train images whose patient also appears in the official test set
    test_patients = set(df.loc[~tr, "patient_id"])
    df["patient_overlap"] = tr & df["patient_id"].isin(test_patients)
    return df


def check(df: pd.DataFrame, data_dir: Path) -> list[str]:
    report = []
    tr, te = df[df.split == "train"], df[df.split == "test"]

    assert len(tr) == 2042, f"train rows {len(tr)} != 2042 (2,044 minus the duplicate pair)"
    assert len(te) == 300, f"test rows {len(te)} != 300"
    assert df["image"].is_unique, "duplicate image names remain"
    assert not set(tr.image) & set(te.image), "image in both train and test"

    on_disk = {p.name for p in (data_dir / "Images").glob("*.png")}
    missing = set(df.image) - on_disk
    assert not missing, f"labelled images not on disk: {sorted(missing)[:5]}"
    unlisted = sorted(on_disk - set(df.image))
    report.append(f"images on disk but not in the table (excluded): {unlisted}")

    # Hard rule: EXP grade 1-2 (codes 0-1) <=> ICM and TE are ND
    for part, name in [(tr, "train"), (te.dropna(subset=GRADES), "test")]:
        early = part["EXP"] <= 1
        nd = (part["ICM"] == ND) & (part["TE"] == ND)
        bad = int((early != nd).sum())
        assert bad == 0 or name == "test", f"ND rule broken in {name}: {bad} rows"
        report.append(f"ND rule exceptions in {name}: {bad}")

    # No patient may cross a fold or the validation boundary
    assert (tr.groupby("patient_id")["fold"].nunique() == 1).all(), "patient split across folds"
    assert (tr.groupby("patient_id")["is_val"].nunique() == 1).all(), "patient split across val"

    report.append(f"validation images: {int(tr.is_val.sum())} ({tr.is_val.mean():.1%} of train)")
    report.append("fold sizes: " + ", ".join(f"{k}:{n}" for k, n in tr.fold.value_counts().sort_index().items()))
    report.append(f"patient-overlap train images: {int(tr.patient_overlap.sum())} "
                  f"-> patient-level train set: {int((~tr.patient_overlap).sum())}")
    for g in GRADES:
        report.append(f"test images missing gold {g}: {int(te[g].isna().sum())}")
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/blastocyst", type=Path)
    args = ap.parse_args()

    df = add_splits(load(args.data_dir))
    for line in check(df, args.data_dir):
        print(line)

    cols = (["image", "split", "patient_id", *GRADES, "fold", "is_val", "patient_overlap"]
            + [c for c in df.columns if c.startswith(("agree_", "raters_", "revised_"))])
    out = args.data_dir / "labels.csv"
    df[cols].sort_values(["split", "image"]).to_csv(out, index=False)
    print(f"wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
