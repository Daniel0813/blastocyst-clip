"""Evaluation harness for Gardner grade predictions on the 300-image gold test set.

Two protocols, both reported:

  "kromp"  Reproduces the dataset authors' evaluate_prediction_results.py exactly,
           so our numbers are comparable with the published baselines:
             - a missing gold EXP becomes class 5 ("not assessable") and is scored
             - a missing gold ICM/TE becomes 3 (ND) and is scored
             - ICM/TE are scored only where BOTH the gold and the *predicted* EXP
               are grade >= 3, so each model is scored on its own subset

  "clean"  Our protocol: same test images for every model.
             - images with a missing gold label are excluded, per grade
             - ICM/TE scored on all images, ND included as a class, so an EXP
               error that wrongly triggers ND counts as an ICM/TE error
             - "*_given_exp": ICM/TE on images whose gold EXP is grade >= 3 only,
               i.e. the quality of the ICM/TE head independent of the EXP head

Usage:
    python src/evaluate.py predictions.csv [more.csv ...] [--labels data/blastocyst/labels.csv]
Prediction files: `image,EXP,ICM,TE` rows with no header (the authors' format),
codes as in labels.csv (EXP 0-4, ICM/TE 0-2, 3 = ND).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ND = 3
GRADES = ["EXP", "ICM", "TE"]
N_BOOT = 1000
SEED = 0


def load_gold(labels_path) -> pd.DataFrame:
    df = pd.read_csv(labels_path)
    gold = df[df["split"] == "test"].set_index("image")[GRADES]
    return gold.astype("Int64")


def load_predictions(path) -> pd.DataFrame:
    pred = pd.read_csv(path, header=None, names=["image", *GRADES])
    pred["image"] = pred["image"].str.strip()
    pred = pred.set_index("image")
    # Same normalisation as the authors' script: any non-ABC ICM/TE value means ND
    for g in ("ICM", "TE"):
        pred[g] = pred[g].where(pred[g].isin([0, 1, 2]), ND)
    return pred.astype(int)


def metrics(y_true, y_pred, rng) -> dict:
    y_true, y_pred = np.asarray(y_true, int), np.asarray(y_pred, int)
    out = {
        "n": len(y_true),
        "acc": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }
    labels = sorted(set(y_true) | set(y_pred))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    out["f1_per_class"] = json.dumps({int(c): round(float(f), 4) for c, f in zip(labels, per_class)})
    out["confusion"] = json.dumps({"labels": [int(c) for c in labels],
                                   "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()})
    # Bootstrap 95% CI on accuracy
    idx = rng.integers(0, len(y_true), size=(N_BOOT, len(y_true)))
    boots = (y_true[idx] == y_pred[idx]).mean(axis=1)
    out["acc_lo"], out["acc_hi"] = np.percentile(boots, [2.5, 97.5])
    return out


def score_kromp(gold: pd.DataFrame, pred: pd.DataFrame, rng) -> dict:
    g = gold.loc[pred.index.intersection(gold.index)].copy()
    p = pred.loc[g.index]
    exp_true = g["EXP"].fillna(5).astype(int)            # missing -> not assessable
    exp_pred = p["EXP"].where(p["EXP"].isin(range(5)), 5)
    res = {"EXP": metrics(exp_true, exp_pred, rng)}
    keep = (exp_true >= 2) & (exp_pred >= 2) & (exp_true != 5) & (exp_pred != 5)
    for grade in ("ICM", "TE"):
        t = g.loc[keep, grade].fillna(ND).astype(int)     # missing -> ND
        res[grade] = metrics(t, p.loc[keep, grade], rng)
    return res


def score_clean(gold: pd.DataFrame, pred: pd.DataFrame, rng) -> dict:
    res = {}
    for grade in GRADES:
        ok = gold[grade].notna() & gold.index.isin(pred.index)
        res[grade] = metrics(gold.loc[ok, grade], pred.loc[gold.index[ok], grade], rng)
    for grade in ("ICM", "TE"):
        ok = gold[grade].notna() & (gold["EXP"] >= 2) & gold.index.isin(pred.index)
        res[f"{grade}_given_exp"] = metrics(gold.loc[ok, grade], pred.loc[gold.index[ok], grade], rng)
    return res


def evaluate(pred_path, gold) -> pd.DataFrame:
    pred = load_predictions(pred_path)
    missing = set(gold.index) - set(pred.index)
    if missing:
        print(f"warning: {pred_path} lacks {len(missing)} test images")
    rows = []
    for protocol, fn in (("kromp", score_kromp), ("clean", score_clean)):
        rng = np.random.default_rng(SEED)
        for target, m in fn(gold, pred, rng).items():
            rows.append({"protocol": protocol, "target": target, **m})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("predictions", nargs="+", type=Path)
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    ap.add_argument("--out", type=Path, help="optional CSV to write all results to")
    ap.add_argument("--name-from-parent", action="store_true",
                    help="name each model by its parent folder (for model/pred.csv layouts)")
    args = ap.parse_args()

    gold = load_gold(args.labels)
    frames = []
    for path in args.predictions:
        name = path.parent.name if args.name_from_parent else path.stem
        frames.append(evaluate(path, gold).assign(model=name))
    res = pd.concat(frames, ignore_index=True)

    for protocol in ("kromp", "clean"):
        sub = res[res.protocol == protocol]
        table = sub.pivot(index="model", columns="target", values="acc")
        print(f"\n=== accuracy, {protocol} protocol ===")
        print(table.round(3).to_string())
        f1 = sub.pivot(index="model", columns="target", values="f1_macro")
        print(f"--- macro F1, {protocol} protocol ---")
        print(f1.round(3).to_string())
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        res.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
