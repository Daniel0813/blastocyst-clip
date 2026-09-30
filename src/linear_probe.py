"""E1 — THE BASELINE: frozen CLIP image embeddings + three linear heads.

Heads: multinomial logistic regression (one linear layer, plain cross-entropy),
no class weights, no ordinal loss — later changes are measured against this.
  EXP head: all training images, 5 classes
  ICM / TE heads: training images with EXP grade >= 3 only, 3 classes (A/B/C)
Prediction: EXP grade 1-2 -> ICM = TE = ND (rule, not learned).
The only hyperparameter, the inverse regularisation strength C, is chosen per
head by 5-fold grouped CV (labels.csv `fold`, grouped by patient), on accuracy
(default) or macro F1 (--select f1_macro; accuracy rewards always predicting
the majority grade).
Logistic regression with lbfgs is deterministic, so there are no seeds to vary.

Runs on both training sets:
  official       all 2,042 training images
  patient-level  1,413 — minus images whose patient also appears in the test set

Usage:
    python src/linear_probe.py --emb-dir results/embeddings/biomedclip
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate  # noqa: E402

ND = 3
C_GRID = np.logspace(-4, 1, 11)
# Feature choice -> file suffixes written by src/embed.py (concatenated if several)
FEATURES = {"proj": [""], "cls": ["_cls"], "patchmean": ["_patchmean"], "cls+patchmean": ["_cls", "_patchmean"]}


def make_head(C):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000))


SELECT = "acc"


def cv_score(y_true, y_pred):
    if SELECT == "f1_macro":
        return f1_score(y_true, y_pred, average="macro", zero_division=0)
    return (y_true == y_pred).mean()


def tune_and_fit(X, y, folds):
    """Pick C by grouped CV score, refit on all rows. Returns (model, C, cv_score)."""
    scores = []
    for C in C_GRID:
        accs = []
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            m = make_head(C).fit(X[tr], y[tr])
            accs.append(cv_score(y[va], m.predict(X[va])))
        scores.append(np.mean(accs))
    best = int(np.argmax(scores))
    return make_head(C_GRID[best]).fit(X, y), C_GRID[best], scores[best]


def run(emb, labels, train_mask, name, out_dir) -> pd.DataFrame:
    tr = labels[train_mask]
    X_tr = emb[train_mask.to_numpy()]
    folds = tr["fold"].to_numpy()
    info = []

    exp_model, C, cv = tune_and_fit(X_tr, tr["EXP"].to_numpy(), folds)
    info.append(("EXP", len(tr), C, cv))

    later = (tr["EXP"] >= 2).to_numpy()
    heads = {}
    for g in ("ICM", "TE"):
        keep = later & tr[g].isin([0, 1, 2]).to_numpy()
        heads[g], C, cv = tune_and_fit(X_tr[keep], tr[g].to_numpy()[keep], folds[keep])
        info.append((g, int(keep.sum()), C, cv))

    test = (labels["split"] == "test").to_numpy()
    X_te = emb[test]
    pred = pd.DataFrame({"image": labels.loc[test, "image"].to_numpy(),
                         "EXP": exp_model.predict(X_te)})
    for g in ("ICM", "TE"):
        pred[g] = heads[g].predict(X_te)
    pred.loc[pred["EXP"] <= 1, ["ICM", "TE"]] = ND

    path = out_dir / f"{name}_pred.csv"
    pred.to_csv(path, header=False, index=False)

    # Class probabilities per head (for track B: confidence vs expert agreement).
    # ICM/TE probabilities are over A/B/C and are given for every test image,
    # including those the ND rule overrides.
    proba = pd.DataFrame({"image": pred["image"]})
    for g, model in (("EXP", exp_model), ("ICM", heads["ICM"]), ("TE", heads["TE"])):
        for cls, col in zip(model.classes_, model.predict_proba(X_te).T):
            proba[f"{g}_p{int(cls)}"] = col
    proba.to_csv(out_dir / f"{name}_proba.csv", index=False)
    print(f"\n[{name}] training images: {len(tr)}")
    for g, n, C, cv in info:
        print(f"  {g:3s} head: trained on {n:4d}, C={C:.4g}, CV {SELECT} {cv:.3f}")
    return evaluate.evaluate(path, evaluate.load_gold(args_labels)).assign(train_set=name)


def majority_baseline(labels, out_dir) -> pd.DataFrame:
    """Always predict the most common training grade (ND rule applied)."""
    tr = labels[labels.split == "train"]
    exp = int(tr["EXP"].mode()[0])
    later = tr[tr["EXP"] >= 2]
    icm, te = int(later["ICM"].mode()[0]), int(later["TE"].mode()[0])
    if exp <= 1:
        icm = te = ND
    test = labels[labels.split == "test"]
    pred = pd.DataFrame({"image": test["image"], "EXP": exp, "ICM": icm, "TE": te})
    path = out_dir / "majority_pred.csv"
    pred.to_csv(path, header=False, index=False)
    return evaluate.evaluate(path, evaluate.load_gold(args_labels)).assign(train_set="majority")


def main():
    global args_labels, SELECT
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="results/embeddings/biomedclip", type=Path)
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    ap.add_argument("--select", choices=["acc", "f1_macro"], default="acc",
                    help="CV criterion for choosing C")
    ap.add_argument("--feature", choices=list(FEATURES), default="proj",
                    help="proj = 512-d CLIP embedding; cls / patchmean = 768-d trunk features")
    args = ap.parse_args()
    args_labels = args.labels
    SELECT = args.select

    labels = pd.read_csv(args.labels)
    names = pd.read_csv(args.emb_dir / "blastocyst_images.csv")["image"]
    assert names.tolist() == labels["image"].tolist(), "embedding order != labels.csv order"
    emb = np.concatenate([np.load(args.emb_dir / f"blastocyst_images{s}.npy") for s in FEATURES[args.feature]],
                         axis=1)
    print(f"features: {args.feature} {emb.shape}")

    tag = args.emb_dir.name + ("" if args.feature == "proj" else f"_{args.feature}") \
        + ("" if SELECT == "acc" else f"_{SELECT}")
    out_dir = Path("results/linear_probe") / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    is_train = labels["split"] == "train"
    res = pd.concat([
        run(emb, labels, is_train, "official", out_dir),
        run(emb, labels, is_train & ~labels["patient_overlap"], "patient_level", out_dir),
        majority_baseline(labels, out_dir),
    ], ignore_index=True)
    res.to_csv(out_dir / "results.csv", index=False)

    for protocol in ("clean", "kromp"):
        sub = res[res.protocol == protocol]
        print(f"\n=== {protocol} protocol: accuracy [95% CI] | macro F1 ===")
        for ts in ("official", "patient_level", "majority"):
            row = sub[sub.train_set == ts].set_index("target")
            cells = [f"{t} {row.loc[t, 'acc']:.3f} [{row.loc[t, 'acc_lo']:.2f}-{row.loc[t, 'acc_hi']:.2f}] "
                     f"| F1 {row.loc[t, 'f1_macro']:.3f}" for t in ("EXP", "ICM", "TE")]
            print(f"  {ts:14s} " + "   ".join(cells))
    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    main()
