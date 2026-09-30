"""Label-efficiency curve: linear heads trained on only N labelled images.

For each label budget N and seed, the N training images come from
src/subsets.labelled_subset (the same images every condition sees), the three
heads are trained exactly as the baseline (logistic regression, C by grouped CV
on macro F1, ICM/TE on EXP >= grade 3, ND rule), and scored on the 300-image
gold test set with both protocols.

  A0  frozen embeddings                   --emb-dir results/embeddings/biomedclip_full
  A1  caption-adapted embeddings          --emb-dir results/embeddings/biomedclip_a1_seed0
  A2  embeddings adapted with N labels    one --emb-dir per (N, seed), run with --n N --seeds s

Usage:
    python src/label_efficiency.py --condition A0 --emb-dir results/embeddings/biomedclip_full \
        --n 50 100 200 500 all --seeds 0 1 2
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate  # noqa: E402
from src import linear_probe as lp  # noqa: E402
from src.subsets import labelled_subset  # noqa: E402

ND = 3


class Constant:
    """Stand-in head when a subset contains only one grade."""
    def __init__(self, value):
        self.value = value

    def predict(self, X):
        return np.full(len(X), self.value)


def tune_and_fit(X, y, folds):
    if len(np.unique(y)) < 2:
        return Constant(y[0]), None
    scores = []
    for C in lp.C_GRID:
        fold_scores = []
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            if va.sum() == 0 or len(np.unique(y[tr])) < 2:
                continue
            m = lp.make_head(C).fit(X[tr], y[tr])
            fold_scores.append(f1_score(y[va], m.predict(X[va]), average="macro", zero_division=0))
        scores.append(np.mean(fold_scores) if fold_scores else -1)
    C = lp.C_GRID[int(np.argmax(scores))] if max(scores) >= 0 else 1.0
    return lp.make_head(C).fit(X, y), C


def run_one(emb, labels, n, seed, out_dir, tag) -> pd.DataFrame:
    sub = labelled_subset(labels, n, seed)
    idx = labels.index.get_indexer(sub.index)
    X, folds = emb[idx], sub["fold"].to_numpy()
    exp_head, _ = tune_and_fit(X, sub["EXP"].to_numpy(), folds)
    heads = {}
    later = (sub["EXP"] >= 2).to_numpy()
    for g in ("ICM", "TE"):
        keep = later & sub[g].isin([0, 1, 2]).to_numpy()
        heads[g], _ = tune_and_fit(X[keep], sub[g].to_numpy()[keep], folds[keep])

    test = (labels["split"] == "test").to_numpy()
    pred = pd.DataFrame({"image": labels.loc[test, "image"].to_numpy(), "EXP": exp_head.predict(emb[test])})
    for g in ("ICM", "TE"):
        pred[g] = heads[g].predict(emb[test])
    pred.loc[pred["EXP"] <= 1, ["ICM", "TE"]] = ND
    path = out_dir / f"{tag}_n{len(sub)}_seed{seed}_pred.csv"
    pred.to_csv(path, header=False, index=False)
    return evaluate.evaluate(path, evaluate.load_gold(lp.args_labels)).assign(n=len(sub), seed=seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", required=True, help="label for this condition, e.g. A0, A1, A2")
    ap.add_argument("--emb-dir", required=True, type=Path)
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    ap.add_argument("--feature", default="cls+patchmean", choices=list(lp.FEATURES))
    ap.add_argument("--n", nargs="+", default=["50", "100", "200", "500", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--out-dir", default="results/label_efficiency", type=Path)
    args = ap.parse_args()
    lp.args_labels = args.labels

    labels = pd.read_csv(args.labels)
    names = pd.read_csv(args.emb_dir / "blastocyst_images.csv")["image"]
    assert names.tolist() == labels["image"].tolist(), "embedding order != labels.csv order"
    emb = np.concatenate([np.load(args.emb_dir / f"blastocyst_images{s}.npy")
                          for s in lp.FEATURES[args.feature]], axis=1)
    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.condition}_{args.emb_dir.name}"

    frames = []
    for n in args.n:
        n_val = None if n == "all" else int(n)
        for seed in (args.seeds if n != "all" else args.seeds[:1]):   # "all" has no randomness
            res = run_one(emb, labels, n_val, seed, pred_dir, tag)
            clean = res[(res.protocol == "clean") & res.target.isin(["EXP", "ICM", "TE"])].set_index("target")
            print(f"  {args.condition} n={clean['n'].iloc[0]:4d} seed {seed}: "
                  + "  ".join(f"{t} {clean.loc[t, 'acc']:.3f}/{clean.loc[t, 'f1_macro']:.3f}" for t in ("EXP", "ICM", "TE")),
                  flush=True)
            frames.append(res)
    out = pd.concat(frames, ignore_index=True).assign(condition=args.condition, emb=args.emb_dir.name)
    path = args.out_dir / f"{tag}.csv"
    out.to_csv(path, index=False)
    print(f"wrote {path}  (columns: accuracy/macro F1 per grade, clean protocol shown above)")


if __name__ == "__main__":
    main()
