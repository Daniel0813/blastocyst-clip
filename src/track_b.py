"""Track B: does model uncertainty match expert disagreement?

For each test image and grade, compare
  model uncertainty   1 - max class probability (ICM/TE over A/B/C only)
  expert disagreement from Gardner_test_gold.xlsx (via labels.csv):
                      1 - vote fraction; images whose grade was only settled by panel
                      discussion ("revised_cons") count as maximal disagreement
Metrics per model and grade:
  spearman   rank correlation, model uncertainty vs expert disagreement
  auroc      how well model uncertainty flags contested images
             (contested = not unanimous, or revised by discussion)
  acc_unan / acc_cont   model accuracy on unanimous vs contested images
ICM/TE use only images whose gold EXP is grade >= 3 and gold ICM/TE is A/B/C.

Models: our baseline (a *_proba.csv from src/linear_probe.py) and the dataset
authors' released models (probability files with per-class mean and SWAG spread).

Usage:
    python src/track_b.py --ours results/linear_probe/biomedclip_full_cls+patchmean_f1_macro/official_proba.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

GRADES = {"EXP": 5, "ICM": 3, "TE": 3}
KROMP_FILES = {"EXP": ("exp.csv", 5), "ICM": ("icm.csv", 4), "TE": ("teq.csv", 4)}


def expert_table(labels_path) -> pd.DataFrame:
    lab = pd.read_csv(labels_path)
    te = lab[lab.split == "test"].set_index("image")
    for g in GRADES:
        revised = te[f"revised_{g}"].astype("boolean").fillna(False).astype(bool)
        te[f"disagree_{g}"] = np.where(revised, 1.0, 1 - te[f"agree_{g}"])
        te[f"contested_{g}"] = revised | (te[f"agree_{g}"] < 1)
        te[f"has_votes_{g}"] = revised | te[f"agree_{g}"].notna()
    return te


def load_ours(path) -> dict[str, pd.DataFrame]:
    p = pd.read_csv(path).set_index("image")
    return {g: p[[f"{g}_p{k}" for k in range(n)]].set_axis(range(n), axis=1) for g, n in GRADES.items()}


def load_kromp(model_dir: Path) -> dict[str, pd.DataFrame]:
    out = {}
    for g, (fname, n_cls) in KROMP_FILES.items():
        d = pd.read_csv(model_dir / fname, header=None).set_index(0)
        means = d.iloc[:, 0::2].iloc[:, :n_cls]          # columns alternate: mean, spread
        if means.isna().all().all():                     # e.g. convnextv2_tiny has no TE probabilities
            out[g] = None
            continue
        means.columns = range(n_cls)
        if g != "EXP":                                   # drop ND, renormalise over A/B/C
            means = means[[0, 1, 2]]
            means = means.div(means.sum(1), axis=0)
        out[g] = means
    return out


def analyse(probs: dict, experts: pd.DataFrame) -> list[dict]:
    rows = []
    for g in GRADES:
        if probs[g] is None:
            continue
        e = experts[experts[f"has_votes_{g}"] & experts[g].notna()]
        if g != "EXP":
            e = e[(e["EXP"] >= 2) & e[g].isin([0, 1, 2])]
        p = probs[g].reindex(e.index)
        unc = 1 - p.max(1)
        pred = p.idxmax(1)
        correct = (pred == e[g].astype(int)).to_numpy()
        cont = e[f"contested_{g}"].to_numpy()
        rho = spearmanr(unc, e[f"disagree_{g}"]).correlation
        auroc = roc_auc_score(cont, unc) if 0 < cont.sum() < len(cont) else np.nan
        rows.append({"grade": g, "n": len(e), "n_contested": int(cont.sum()),
                     "spearman": rho, "auroc": auroc,
                     "acc_unan": correct[~cont].mean(), "acc_cont": correct[cont].mean()})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", type=Path, required=True, help="*_proba.csv from src/linear_probe.py")
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    ap.add_argument("--kromp-dir", default="data/baselines/kromp_predictions", type=Path)
    ap.add_argument("--out", default="results/track_b/uncertainty_vs_agreement.csv", type=Path)
    args = ap.parse_args()

    experts = expert_table(args.labels)
    rows = [dict(r, model="ours (BiomedCLIP baseline)") for r in analyse(load_ours(args.ours), experts)]
    for d in sorted(p for p in args.kromp_dir.iterdir() if p.is_dir()):
        rows += [dict(r, model=d.name) for r in analyse(load_kromp(d), experts)]
    res = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(args.out, index=False)

    for g in GRADES:
        sub = res[res.grade == g].set_index("model")
        n, nc = sub["n"].iloc[0], sub["n_contested"].iloc[0]
        print(f"\n=== {g}: {n} images, {nc} contested ===")
        print(sub[["spearman", "auroc", "acc_unan", "acc_cont"]].round(3).to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
