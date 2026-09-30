"""Zero-shot Gardner grading on the expert-caption images (Gomez / EmbryoScope).

Uses the grades stated in the caption text as labels (noisy, regex-extracted).
Reports the held-out captions (never used for fine-tuning) and, separately, the
training captions, so an adapted model can be checked for in-domain learning
versus memorisation. Compare several embedding folders side by side.

Usage:
    python src/caption_zeroshot.py results/embeddings/biomedclip_full results/embeddings/biomedclip_a1_seed0
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

LETTER = {"A": 0, "B": 1, "C": 2}


def load_caption_labels(captions_file: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in captions_file.open(encoding="utf-8")]
    df = pd.DataFrame(rows).set_index("filename")
    df["EXP"] = df["exp_grade"].map(lambda g: g - 1 if pd.notna(g) and 1 <= g <= 5 else np.nan)
    df["ICM"] = df["icm_grade"].map(LETTER)
    df["TE"] = df["te_grade"].map(LETTER)
    return df


def zero_shot(emb_dir: Path) -> pd.DataFrame:
    img = np.load(emb_dir / "captions_images.npy")
    names = pd.read_csv(emb_dir / "captions_images.csv")["image"]
    prompts = np.load(emb_dir / "prompts.npz")
    pred = pd.DataFrame(index=names)
    for g in ("EXP", "ICM", "TE"):
        pred[g] = prompts[f"{g}_codes"][(img @ prompts[f"{g}_emb"].T).argmax(1)]
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("emb_dirs", nargs="+", type=Path)
    ap.add_argument("--captions", default="data/captions/blastocyst_captions.jsonl", type=Path)
    args = ap.parse_args()

    labels = load_caption_labels(args.captions)
    rows = []
    for emb_dir in args.emb_dirs:
        pred = zero_shot(emb_dir)
        for part, mask in (("held-out", labels["holdout"]), ("training", ~labels["holdout"])):
            for g in ("EXP", "ICM", "TE"):
                ok = mask & labels[g].notna()
                y = labels.loc[ok, g].astype(int)
                p = pred.loc[y.index, g]
                majority = (y == y.mode()[0]).mean() if len(y) else np.nan
                rows.append({"model": emb_dir.name, "captions": part, "grade": g, "n": len(y),
                             "acc": (p.to_numpy() == y.to_numpy()).mean(),
                             "f1_macro": f1_score(y, p, average="macro", zero_division=0),
                             "majority_acc": majority})
    res = pd.DataFrame(rows)
    for part in ("held-out", "training"):
        sub = res[res.captions == part]
        n = sub.groupby("grade")["n"].first()
        print(f"\n=== {part} captions (n per grade: EXP {n['EXP']}, ICM {n['ICM']}, TE {n['TE']}) — accuracy (macro F1) ===")
        for model, r in sub.groupby("model", sort=False):
            r = r.set_index("grade")
            print(f"  {model:24s} " + "   ".join(f"{g} {r.loc[g, 'acc']:.3f} ({r.loc[g, 'f1_macro']:.3f})"
                                             for g in ("EXP", "ICM", "TE")))
        r = sub.drop_duplicates("grade").set_index("grade")
        print(f"  {'majority-grade guess':24s} " + "   ".join(f"{g} {r.loc[g, 'majority_acc']:.3f}" for g in ("EXP", "ICM", "TE")))


if __name__ == "__main__":
    main()
