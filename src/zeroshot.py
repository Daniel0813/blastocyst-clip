"""E0: zero-shot Gardner grading from saved embeddings (no training).

Each image gets, per grade, the class whose averaged prompt embedding is most
similar (cosine). Then the ND rule: predicted EXP grade 1-2 -> ICM = TE = ND.
Writes predictions in the evaluate.py format and scores them.

Usage:
    python src/zeroshot.py --emb-dir results/embeddings/biomedclip
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate  # noqa: E402

ND = 3


def predict(emb_dir: Path) -> pd.DataFrame:
    img = np.load(emb_dir / "blastocyst_images.npy")
    names = pd.read_csv(emb_dir / "blastocyst_images.csv")["image"]
    prompts = np.load(emb_dir / "prompts.npz")
    pred = pd.DataFrame({"image": names})
    for g in ("EXP", "ICM", "TE"):
        sims = img @ prompts[f"{g}_emb"].T
        pred[g] = prompts[f"{g}_codes"][sims.argmax(1)]
    early = pred["EXP"] <= 1
    pred.loc[early, ["ICM", "TE"]] = ND
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="results/embeddings/biomedclip", type=Path)
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    args = ap.parse_args()

    pred = predict(args.emb_dir)
    out = args.emb_dir.parent.parent / "zeroshot" / f"{args.emb_dir.name}_pred.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels)
    test = pred[pred["image"].isin(labels.loc[labels.split == "test", "image"])]
    test.to_csv(out, header=False, index=False)
    print(f"predicted classes on test (EXP): {test['EXP'].value_counts().sort_index().to_dict()}")

    gold = evaluate.load_gold(args.labels)
    if len(test) < len(gold):
        print(f"note: only {len(test)} of {len(gold)} test images embedded (smoke test?)")
        gold = gold.loc[gold.index.isin(test["image"])]
    res = evaluate.evaluate(out, gold)
    print(res.pivot(index="protocol", columns="target", values="acc").round(3).to_string())
    print("macro F1:")
    print(res.pivot(index="protocol", columns="target", values="f1_macro").round(3).to_string())


if __name__ == "__main__":
    main()
