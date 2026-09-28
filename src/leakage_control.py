"""Is the patient-level drop leakage, or just fewer training images?

The patient-level split removes the 629 training images of the 200 patients who
also appear in the test set. This control removes the same number of images by
dropping randomly chosen *other* patients (whole patients, so the grouping is
identical), repeated over several seeds. Same training size, but no leakage
removed.

  control ~ official      -> the drop is leakage
  control ~ patient-level -> the drop is just less training data

Uses the baseline configuration of src/linear_probe.py.

Usage:
    python src/leakage_control.py --emb-dir results/embeddings/biomedclip_full \
        --feature cls+patchmean --select f1_macro --seeds 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import linear_probe as lp  # noqa: E402


def random_patient_drop(labels: pd.DataFrame, n_images: int, rng) -> pd.Series:
    """Training mask with randomly chosen non-overlap patients removed (~n_images)."""
    tr = labels[(labels.split == "train") & ~labels.patient_overlap]
    sizes = tr.groupby("patient_id").size()
    order = rng.permutation(sizes.index.to_numpy())
    dropped, total = [], 0
    for pid in order:
        if total >= n_images:
            break
        dropped.append(pid)
        total += sizes[pid]
    return (labels.split == "train") & ~labels.patient_id.isin(dropped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="results/embeddings/biomedclip_full", type=Path)
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    ap.add_argument("--feature", default="cls+patchmean", choices=list(lp.FEATURES))
    ap.add_argument("--select", default="f1_macro", choices=["acc", "f1_macro"])
    ap.add_argument("--seeds", type=int, default=5)
    args = ap.parse_args()
    lp.args_labels, lp.SELECT = args.labels, args.select

    labels = pd.read_csv(args.labels)
    names = pd.read_csv(args.emb_dir / "blastocyst_images.csv")["image"]
    assert names.tolist() == labels["image"].tolist()
    emb = np.concatenate([np.load(args.emb_dir / f"blastocyst_images{s}.npy")
                          for s in lp.FEATURES[args.feature]], axis=1)
    out_dir = Path("results/leakage_control") / f"{args.emb_dir.name}_{args.feature}_{args.select}"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_drop = int(labels.patient_overlap.sum())
    frames = []
    for seed in range(args.seeds):
        mask = random_patient_drop(labels, n_drop, np.random.default_rng(seed))
        print(f"\nseed {seed}: dropped {int((labels.split == 'train').sum() - mask.sum())} "
              f"random non-overlap images, training on {int(mask.sum())}")
        frames.append(lp.run(emb, labels, mask, f"control_seed{seed}", out_dir).assign(seed=seed))
    res = pd.concat(frames, ignore_index=True)
    res.to_csv(out_dir / "results.csv", index=False)

    ref_dir = Path("results/linear_probe") / (f"{args.emb_dir.name}_{args.feature}"
                                              + ("" if args.select == "acc" else f"_{args.select}"))
    ref = pd.read_csv(ref_dir / "results.csv")
    for protocol in ("clean", "kromp"):
        print(f"\n=== {protocol} protocol, accuracy (macro F1) ===")
        for ts in ("official", "patient_level"):
            r = ref[(ref.protocol == protocol) & (ref.train_set == ts)].set_index("target")
            print(f"  {ts:24s} " + "   ".join(f"{t} {r.loc[t, 'acc']:.3f} ({r.loc[t, 'f1_macro']:.3f})"
                                             for t in ("EXP", "ICM", "TE")))
        c = res[res.protocol == protocol].groupby("target")
        m, s = c["acc"].mean(), c["acc"].std()
        f = c["f1_macro"].mean()
        print(f"  {'control (mean ± sd, ' + str(args.seeds) + ' seeds)':24s} "
              + "   ".join(f"{t} {m[t]:.3f}±{s[t]:.3f} ({f[t]:.3f})" for t in ("EXP", "ICM", "TE")))
    print(f"\nwrote {out_dir}/")


if __name__ == "__main__":
    main()
