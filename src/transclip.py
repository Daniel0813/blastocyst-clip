"""TransCLIP (Zanella et al., NeurIPS 2024): transductive inference on saved embeddings.

All 300 test images are classified jointly: a Gaussian mixture over their image
features, a smoothing term over each image's 3 nearest test neighbours, and a KL
term that keeps the assignment close to a prior y_hat. No encoder is trained and
no test label is used.

Port of the authors' solver (github.com/MaxZanella/transduction-for-vlms,
TransCLIP_solver/), zero-shot / "on-top" mode: no support set, lambda = 1,
10 outer iterations, 5 inner z-iterations, 3 neighbours, mu initialised from the
8 most confident images per class. The prior is
  text   prompt-similarity softmax (TransCLIP-ZS; frozen BiomedCLIP zero-shot)
  heads  the linear heads' class probabilities ("on top" of the
         label-efficiency A0 heads: same N labelled images, same C selection)
ICM/TE are solved over the test images whose transductive EXP is grade >= 3; the
ND rule fills in the rest.

Mixture components are equal-weight (the paper's "balanced mixture"), which
matters here: the test grades are far from balanced.

--validate runs the same thing on the 5 grouped training folds instead of the test
set (heads on 4 folds, transduction on the 5th) to choose the feature space for
the mixture without looking at the test set.

Usage:
    python src/transclip.py --validate
    python src/transclip.py --prior heads --trans-feature proj --n 50 100 200 500 all --seeds 0 1 2
    python src/transclip.py --prior text [--emb-dir <adapted model's embeddings>]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import evaluate  # noqa: E402
from src import label_efficiency as le  # noqa: E402
from src import linear_probe as lp  # noqa: E402
from src.subsets import labelled_subset  # noqa: E402

ND = 3
MAX_ITER, Z_ITER, N_NEIGHBORS, N_CONFIDENT, LAMBDA = 10, 5, 3, 8, 1.0


def knn_affinity(x: torch.Tensor, k: int) -> torch.Tensor:
    sim = x @ x.T
    idx = sim.topk(k + 1, dim=1).indices[:, 1:]
    rows = torch.arange(len(x)).repeat_interleave(k)
    W = torch.zeros_like(sim)
    W[rows, idx.flatten()] = sim[rows, idx.flatten()]
    return W


def solve(feats: np.ndarray, prior: np.ndarray) -> np.ndarray:
    """feats (N, d) L2-normalised; prior (N, K) probabilities. Returns z (N, K)."""
    x = torch.tensor(feats, dtype=torch.float64)
    y_hat = torch.tensor(prior, dtype=torch.float64)
    K, d = y_hat.shape[1], x.shape[1]
    z = y_hat.clone()

    top = z.topk(min(N_CONFIDENT, len(x)), dim=0).indices          # (8, K)
    mu = torch.stack([(x[top[:, c]] * z[top[:, c], c:c + 1]).mean(0) for c in range(K)])
    mu = F.normalize(mu, dim=-1)
    std = torch.full((d,), 1.0 / d, dtype=torch.float64)
    W = knn_affinity(x, N_NEIGHBORS)

    for it in range(MAX_ITER + 1):
        loglik = -0.5 * (((x[:, None, :] - mu[None]) ** 2) / std).sum(-1)    # (N, K)
        for _ in range(Z_ITER):
            inter = loglik + (50 / (N_NEIGHBORS * 2)) * (W.T @ z + W @ z)
            inter = inter - inter.max(1, keepdim=True).values
            inter = (y_hat ** LAMBDA) * torch.exp(inter / 50)
            z = inter / inter.sum(1, keepdim=True)
        if it == MAX_ITER:
            break
        mu = F.normalize((z.T @ x) / z.sum(0)[:, None], dim=-1)
        std = torch.einsum("nk,nkd->d", z, (x[:, None, :] - mu[None]) ** 2) / z.sum()
    return z.numpy()


def head_proba(head, X) -> tuple[np.ndarray, np.ndarray]:
    """Class probabilities over the classes the head knows, and those class codes."""
    if isinstance(head, le.Constant):
        return np.ones((len(X), 1)), np.array([head.value])
    return head.predict_proba(X), head.classes_


def transduce(head, X, F_):
    """(baseline prediction, TransCLIP prediction) for one head."""
    proba, classes = head_proba(head, X)
    base = classes[proba.argmax(1)]
    if len(classes) < 2 or len(X) <= N_NEIGHBORS:
        return base, base
    z = solve(F_, proba)
    return base, classes[z.argmax(1)]


def trans_features(labels, name, emb_dir) -> np.ndarray:
    """Features the mixture and neighbour graph live in, L2-normalised."""
    if name == "proj":
        return np.load(emb_dir / "blastocyst_images.npy")
    X = np.concatenate([np.load(emb_dir / f"blastocyst_images{s}.npy") for s in lp.FEATURES[name]], axis=1)
    tr = (labels["split"] == "train").to_numpy()
    X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
    return X / np.linalg.norm(X, axis=1, keepdims=True)


def fit_heads(X, sub):
    folds = sub["fold"].to_numpy()
    heads = {"EXP": le.tune_and_fit(X, sub["EXP"].to_numpy(), folds)[0]}
    later = (sub["EXP"] >= 2).to_numpy()
    for g in ("ICM", "TE"):
        keep = later & sub[g].isin([0, 1, 2]).to_numpy()
        heads[g] = le.tune_and_fit(X[keep], sub[g].to_numpy()[keep], folds[keep])[0]
    return heads


def predict_heads(heads, X_te, F_te):
    """Baseline and TransCLIP predictions for the test rows, ND rule applied."""
    base_exp, tc_exp = transduce(heads["EXP"], X_te, F_te)
    out = {"base": {"EXP": base_exp}, "trans": {"EXP": tc_exp}}
    for kind, exp in (("base", base_exp), ("trans", tc_exp)):
        later = exp >= 2
        for g in ("ICM", "TE"):
            pred = np.full(len(X_te), ND)
            if later.any():
                b, t = transduce(heads[g], X_te[later], F_te[later])
                pred[later] = b if kind == "base" else t
            out[kind][g] = pred
    return out


def predict_text(emb_dir, F_te, test_rows):
    """TransCLIP-ZS: prior = softmax of 100 x cosine to the averaged Gardner prompts."""
    img = np.load(emb_dir / "blastocyst_images.npy")[test_rows]
    prompts = np.load(emb_dir / "prompts.npz")

    def prior(g, rows):
        logits = 100 * img[rows] @ prompts[f"{g}_emb"].T
        return torch.softmax(torch.tensor(logits), 1).numpy(), prompts[f"{g}_codes"]

    out = {"base": {}, "trans": {}}
    p, codes = prior("EXP", slice(None))
    out["base"]["EXP"], out["trans"]["EXP"] = codes[p.argmax(1)], codes[solve(F_te, p).argmax(1)]
    for kind in ("base", "trans"):
        later = out[kind]["EXP"] >= 2
        for g in ("ICM", "TE"):
            pred = np.full(len(img), ND)
            p, codes = prior(g, later)
            pred[later] = codes[(p if kind == "base" else solve(F_te[later], p)).argmax(1)]
            out[kind][g] = pred
    return out


def score(preds, names, path, gold):
    df = pd.DataFrame({"image": names, **preds})
    df.to_csv(path, header=False, index=False)
    return evaluate.evaluate(path, gold)


def validate(labels, X, feats: dict):
    """Heads on 4 training folds, transduction on the 5th; per-head macro F1 / accuracy."""
    tr = labels[labels.split == "train"].reset_index(drop=True)
    rows = (labels["split"] == "train").to_numpy()
    Xtr = X[rows]
    Ftr = {k: v[rows] for k, v in feats.items()}
    recs = []
    for k in sorted(tr["fold"].unique()):
        fit, held = (tr["fold"] != k).to_numpy(), (tr["fold"] == k).to_numpy()
        heads = fit_heads(Xtr[fit], tr[fit])
        for g in ("EXP", "ICM", "TE"):
            m = held & ((tr["EXP"] >= 2) & tr[g].isin([0, 1, 2])).to_numpy() if g != "EXP" else held
            y = tr.loc[m, g].to_numpy()
            for fname, Fv in Ftr.items():
                base, trans = transduce(heads[g], Xtr[m], Fv[m])
                for kind, p in (("heads only", base), (f"+TransCLIP ({fname})", trans)):
                    recs.append({"fold": k, "grade": g, "method": kind, "acc": (p == y).mean(),
                                 "f1_macro": f1_score(y, p, average="macro", zero_division=0)})
        print(f"  fold {k} done", flush=True)
    df = pd.DataFrame(recs).drop_duplicates(["fold", "grade", "method"])
    print("\n=== validation on held-out training folds (mean of 5 folds) ===")
    print(df.groupby(["grade", "method"])[["acc", "f1_macro"]].mean().round(3).to_string())
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb-dir", default="results/embeddings/biomedclip_full", type=Path)
    ap.add_argument("--labels", default="data/blastocyst/labels.csv", type=Path)
    ap.add_argument("--prior", choices=["heads", "text"], default="heads")
    ap.add_argument("--trans-feature", choices=["proj", "cls+patchmean"], default="proj")
    ap.add_argument("--n", nargs="+", default=["50", "100", "200", "500", "all"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--train-set", choices=["official", "patient_level"], default="official")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--out-dir", default="results/transclip", type=Path)
    args = ap.parse_args()
    lp.args_labels = args.labels
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(args.labels)
    names = pd.read_csv(args.emb_dir / "blastocyst_images.csv")["image"]
    assert names.tolist() == labels["image"].tolist(), "embedding order != labels.csv order"
    X = np.concatenate([np.load(args.emb_dir / f"blastocyst_images{s}.npy")
                        for s in lp.FEATURES["cls+patchmean"]], axis=1)
    test = (labels["split"] == "test").to_numpy()
    gold = evaluate.load_gold(args.labels)

    if args.validate:
        feats = {f: trans_features(labels, f, args.emb_dir) for f in ("proj", "cls+patchmean")}
        validate(labels, X, feats).to_csv(args.out_dir / "validation.csv", index=False)
        return

    F_all = trans_features(labels, args.trans_feature, args.emb_dir)
    test_names = labels.loc[test, "image"].to_numpy()
    pred_dir = args.out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)
    tag = f"{args.prior}_{args.emb_dir.name}_{args.train_set}_{args.trans_feature}"
    frames = []

    if args.prior == "text":
        out = predict_text(args.emb_dir, F_all[test], test)
        for kind, cond in (("base", "zero-shot"), ("trans", "TransCLIP-ZS")):
            res = score(out[kind], test_names, pred_dir / f"{tag}_{kind}_pred.csv", gold)
            frames.append(res.assign(condition=cond, n=0, seed=0))
    else:
        for n in args.n:
            n_val = None if n == "all" else int(n)
            for seed in (args.seeds if n != "all" else args.seeds[:1]):
                sub = labelled_subset(labels, n_val, seed, args.train_set)
                heads = fit_heads(X[labels.index.get_indexer(sub.index)], sub)
                out = predict_heads(heads, X[test], F_all[test])
                for kind, cond in (("base", "A0"), ("trans", "A0+TransCLIP")):
                    path = pred_dir / f"{tag}_{cond}_n{len(sub)}_seed{seed}_pred.csv"
                    frames.append(score(out[kind], test_names, path, gold).assign(condition=cond, n=len(sub),
                                                                                    seed=seed))
                last = pd.concat(frames[-2:])
                c = last[(last.protocol == "clean") & last.target.isin(["EXP", "ICM", "TE"])]
                print(f"  n={len(sub):4d} seed {seed}: " + "   ".join(
                    f"{cond} " + " ".join(f"{r.acc:.3f}/{r.f1_macro:.3f}" for r in c[c.condition == cond].itertuples())
                    for cond in ("A0", "A0+TransCLIP")), flush=True)

    res = pd.concat(frames, ignore_index=True).assign(prior=args.prior, trans_feature=args.trans_feature,
                                                      train_set=args.train_set)
    path = args.out_dir / f"{tag}.csv"
    res.to_csv(path, index=False)
    summ = (res[(res.protocol == "clean") & res.target.isin(["EXP", "ICM", "TE"])]
            .groupby(["condition", "n", "target"])[["acc", "f1_macro"]].mean().unstack("target").round(3))
    print("\n=== clean protocol, mean over seeds: accuracy | macro F1 ===")
    print(summ.to_string())
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
