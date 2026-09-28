"""Encode images and Gardner prompts once; save to disk for every later experiment.

Outputs (in --out-dir, default results/embeddings/<model>/):
  blastocyst_images.npy   (N, 512) float32, L2-normalised CLIP embedding, rows in labels.csv order
  blastocyst_images_cls.npy        (N, 768) CLS token before the 512-d projection
  blastocyst_images_patchmean.npy  (N, 768) mean of the 196 patch tokens (local detail)
  blastocyst_images.csv   the image names, same order
  captions_images*.npy/.csv  same three feature types for the 172 blastocyst-stage
                             caption images (cross-clinic set)
  prompts.npz             per grade: class codes + mean prompt embedding per class

Usage (local CPU or Colab GPU):
    python src/embed.py --data-dir data --model biomedclip [--limit 16]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import clip_model  # noqa: E402
from src.text import class_prompts, load_bank  # noqa: E402


@torch.no_grad()
def encode_images(model, transform, device, paths, batch_size=64) -> dict[str, np.ndarray]:
    """Returns {"": 512-d CLIP embedding, "_cls": 768-d CLS, "_patchmean": 768-d patch mean}.

    For timm-based models (BiomedCLIP) the CLS and patch tokens are read from the
    vision trunk; the 512-d embedding is the trunk's CLS passed through the head,
    identical to model.encode_image.
    """
    out, t0 = {"": [], "_cls": [], "_patchmean": []}, time.time()
    use_amp = device == "cuda"
    visual = model.visual
    n_prefix = getattr(visual.trunk, "num_prefix_tokens", 1)
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([transform(Image.open(p)) for p in paths[i:i + batch_size]]).to(device)
        with torch.autocast("cuda", enabled=use_amp):
            tokens = visual.trunk.forward_features(batch)            # (B, 1 + 196, 768)
            cls = visual.trunk.forward_head(tokens, pre_logits=True)  # CLS token
            proj = visual.head(cls)
        out[""].append(torch.nn.functional.normalize(proj.float(), dim=-1).cpu().numpy())
        out["_cls"].append(cls.float().cpu().numpy())
        out["_patchmean"].append(tokens[:, n_prefix:].float().mean(1).cpu().numpy())
        done = min(i + batch_size, len(paths))
        print(f"\r  {done}/{len(paths)} images ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return {k: np.concatenate(v).astype(np.float32) for k, v in out.items()}


def save_features(out_dir: Path, stem: str, feats: dict, names: list[str]):
    for suffix, arr in feats.items():
        np.save(out_dir / f"{stem}{suffix}.npy", arr)
    pd.Series(names, name="image").to_csv(out_dir / f"{stem}.csv", index=False)


@torch.no_grad()
def encode_prompts(model, tokenize, device) -> dict:
    bank, out = load_bank(), {}
    for grade in ("EXP", "ICM", "TE"):
        codes, means = [], []
        for code, texts in sorted(class_prompts(bank, grade).items()):
            e = model.encode_text(tokenize(texts).to(device)).float()
            e = torch.nn.functional.normalize(e, dim=-1).mean(0)    # prompt ensembling
            means.append(torch.nn.functional.normalize(e, dim=-1).cpu().numpy())
            codes.append(code)
        out[f"{grade}_codes"] = np.array(codes)
        out[f"{grade}_emb"] = np.stack(means).astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--model", default="biomedclip")
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--limit", type=int, help="encode only the first N images (smoke test)")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    out_dir = args.out_dir or Path("results/embeddings") / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    model, transform, tokenize, device = clip_model.load(args.model)
    print(f"model {args.model} on {device}")

    np.savez(out_dir / "prompts.npz", **encode_prompts(model, tokenize, device))
    print("  prompts encoded")

    labels = pd.read_csv(args.data_dir / "blastocyst" / "labels.csv")
    names = labels["image"].tolist()[: args.limit]
    feats = encode_images(model, transform, device,
                          [args.data_dir / "blastocyst" / "Images" / n for n in names], args.batch_size)
    save_features(out_dir, "blastocyst_images", feats, names)

    cap_file = args.data_dir / "captions" / "blastocyst_captions.jsonl"
    if cap_file.exists():
        caps = [json.loads(line)["filename"] for line in cap_file.open(encoding="utf-8")][: args.limit]
        cap_dir = args.data_dir / "captions" / "images_2"
        feats = encode_images(model, transform, device, [cap_dir / n for n in caps], args.batch_size)
        save_features(out_dir, "captions_images", feats, caps)

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
