"""Encode images and Gardner prompts once; save to disk for every later experiment.

Outputs (in --out-dir, default results/embeddings/<model>/):
  blastocyst_images.npy   (N, 512) float32, L2-normalised, rows in labels.csv order
  blastocyst_images.csv   the image names, same order
  captions_images.npy/.csv  the 172 blastocyst-stage caption images (cross-clinic set)
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
def encode_images(model, transform, device, paths, batch_size=64) -> np.ndarray:
    out, t0 = [], time.time()
    use_amp = device == "cuda"
    for i in range(0, len(paths), batch_size):
        batch = torch.stack([transform(Image.open(p)) for p in paths[i:i + batch_size]]).to(device)
        with torch.autocast("cuda", enabled=use_amp):
            e = model.encode_image(batch)
        out.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
        done = min(i + batch_size, len(paths))
        print(f"\r  {done}/{len(paths)} images ({time.time() - t0:.0f}s)", end="", flush=True)
    print()
    return np.concatenate(out).astype(np.float32)


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
    emb = encode_images(model, transform, device,
                        [args.data_dir / "blastocyst" / "Images" / n for n in names], args.batch_size)
    np.save(out_dir / "blastocyst_images.npy", emb)
    pd.Series(names, name="image").to_csv(out_dir / "blastocyst_images.csv", index=False)

    cap_file = args.data_dir / "captions" / "blastocyst_captions.jsonl"
    if cap_file.exists():
        caps = [json.loads(line)["filename"] for line in cap_file.open(encoding="utf-8")][: args.limit]
        cap_dir = args.data_dir / "captions" / "images_2"
        emb = encode_images(model, transform, device, [cap_dir / n for n in caps], args.batch_size)
        np.save(out_dir / "captions_images.npy", emb)
        pd.Series(caps, name="image").to_csv(out_dir / "captions_images.csv", index=False)

    print(f"saved to {out_dir}")


if __name__ == "__main__":
    main()
