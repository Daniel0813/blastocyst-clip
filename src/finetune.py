"""Track A: contrastive image-text fine-tuning of a CLIP model (Phase 6).

Pairs come from two sources:
  captions  real expert captions of blastocyst-stage images from another clinic
            (Gomez / EmbryoScope), excluding the held-out cross-clinic split
  pseudo    Kromp training images with captions generated from their grade labels
            (a fresh random phrasing every epoch); --n-labels picks an
            EXP-stratified subset so label budgets can be compared
  --mode both uses the union.

Loss: symmetric contrastive loss with multiple positives (as in UniCL, Yang et al.
CVPR 2022): pseudo-captioned images with the same grade triple count as matching
pairs, so identical-meaning captions are not pushed apart. Real captions only
match their own image.

Only the last --unfreeze blocks of the image and text encoders (plus their final
norm / projections) are trained. Saves the trained parameters to
<out-dir>/adapter.pt; load them with `src/embed.py --weights`.

Usage:
    python src/finetune.py --data-dir data --mode captions --out-dir results/finetune/a1_seed0
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import clip_model  # noqa: E402
from src.subsets import labelled_subset  # noqa: E402
from src.text import load_bank, pseudo_caption  # noqa: E402


def augment(img: Image.Image, rng: random.Random) -> Image.Image:
    """Embryo images have no canonical orientation: random flips and 90° turns."""
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    k = rng.randrange(4)
    if k:
        img = img.rotate(90 * k, expand=True)
    return img


def caption_pairs(data_dir: Path, limit=None) -> list[dict]:
    rows = [json.loads(line) for line in (data_dir / "captions" / "blastocyst_captions.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if not r["holdout"]][:limit]
    return [{"path": data_dir / "captions" / "images_2" / r["filename"], "text": r["caption"],
             "group": f"cap:{r['filename']}", "grades": None} for r in rows]


def pseudo_pairs(data_dir: Path, n_labels, train_set: str, seed: int, limit=None) -> list[dict]:
    labels = pd.read_csv(data_dir / "blastocyst" / "labels.csv")
    tr = labelled_subset(labels, n_labels, seed, train_set).iloc[:limit]
    return [{"path": data_dir / "blastocyst" / "Images" / r.image, "text": None,
             "group": f"grade:{r.EXP}{r.ICM}{r.TE}", "grades": (int(r.EXP), int(r.ICM), int(r.TE)),
             "image": r.image} for r in tr.itertuples()]


def set_trainable(model, k: int) -> list[torch.nn.Parameter]:
    for p in model.parameters():
        p.requires_grad = False
    v, t = model.visual, model.text
    modules = [*v.trunk.blocks[-k:], v.trunk.norm, v.head, *t.transformer.encoder.layer[-k:], t.proj]
    params = []
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True
            params.append(p)
    return params


def multi_positive_loss(img, txt, groups, scale):
    logits = scale * img @ txt.T
    same = torch.tensor([[a == b for b in groups] for a in groups], device=img.device, dtype=torch.float32)
    target = same / same.sum(1, keepdim=True)
    loss_i = -(target * F.log_softmax(logits, 1)).sum(1).mean()
    loss_t = -(target.T * F.log_softmax(logits.T, 1)).sum(1).mean()
    return (loss_i + loss_t) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--model", default="biomedclip")
    ap.add_argument("--mode", choices=["captions", "pseudo", "both"], default="captions")
    ap.add_argument("--n-labels", type=int, help="pseudo/both: number of labelled Kromp training images")
    ap.add_argument("--train-set", choices=["official", "patient_level"], default="official")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--unfreeze", type=int, default=2, help="last N blocks of each encoder to train")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only N pairs per source (smoke test)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    if args.mode in ("captions", "both"):
        pairs += caption_pairs(args.data_dir, args.limit)
    if args.mode in ("pseudo", "both"):
        pairs += pseudo_pairs(args.data_dir, args.n_labels, args.train_set, args.seed, args.limit)
        used = [p["image"] for p in pairs if p.get("image")]
        pd.Series(used, name="image").to_csv(args.out_dir / "labelled_images.csv", index=False)
    print(f"{len(pairs)} image-text pairs ({args.mode})")

    model, transform, tokenize, device = clip_model.load(args.model)
    params = set_trainable(model, args.unfreeze)
    print(f"model on {device}; training {sum(p.numel() for p in params) / 1e6:.1f}M "
          f"of {sum(p.numel() for p in model.parameters()) / 1e6:.0f}M parameters")
    bank = load_bank()
    scale = model.logit_scale.exp().detach()

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(pairs) / args.batch_size)
    total, warmup = steps_per_epoch * args.epochs, max(1, steps_per_epoch * args.epochs // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    history, t0 = [], time.time()
    model.train()
    for epoch in range(args.epochs):
        order = list(range(len(pairs)))
        rng.shuffle(order)
        losses = []
        for b in range(0, len(order), args.batch_size):
            batch = [pairs[i] for i in order[b:b + args.batch_size]]
            if len(batch) < 2:
                continue
            images = torch.stack([transform(augment(Image.open(p["path"]), rng)) for p in batch]).to(device)
            texts = [p["text"] if p["text"] is not None else pseudo_caption(bank, *p["grades"], rng) for p in batch]
            tokens = tokenize(texts).to(device)
            with torch.autocast("cuda", enabled=use_amp):
                img = F.normalize(model.encode_image(images), dim=-1)
                txt = F.normalize(model.encode_text(tokens), dim=-1)
                loss = multi_positive_loss(img.float(), txt.float(), [p["group"] for p in batch], scale)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(loss.item())
        history.append(float(np.mean(losses)))
        print(f"  epoch {epoch + 1:2d}/{args.epochs}  loss {history[-1]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(trained, args.out_dir / "adapter.pt")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(n_pairs=len(pairs), loss_history=history, device=device)
    json.dump(config, open(args.out_dir / "config.json", "w"), indent=2)
    print(f"saved {len(trained)} tensors to {args.out_dir / 'adapter.pt'}")


if __name__ == "__main__":
    main()
