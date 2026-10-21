"""Track A, condition A3: semi-supervised CLIP adaptation (SemiCLIP, Gan et al., ICLR 2025).

Data, for a label budget N and seed:
  labelled pairs  the 119 real expert captions + pseudo-captions of the N labelled Kromp
                  images (exactly the pairs A2 trains on)
  unlabelled      the other Kromp training images; their grade labels are never used
                  (only to report how accurate the pseudo-labels were)

Stage 1, supervised (--epochs1 passes over the labelled pairs):
    L = L_CLIP (multi-positive, as in A2) + L_SCM (concept classifier on image embeddings)
Pseudo-labels: concepts for every unlabelled image from the concept classifier, fixed after stage 1.
Stage 2, semi-supervised (--epochs2 passes over the unlabelled images; each step also draws a
batch of labelled pairs):
    L = L_SCM^u   strongly augmented unlabelled image -> its pseudo-concepts
      + L_Trap    L_CLIP on the labelled pairs (upper base of the trapezoid) + equal diagonals
                  and equal legs over the labelled pairs and the top-P% unlabelled images paired
                  with learnable surrogate captions built from their pseudo-concepts

Changes from the paper, each forced by this data:
  - Concepts: 12 Gardner grade concepts (EXP 1-5, ICM A-C, TE A-C, ICM/TE not defined) instead
    of nouns mined with NLTK; in the expert captions the grading words are adjectives and grade
    codes, which noun mining drops. Caption images get the concepts of the grades they state,
    Kromp images those of their labels.
  - Pseudo-labels: the most likely concept within each grade (EXP; then ICM and TE, or "not
    defined" for EXP 1-2) instead of a flat top-k, so no image gets two expansion grades.
  - Only the last --unfreeze blocks of each encoder train (as in A2), not the whole model.

--no-text is the control: same trainable blocks, concept classifier, pseudo-labels, strong
augmentation and schedule, but no text anywhere (stage 1 = L_SCM; stage 2 = L_SCM^u + L_SCM on
the labelled images; the classifier starts random rather than from concept text embeddings).
If A3 beats A2 but not this control, the gain came from the unlabelled images, not language.

Saves adapter.pt (encoder weights, loadable by src/embed.py --weights), heads.pt (concept
classifier and prompt vectors), pseudo_labels.csv and config.json to --out-dir.

Usage:
    python src/semiclip.py --data-dir data --n-labels 100 --seed 0 --out-dir results/finetune/a3_n100_seed0
    python src/semiclip.py ... --no-text
"""
import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import open_clip
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import RandAugment

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import clip_model  # noqa: E402
from src.finetune import augment, caption_pairs, multi_positive_loss, pseudo_pairs, set_trainable  # noqa: E402
from src.subsets import labelled_subset  # noqa: E402
from src.text import load_bank, pseudo_caption  # noqa: E402

ND = 3
CONCEPTS = [
    "an early blastocyst", "a blastocyst with a cavity filling half the embryo", "a full blastocyst",
    "an expanded blastocyst", "a hatching blastocyst",
    "a compact inner cell mass of many cells", "a loose inner cell mass of several cells",
    "an inner cell mass of very few cells",
    "a cohesive trophectoderm of many cells", "a loose trophectoderm of few cells",
    "a trophectoderm of very few large cells",
    "an inner cell mass and trophectoderm that are not yet distinguishable",
]
EXP_C, ICM_C, TE_C, ND_C = range(0, 5), range(5, 8), range(8, 11), 11
PROMPT_INIT = "a photo includes"


def grade_concepts(exp, icm, te) -> list[int]:
    """Concept indices for a grade triple (codes as in labels.csv; None = not stated)."""
    out = [] if exp is None else [EXP_C[exp]]
    if exp is not None and exp <= 1:
        return out + [ND_C]
    if icm is not None and icm != ND:
        out.append(ICM_C[icm])
    if te is not None and te != ND:
        out.append(TE_C[te])
    return out


def caption_concepts(data_dir: Path) -> dict:
    """{caption filename: concept indices} from the grades the caption states."""
    abc = {"A": 0, "B": 1, "C": 2}
    out = {}
    for line in (data_dir / "captions" / "blastocyst_captions.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        exp = None if r["exp_grade"] is None else min(int(r["exp_grade"]), 5) - 1
        out[r["filename"]] = grade_concepts(exp, abc.get(r["icm_grade"]), abc.get(r["te_grade"]))
    return out


def multi_hot(idx_lists, device) -> torch.Tensor:
    y = torch.zeros(len(idx_lists), len(CONCEPTS), device=device)
    for i, idx in enumerate(idx_lists):
        y[i, idx] = 1
    return y


def scm_loss(logits, y) -> torch.Tensor:
    """Soft cross-entropy against the normalised multi-hot concepts (rows without concepts skipped)."""
    keep = y.sum(1) > 0
    if not keep.any():
        return logits.sum() * 0
    target = y[keep] / y[keep].sum(1, keepdim=True)
    return -(target * F.log_softmax(logits[keep], 1)).sum(1).mean()


class ConceptHead(torch.nn.Module):
    """Concept classifier psi: cosine similarity to one learnable vector per concept."""

    def __init__(self, init: torch.Tensor, scale: float):
        super().__init__()
        self.weight = torch.nn.Parameter(init.clone())
        self.scale = scale

    def forward(self, img_emb):
        return self.scale * F.normalize(img_emb, dim=-1) @ F.normalize(self.weight, dim=-1).T


class SurrogateCaptions(torch.nn.Module):
    """Learnable surrogate captions: [CLS] [V]1..n concept1 [V]1..n concept2 ... [SEP].

    One set of n prompt vectors per concept slot, shared by all images, initialised from the
    token embeddings of PROMPT_INIT (as in the paper's eq. 4).
    """

    def __init__(self, text_encoder, hf_tokenizer, slots: int = 3):
        super().__init__()
        self.tok = hf_tokenizer
        words = text_encoder.transformer.embeddings.word_embeddings
        init_ids = torch.tensor(hf_tokenizer(PROMPT_INIT, add_special_tokens=False)["input_ids"])
        self.ctx = torch.nn.Parameter(words.weight[init_ids].detach().clone().repeat(slots, 1, 1))
        self.concept_ids = [hf_tokenizer(c, add_special_tokens=False)["input_ids"] for c in CONCEPTS]

    def forward(self, text_encoder, concept_lists) -> torch.Tensor:
        words = text_encoder.transformer.embeddings.word_embeddings
        dev = words.weight.device
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        seqs = []
        for concepts in concept_lists:
            parts = [words(torch.tensor([cls], device=dev))]
            for slot, c in enumerate(concepts):
                parts += [self.ctx[slot], words(torch.tensor(self.concept_ids[c], device=dev))]
            parts.append(words(torch.tensor([sep], device=dev)))
            seqs.append(torch.cat(parts))
        length = max(len(s) for s in seqs)
        embeds = torch.zeros(len(seqs), length, seqs[0].shape[1], device=dev, dtype=seqs[0].dtype)
        mask = torch.zeros(len(seqs), length, dtype=torch.long, device=dev)
        for i, s in enumerate(seqs):
            embeds[i, :len(s)], mask[i, :len(s)] = s, 1
        out = text_encoder.transformer(inputs_embeds=embeds, attention_mask=mask)
        return text_encoder.proj(text_encoder.pooler(out, mask))


def trapezoid_loss(img_l, txt_l, groups, img_u, txt_u, scale) -> torch.Tensor:
    """Eq. 5: CLIP loss on the labelled pairs + (diagonals) + (legs) over all pairs."""
    upper = multi_positive_loss(img_l, txt_l, groups, scale)
    img, txt = torch.cat([img_l, img_u]), torch.cat([txt_l, txt_u])
    sim = img @ txt.T
    diagonals = ((sim - sim.T) ** 2).sum()
    legs = ((img @ img.T - txt @ txt.T) ** 2).sum()
    return upper + (diagonals + legs) / len(img)


def load_image(path, rng, strong=None):
    img = augment(Image.open(path).convert("RGB"), rng)
    return strong(img) if strong is not None else img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", type=Path)
    ap.add_argument("--model", default="biomedclip")
    ap.add_argument("--n-labels", type=int, required=True, help="labelled Kromp training images")
    ap.add_argument("--train-set", choices=["official", "patient_level"], default="official")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--no-text", action="store_true", help="control: no captions, no text loss")
    ap.add_argument("--epochs1", type=int, default=20, help="stage 1 passes over the labelled pairs")
    ap.add_argument("--epochs2", type=int, default=5, help="stage 2 passes over the unlabelled images")
    ap.add_argument("--batch-size", type=int, default=32, help="labelled pairs per step (and unlabelled images)")
    ap.add_argument("--lr", type=float, default=2e-5, help="encoder blocks")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="concept classifier and prompt vectors")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--top-p", type=float, default=0.3, help="share of unlabelled images in the trapezoid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="use only N items per source (smoke test)")
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    use_text = not args.no_text

    # ---- data
    labelled = pseudo_pairs(args.data_dir, args.n_labels, args.train_set, args.seed, args.limit)
    for p in labelled:
        p["concepts"] = grade_concepts(*p["grades"])
    if use_text:
        cap_concepts = caption_concepts(args.data_dir)
        caps = caption_pairs(args.data_dir, args.limit)
        for p in caps:
            p["concepts"] = cap_concepts[p["path"].name]
        labelled += caps
    labels = pd.read_csv(args.data_dir / "blastocyst" / "labels.csv")
    pool = labels[labels["split"] == "train"]
    if args.train_set == "patient_level":
        pool = pool[~pool["patient_overlap"]]
    used = set(labelled_subset(labels, args.n_labels, args.seed, args.train_set)["image"])
    unlab = pool[~pool["image"].isin(used)].iloc[:args.limit]
    assert len(unlab), "no unlabelled images: A3 needs N below the training-set size"
    unlab_paths = [args.data_dir / "blastocyst" / "Images" / n for n in unlab["image"]]
    pd.Series(sorted(used), name="image").to_csv(args.out_dir / "labelled_images.csv", index=False)
    print(f"{len(labelled)} labelled pairs ({'with' if use_text else 'without'} text), "
          f"{len(unlab)} unlabelled images")

    # ---- model
    model, transform, tokenize, device = clip_model.load(args.model)
    hf_tok = open_clip.get_tokenizer(clip_model.MODELS[args.model]).tokenizer
    params = set_trainable(model, args.unfreeze)
    scale = model.logit_scale.exp().detach().item()
    with torch.no_grad():
        concept_txt = F.normalize(model.encode_text(tokenize([f"{PROMPT_INIT} {c}" for c in CONCEPTS]).to(device)), dim=-1)
    head = ConceptHead(concept_txt if use_text else torch.randn_like(concept_txt) * 0.02, scale).to(device)
    surrogate = SurrogateCaptions(model.text, hf_tok).to(device) if use_text else None
    head_params = list(head.parameters()) + (list(surrogate.parameters()) if use_text else [])
    print(f"model on {device}; training {sum(p.numel() for p in params) / 1e6:.1f}M encoder parameters "
          f"+ {sum(p.numel() for p in head_params) / 1e3:.0f}k head/prompt parameters")
    bank = load_bank()
    strong = RandAugment(num_ops=2, magnitude=9)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    def make_opt(total_steps):
        opt = torch.optim.AdamW([{"params": params, "lr": args.lr, "weight_decay": args.weight_decay},
                                 {"params": head_params, "lr": args.head_lr, "weight_decay": 0.0}])
        warm = max(1, total_steps // 10)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else
                                                  0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
        return opt, sched

    def step(opt, sched, loss):
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()

    def encode_imgs(paths, strong_aug=False):
        x = torch.stack([transform(load_image(p, rng, strong if strong_aug else None)) for p in paths]).to(device)
        return F.normalize(model.encode_image(x), dim=-1)

    def labelled_terms(batch):
        """Image embeddings, concept targets and (with text) caption embeddings of a labelled batch."""
        img = encode_imgs([p["path"] for p in batch])
        y = multi_hot([p["concepts"] for p in batch], device)
        txt = None
        if use_text:
            texts = [p["text"] if p["text"] is not None else pseudo_caption(bank, *p["grades"], rng) for p in batch]
            txt = F.normalize(model.encode_text(tokenize(texts).to(device)), dim=-1)
        return img, y, txt

    history = {"stage1": [], "stage2": []}
    t0 = time.time()

    # ---- stage 1: supervised
    model.train()
    steps1 = math.ceil(len(labelled) / args.batch_size) * args.epochs1
    opt, sched = make_opt(steps1)
    for epoch in range(args.epochs1):
        order = list(range(len(labelled)))
        rng.shuffle(order)
        losses = []
        for b in range(0, len(order), args.batch_size):
            batch = [labelled[i] for i in order[b:b + args.batch_size]]
            if len(batch) < 2:
                continue
            with torch.autocast("cuda", enabled=use_amp):
                img, y, txt = labelled_terms(batch)
                loss = scm_loss(head(img.float()), y)
                if use_text:
                    loss = loss + multi_positive_loss(img.float(), txt.float(), [p["group"] for p in batch], scale)
            step(opt, sched, loss)
            losses.append(loss.item())
        history["stage1"].append(float(np.mean(losses)))
        print(f"  stage 1 epoch {epoch + 1:2d}/{args.epochs1}  loss {history['stage1'][-1]:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- pseudo-labels for the unlabelled images (weak view, fixed for stage 2)
    model.eval()
    probs = []
    with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
        for b in range(0, len(unlab_paths), 64):
            x = torch.stack([transform(Image.open(p)) for p in unlab_paths[b:b + 64]]).to(device)
            probs.append(F.softmax(head(F.normalize(model.encode_image(x), dim=-1).float()), 1).cpu())
    probs = torch.cat(probs)
    exp = probs[:, EXP_C.start:EXP_C.stop].argmax(1).numpy()
    icm = probs[:, ICM_C.start:ICM_C.stop].argmax(1).numpy()
    te = probs[:, TE_C.start:TE_C.stop].argmax(1).numpy()
    icm, te = np.where(exp <= 1, ND, icm), np.where(exp <= 1, ND, te)
    pseudo = [grade_concepts(int(e), int(i), int(t)) for e, i, t in zip(exp, icm, te)]
    pl = unlab[["image", "EXP", "ICM", "TE"]].assign(pl_EXP=exp, pl_ICM=icm, pl_TE=te)
    pl.to_csv(args.out_dir / "pseudo_labels.csv", index=False)
    pl_acc = {g: float((pl[g] == pl[f"pl_{g}"]).mean()) for g in ("EXP", "ICM", "TE")}
    print("  pseudo-label accuracy on the unlabelled images (diagnostic only): "
          + " ".join(f"{g} {a:.3f}" for g, a in pl_acc.items()), flush=True)

    # ---- stage 2: semi-supervised
    model.train()
    steps_per_epoch = math.ceil(len(unlab_paths) / args.batch_size)
    opt, sched = make_opt(steps_per_epoch * args.epochs2)
    lab_order = []
    for epoch in range(args.epochs2):
        order = list(range(len(unlab_paths)))
        rng.shuffle(order)
        losses = []
        for b in range(0, len(order), args.batch_size):
            ub = order[b:b + args.batch_size]
            if len(ub) < 2:
                continue
            if len(lab_order) < args.batch_size:
                lab_order += rng.sample(range(len(labelled)), len(labelled))
            lb, lab_order = [labelled[i] for i in lab_order[:args.batch_size]], lab_order[args.batch_size:]
            with torch.autocast("cuda", enabled=use_amp):
                y_u = multi_hot([pseudo[i] for i in ub], device)
                img_strong = encode_imgs([unlab_paths[i] for i in ub], strong_aug=True)
                loss = scm_loss(head(img_strong.float()), y_u)
                img_l, y_l, txt_l = labelled_terms(lb)
                if use_text:
                    img_u = encode_imgs([unlab_paths[i] for i in ub])
                    txt_u = F.normalize(surrogate(model.text, [pseudo[i] for i in ub]), dim=-1)
                    k = max(1, math.ceil(args.top_p * len(ub)))
                    top = (img_u * txt_u).sum(1).detach().topk(k).indices
                    loss = loss + trapezoid_loss(img_l.float(), txt_l.float(), [p["group"] for p in lb],
                                                 img_u[top].float(), txt_u[top].float(), scale)
                else:
                    loss = loss + scm_loss(head(img_l.float()), y_l)
            step(opt, sched, loss)
            losses.append(loss.item())
        history["stage2"].append(float(np.mean(losses)))
        print(f"  stage 2 epoch {epoch + 1:2d}/{args.epochs2}  loss {history['stage2'][-1]:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    trained = {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}
    torch.save(trained, args.out_dir / "adapter.pt")
    torch.save({"concept_head": head.state_dict(), "concepts": CONCEPTS,
                "prompt_ctx": surrogate.ctx.detach().cpu() if use_text else None}, args.out_dir / "heads.pt")
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    config.update(n_labelled_pairs=len(labelled), n_unlabelled=len(unlab), loss_history=history,
                  pseudo_label_acc=pl_acc, device=device, seconds=round(time.time() - t0))
    json.dump(config, open(args.out_dir / "config.json", "w"), indent=2)
    print(f"saved {len(trained)} tensors to {args.out_dir / 'adapter.pt'}")


if __name__ == "__main__":
    main()
