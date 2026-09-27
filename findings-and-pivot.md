# Prior work on our dataset, and the proposed pivot

_Last updated: 2026-09-23_

## Summary

We checked whether anyone has already solved Gardner grading on the Kromp 2023
blastocyst dataset — the dataset we downloaded and planned to build on.

**They have.** At least four published models train and evaluate on it, and the
best two report results **at or above human expert level**. Our originally
planned baseline (one shared encoder, three grade heads, a class-imbalance loss)
is, in substance, an already-published paper (MDMBG-Net).

**Therefore: competing on grading accuracy is not worth our time.** This document
proposes pivoting to two questions nobody on this dataset has addressed —
grading taught by expert language rather than grade labels, and uncertainty that
matches where the experts themselves disagree. Patient-level leakage is checked
throughout as a supporting experiment.

## How we checked

- Pulled **all 26 papers citing the dataset paper** (Semantic Scholar citation
  list for DOI `10.1038/s41597-023-02182-3`) and read each citation context to
  separate papers that *used* the data from those that merely mention it.
- Targeted web searches for the papers that use it, to recover results.
- **Coverage caveat:** Google Scholar usually lists more citing papers than
  Semantic Scholar; that check is still outstanding.

## What has already been done on this dataset

| Paper | Method | EXP | ICM | TE | Metric | Official split confirmed? |
|---|---|---|---|---|---|---|
| Kromp 2023 (dataset paper) | Separate Xception / Swin per grade | 85.2 | 67.1 | 67.4 | accuracy % | yes |
| **Embryo2live** (Yu et al. 2024, medRxiv preprint) | Vision-transformer system | 86.2 | 76.4 | 74.0 | accuracy % | **yes** — states silver set for training, gold set for test |
| **MDMBG-Net** (Barhoun et al. 2026, Biomed. Signal Proc. & Control) | Multi-task shared representation + class-balanced loss | **86.25** | **78.34** | **74.40** | accuracy % | not verified (paywalled) |
| **BM-DINO** (Yang et al. 2025, IEEE BIBM) | DINOv2, last 4 blocks fine-tuned, + annotation-free region masks | 0.78 | **0.80** | 0.73 | macro-F1 | not verified (paywalled) |
| MFT-Embryo (Du et al. 2026, ICAACE) | Multi-scale CNN–Transformer | 83.22 | 69.52 | 62.50 | accuracy % | not verified |
| _Human experts vs. consensus_ | _(reference)_ | _78_ | _74_ | _70_ | accuracy % | yes |

Links: [Embryo2live](https://www.medrxiv.org/content/10.1101/2024.10.28.24316259v1) ·
[MDMBG-Net](https://doi.org/10.1016/j.bspc.2026.109455) ·
[BM-DINO](https://ieeexplore.ieee.org/document/11356257/) ·
[MFT-Embryo](https://doi.org/10.1109/ICAACE69793.2026.11509095) ·
[Kromp 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10175281/)

Used the same images but not comparable: KGAttention (binary good/poor, not
Gardner grades), a self-organizing-map paper (own split, unsupervised), a
cross-modal multi-task paper (adds clinical data, no numbers in abstract).
Different datasets entirely: MEmEBG and Blasto-Net (Saeedi, 249 images),
a YOLOv8 grading paper (private Cairo data, 1,015 images).

## Architectures of the prior work

| Paper | Architecture | Source |
|---|---|---|
| Kromp 2023 | Xception, DeiT and Swin (ImageNet-pretrained), **a separate model per grade** — nine in total. Input 224×224×3, 60 epochs, batch 64, Adam + cosine schedule, flip/rotation augmentation, SWA-Gaussian weight averaging. Best backbone differed per grade: Swin for expansion, Xception for ICM. | full text |
| Embryo2live 2024 | **One model per task, each with its own backbone**: ResNeXt50 for EXP and ICM, ViT-B/16 for TE and the outcome tasks (ViT-L/16 for live birth on the larger frozen set). Small MLP head each (64 → 1, tanh then sigmoid). Attention rollout for heatmaps. Loss accounts for the grades not being independent; confirms ICM/TE = ND when EXP < 3. | full text |
| MDMBG-Net 2026 | One shared representation with multi-task heads for all three grades, plus a class-balanced loss for rare grades. Backbone not named in the abstract. | abstract only (paywalled) |
| BM-DINO 2025 | DINOv2 with the last four blocks fine-tuned. A fixed algorithm produces four-class region masks at no labelling cost; those masks drive ROI-guided pooling and a mask-encoder CNN. Prompt tokens, ROI features and mask priors are summed. | abstract only (paywalled) |
| MFT-Embryo 2026 | Dual-branch CNN–Transformer with multi-scale patch embedding and a local self-attention module. | abstract only |

Two of the five share an encoder across grades; **none of them uses text or a
vision-language model.**

## Why accuracy is a dead end here

- **The test set is small.** 300 images, so a 2-point gain is about 6 images —
  inside the noise band (roughly ±5 points at 95% confidence).
- **The labels have a ceiling.** Experts agree with the panel consensus only
  70–78% of the time, and the best models are already there.
- **The obvious designs are taken.** Multi-task + imbalance loss = MDMBG-Net.
  Self-supervised ViT + partial fine-tuning = BM-DINO.

## FEMI is disqualified on this dataset

FEMI's Methods (p11) state its pretraining corpus incorporated "2344 blastocyst
images from clinics across Europe" — this dataset, **including our 300 test
images**. Any FEMI score on the official test set would be inflated by
pretraining exposure. This is independent of, and adds to, the practical
problems the stage-prediction work hit with FEMI.

## Gaps nobody has filled

1. **No vision-language model has been tried on this dataset.** None of the 26
   citing papers uses BiomedCLIP, BMCA-CLIP or any CLIP-style model. Open
   question: can a biomedical VLM reach expert level zero-shot, or with far
   fewer labels?
2. **Patient-level leakage is unexamined.** 200 patient prefixes appear in both
   the official train and test splits (our own count). Sibling embryos from one
   cycle therefore sit on both sides. Whether the published numbers are inflated
   by this is unknown — and checkable.
3. **Data efficiency is unmeasured.** How many labelled images does expert-level
   grading actually need?
4. **Annotator disagreement is unused.** `Gardner_test_gold.xlsx` carries
   per-image agreement scores. Nobody has asked whether models fail on precisely
   the images experts dispute.

## Proposed pivot

**From:** "train a Gardner grading model and beat the published baselines."

**To:** "how well do biomedical vision-language models grade blastocysts, and
how much of published performance survives a leakage-free split?"

Two contributions, both reachable with the setup already planned:

**A. Grading taught by language.** Adapt a biomedical VLM on the expert captions
that state Gardner grades in words (20 of the 100 gold captions do; the silver
set should hold roughly 200 more), then grade Kromp images with 0, 10, 50 and
200 labels, against a supervised CNN trained on all 2,044. The captions come
from a **different clinic and microscope** (Gomez, EmbryoScope time-lapse) than
the Kromp images (Olympus, static), so this measures cross-clinic transfer as
well as label efficiency. It answers the problem the dataset paper itself
names — adapting automated solutions to custom clinical data, where a clinic has
images and expert talk but no thousands of Gardner labels.

**B. Uncertainty that experts recognise.** Output a grade, a sentence and a
confidence, then test whether the model hesitates on the images where the 11
embryologists disagreed. `Gardner_test_gold.xlsx` carries per-image agreement
scores that nobody has used. Existing explainability work here stops at
Grad-CAM heatmaps; this measures whether the model's doubt is real.

**Supporting experiment (inside both tracks):** every result is reported on the
official split *and* the patient-level split — drop the 629 training images
whose patient also appears in the test set, keep all 300 test images. This
quantifies how much published performance is patient leakage.

Why this survives the prior work: none of these questions is answered by
MDMBG-Net, BM-DINO or Embryo2live, and a negative result is still publishable —
"VLMs do not beat supervised CNNs here" and "published scores drop by X points
without patient leakage" are both useful findings.

### Proposed baseline architecture

```
image 512×384  →  square pad + bicubic resize 224  →  BiomedCLIP ViT-B/16 (frozen)
                →  three linear heads: EXP (5) · ICM (3) · TE (3)
                →  rule, not learned: EXP grade 1–2  ⇒  ICM = TE = ND
```

- Only the three heads train, which suits 2,044 images — the setting where Kim
  et al. 2024 found fine-tuning the encoder made things worse.
- Track A adds caption supervision on the text encoder; track B adds a
  confidence and a generated sentence per prediction.
- Encoder swaps (BMCA-CLIP, DINOv2), an ordinal loss and imbalance correction
  are each measured against this fixed baseline, one change at a time.
- FEMI is excluded — see above.

## Slide deck

The same story in 8 slides: `Blastocyst grading - prior work and pivot.pptx`
(prior work, dead-end argument, FEMI, gaps, pivot, architectures of prior work,
our architecture).

## Open items before committing

- [ ] Read MDMBG-Net and BM-DINO in full (needs library access to Elsevier and
      IEEE). Confirm their split, and whether either addresses patient leakage.
- [ ] Check Google Scholar's "cited by" list for papers Semantic Scholar missed.
- [ ] Confirm with the stage-prediction member which model they used — FEMI
      (a ViT masked autoencoder) or DINOv2. These are different models, and the
      problems they hit may not transfer.
- [ ] Decide whether the outcome-prediction stretch goal (234 live births among
      754 labelled images) stays in scope.
