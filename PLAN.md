# Baseline plan: BiomedCLIP on the blastocyst dataset

Direction: CLIP-based backbones (starting with BiomedCLIP) on embryo image–text
pairs with Gardner grade labels. "Reproducing the baseline" means, in order:
1. reproduce the published numbers with our own evaluation code, then
2. build our baseline (frozen BiomedCLIP + three linear heads) and measure it
   with that same code.

## Decisions (fixed)
| Decision | Choice | Why |
|---|---|---|
| Duplicate `838_02.png` (conflicting EXP 4 vs 5) | Drop both rows | No way to know which label is right |
| `846_01.png` (in neither split) | Exclude | Not part of the benchmark |
| Validation set | 15% of train, grouped by patient, stratified by EXP | Test set is used once, at the end; grouping stops sibling-embryo leakage |
| Tuning linear heads | 5-fold grouped cross-validation | Heads train in seconds, so the more robust option is cheap |
| Tuning fine-tuning runs | The fixed 15% validation split | Each run is expensive |
| Compute | Local CPU for data + eval; Colab (T4) for encoding and fine-tuning | This machine has no GPU |
| Code sync | Private GitHub repo, cloned in Colab | Every result traceable to a commit |
| Data sync | `data/` uploaded to Google Drive, mounted in Colab | Never committed to git |

## Where each step runs
| Step | Where | What |
|---|---|---|
| 1 | Local | Phase 1 label table, Phase 2 evaluation harness, prompt bank |
| 2 | Colab | Encode all images with BiomedCLIP, save embeddings to Drive |
| 3 | Local or Colab | E0 zero-shot + E1 baseline heads from saved embeddings |
| 4 | Colab | Silver captions + track A fine-tuning |

Notebooks: `01_extract_embeddings.ipynb`, `02_finetune_track_a.ipynb`.
All logic lives in `src/`; notebooks only mount Drive, clone the repo, and call it.
Save embeddings and checkpoints to Drive as you go — Colab sessions disconnect.
First notebook cell records package versions.

---

## Phase 0 — Environment
- [ ] Install `open_clip_torch`, `timm`, `torchvision` (CPU locally)
- [ ] Load `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`; embed one image
- [ ] Confirm the preprocessing resizes with **bicubic** interpolation
- [ ] Layout: `src/`, `prompts/`, `results/`, `notebooks/`; seeds fixed in one place

Done when: one blastocyst image yields a 512-d embedding.

## Phase 1 — One clean label table → `data/blastocyst/labels.csv` ✅ done
Built by `python src/build_labels.py`. Result: 2,042 train / 300 test; validation
334 images (16.4% — grouped splits move whole patients, so exactly 15% is not
reachable; EXP 5 has only 13 validation images); 5 folds of 396–432; patient-level
train 1,413; ND rule holds with zero exceptions in both splits.
Test agreement parsed from `Gardner_test_gold.xlsx`: `agree_*` = vote fraction,
`raters_*` = number of annotators, `revised_*` = grade settled by panel
discussion after disagreement (EXP 23, ICM 44, TE 37 images).

- [x] Columns: `image, split, patient_id, EXP, ICM, TE, agreement_EXP/ICM/TE, fold, is_val, patient_overlap`
- [x] ND = 3 everywhere; test grades as integers; raw filenames kept exactly (never re-pad)
- [x] Apply the fixes above; keep the 21 test images with a missing gold label, exclude per grade when scoring
- [x] Validation split + 5 grouped folds; flag the 629 train images whose patient appears in test
- [x] Automated checks: 2,042 train (2,044 minus the duplicate pair), 300 test, patient-level train count

Done when: checks pass and nothing downstream reads the raw CSVs.

## Phase 2 — Evaluation harness (reproduce published numbers)
- [ ] Per grade: accuracy, per-class F1, macro F1, weighted F1, confusion matrix
- [ ] Bootstrap 95% confidence intervals
- [ ] `prediction_xception.csv` → must give 0.785 / 0.723 / 0.670
- [ ] Authors' repo predictions (Swin, DeiT) → best per grade should give 85.2 / 67.1 / 67.4
- [ ] Cross-check with the authors' `evaluate_prediction_results.py`

Done when: our numbers match theirs (also settles the conflicting Swin figures in CLAUDE.md).

## Phase 3 — Text side
- [ ] `prompts/gardner_prompts.json`: 3–5 phrasings per grade from the Gardner criteria and Istanbul Consensus
- [ ] Download `caption_silver.jsonl` + `images.zip` (Figshare 32024349); **count blastocyst-stage captions**
  — go/no-go gate for track A (need roughly 200)
- [ ] Pseudo-caption generator: grade triple → sentence

## Phase 4 — E0: zero-shot (no training)
- [ ] Encode all images once, cache embeddings to Drive
- [ ] Encode prompts, average phrasings per grade, predict by cosine similarity, apply ND rule
- [ ] Report on both splits (expected to be weak — it is the starting point)

## Phase 5 — E1: THE BASELINE
- [ ] Three linear heads on cached embeddings: EXP (5), ICM (3), TE (3)
- [ ] ICM/TE heads train only on EXP ≥ 3; plain cross-entropy
- [ ] Tune regularization with 5-fold grouped CV
- [ ] Prediction: EXP 1–2 → ICM = TE = ND; also report ICM/TE given the true EXP
- [ ] 5 seeds; official and patient-level splits
- [ ] Table vs Xception, Swin, human experts, MDMBG-Net

Go/no-go: within ~5 points of Xception on the official split → proceed to Phase 6.

## Phase 6 — Track A: image–text fine-tuning (Colab)
- [ ] Contrastive fine-tuning on pseudo-captions + silver captions (last block or LoRA)
- [ ] Rerun E0 and E1 on the adapted model
- [ ] Label-efficiency curve: 0 / 10 / 50 / 200 / all labels, adapted vs frozen

## Phase 7 — Later, each measured against the Phase 5 baseline
- [ ] Encoder swap: BMCA-CLIP, DINOv2
- [ ] Ordinal loss: CORN, Kragh's ordinal cross-entropy
- [ ] Imbalance correction: logit adjustment
- [ ] Track B: confidence vs per-image annotator agreement
