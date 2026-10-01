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

## Phase 2 — Evaluation harness (reproduce published numbers) ✅ done
`src/evaluate.py`; all 9 accuracy and weighted-F1 numbers of the paper's Table 2
reproduce under the authors' protocol. See `results/baselines/README.md`.

- [x] Per grade: accuracy, per-class F1, macro F1, weighted F1, confusion matrix
- [x] Bootstrap 95% confidence intervals
- [x] `prediction_xception.csv` → 0.780 / 0.688 / 0.617 (kromp protocol, = paper) and 0.785 / 0.723 / 0.670 (clean protocol)
- [x] All 12 released models scored; Embryo2live's quoted 85.2 / 67.4 match Swin under the clean protocol, its 67.1 ICM matches DeiT, not Xception
- [x] Cross-check with the authors' `evaluate_prediction_results.py`

Done when: our numbers match theirs (also settles the conflicting Swin figures in CLAUDE.md).

## Phase 3 — Text side ✅ done
- [x] `prompts/gardner_prompts.json`: 3 phrasings per grade (Gardner criteria) × 3 templates = 9 prompts per class; `src/text.py` loads them
- [x] Silver captions + images downloaded to `data/captions/` (checksums verified)
- [x] **Go/no-go count: GO, but thinner than estimated.** `src/select_blastocyst_captions.py` finds
      **172 blastocyst-stage captions** (silver 152 + gold 20; estimate was ~200).
      Expansion grade stated in 120 (grades 1–5: 24/27/46/18/5); ICM letter grade in 53, TE in 52.
      Expansion is well covered, ICM/TE quality much less. About 3 of 120 stated grades contradict the stage (caption noise).
      Bonus: the 120 grade-stating captions form a small cross-clinic (EmbryoScope) Gardner test set.
- [x] Pseudo-caption generator `pseudo_caption()` in `src/text.py`: grade triple → sentence

## Phase 4 — E0: zero-shot (no training) ✅ done
Ran on Colab T4 via `notebooks/01_extract_embeddings.ipynb` (commit a371795, 2026-09-27).
Embeddings saved to Drive: `MyDrive/blastocyst-clip/results/embeddings/biomedclip/`.
- [x] Encode all images once, cache embeddings to Drive (2,342 blastocyst + 172 caption images, 512-d)
- [x] Encode prompts, average phrasings per grade, predict by cosine similarity, apply ND rule
- [x] Result — **worse than always guessing the most common grade**:
      clean accuracy EXP 0.134 / ICM 0.168 / TE 0.184 (macro F1 0.105 / 0.140 / 0.157);
      kromp protocol 0.133 / 0.167 / 0.211. Predicted grade 1 for 188 of 300 test images
      (true grade 1: 23). BiomedCLIP's text side does not link Gardner descriptions to these
      images; the link must be learned (track A). This is the floor of the label-efficiency curve.

## Phase 5 — E1: THE BASELINE ✅ GO (after check b)
`src/linear_probe.py` on the Colab embeddings; results in `results/linear_probe/biomedclip/`.
Logistic-regression heads, C tuned by 5-fold grouped CV on accuracy, no class weights.

| clean protocol, accuracy | EXP | ICM | TE |
|---|---|---|---|
| Official split | 0.758 | 0.620 | 0.559 |
| Patient-level split | 0.748 | 0.613 | 0.562 |
| Xception (published) | 0.785 | 0.723 | 0.670 |
| Majority-grade guess | 0.513 | 0.524 | 0.424 |

- EXP within 3 points of Xception (pass); ICM −10.3, TE −11.1 (fail).
- **ICM head collapsed to "always A"** (predicts A for 65/68 true B, 16/17 true C); CV chose strong regularisation (C = 0.001).
- **Label shift between splits:** grade-A share is 79% (ICM) / 65% (TE) in the silver training labels (one expert)
  but 64% / 52% in the gold test labels (panel) — the silver grader is more lenient. Affects every model on this dataset.
- Patient leakage barely changes a frozen model (0.758 vs 0.748).

Diagnostics before deciding:
- [x] (a) Tune C on macro F1 instead of accuracy (local) → **minor factor only.**
      Macro F1 rises (EXP 0.534→0.576, ICM 0.369→0.427, TE 0.415→0.461) and the head no longer
      predicts only A, but accuracy is flat (0.752 / 0.616 / 0.587): 53/68 true-B still called A,
      0/17 true-C recognised. Gap to Xception stays ~10 points → the 512-d features are the limit.
      Results: `results/linear_probe/biomedclip_f1_macro/`.
- [x] (b) Pre-projection 768-d features + mean patch features (computed locally; the 512-d part
      matches Colab exactly) → **fixes it.** Results in `results/linear_probe/biomedclip_full_*`.

| clean, official, accuracy (macro F1) | EXP | ICM | TE | mean CV macro F1 |
|---|---|---|---|---|
| 512-d embedding | 0.752 (0.576) | 0.616 (0.427) | 0.587 (0.461) | 0.471 |
| CLS 768-d | 0.748 (0.565) | 0.647 (0.453) | 0.639 (0.545) | 0.482 |
| Patch mean 768-d | 0.748 (0.591) | 0.671 (0.474) | 0.611 (0.539) | 0.495 |
| **CLS + patch mean 1,536-d** | **0.779 (0.614)** | **0.678 (0.505)** | **0.663 (0.594)** | **0.498** |
| Xception (published) | 0.785 (0.657) | 0.723 (0.555) | 0.670 (0.554) | — |

Feature variant chosen by mean CV macro F1 on training data (not by test score): CLS + patch mean.
Within the go/no-go margin on all three grades; TE macro F1 beats Xception.
Kromp protocol: 0.773 / 0.642 / 0.613 (Xception 0.780 / 0.688 / 0.617).

**Final baseline definition:** frozen BiomedCLIP; features = CLS token + mean of patch tokens
(1,536-d, pre-projection); three logistic-regression heads (EXP 5, ICM 3, TE 3; ICM/TE on EXP ≥ 3);
C per head by 5-fold grouped CV on macro F1; ND rule at prediction time.
Run: `python src/linear_probe.py --emb-dir results/embeddings/biomedclip_full --feature cls+patchmean --select f1_macro`

- [x] **Leakage control** (`src/leakage_control.py`, results in `results/leakage_control/`): drop 629
      images of *random* non-overlap patients instead, 5 seeds. Clean accuracy EXP / ICM / TE:
      official 0.779 / 0.678 / 0.663; control 0.769±0.028 / 0.645±0.015 / 0.645±0.019;
      patient-level 0.782 / 0.627 / 0.615. → Fewer images costs ICM 3.3 / TE 1.8 points;
      **leakage costs ICM 1.8 / TE 3.0 points, EXP none** — suggestive only (1–1.6 sd of the
      control spread; test-set CI is ±5). Frozen encoders gain little from leakage; whether
      fine-tuned published models gain more would need one retrained on the patient-level split
      (future work, not pursued now).


## Phase 6 — Track A: image–text fine-tuning (Colab) — in progress
Code: `src/finetune.py` (contrastive fine-tuning, last 2 blocks of each encoder + projections =
29.6M of 196M parameters; multi-positive loss so same-grade pseudo-captions are not pushed apart,
as in UniCL, Yang et al. CVPR 2022), `src/embed.py --weights`, `notebooks/02_finetune_track_a.ipynb`.

Real captions: 172 blastocyst-stage, of which **53 (33 embryos) are held out for cross-clinic
testing**, fixed before any training; 119 used for adaptation.

| Condition | Adapted on | Kromp labels used |
|---|---|---|
| A0 | nothing (frozen baseline) | N, heads only |
| A1 | 119 real expert captions (other clinic) | none during adaptation |
| A2 | real captions + pseudo-captions of the same N images | N |

- [x] Step 1: A1, seed 0 — **negative result: 119 free-text captions do not teach grading.**
      Training loss 4.60 → 2.71 over 20 epochs, still falling (learned, no overfitting, maybe undertrained).
      Zero-shot on Kromp test: 0.134 / 0.168 / 0.184 (frozen) → 0.178 / 0.284 / 0.288 (A1), still below
      majority guessing. **Zero-shot on held-out caption images (same microscope as the training
      captions):** EXP 0.368 → 0.368, ICM 0.333 → 0.444, TE 0.278 → 0.278 (majority 0.447 / 0.444 / 0.611;
      n = 38 / 18 / 18) — and even on the training captions EXP only 0.366 → 0.378. So the model learned
      to match each caption to its own image, not the grading concepts; the microscope gap is not the
      explanation. Linear heads on A1 features (clean, official): 0.785 / 0.671 / 0.625 vs baseline
      0.779 / 0.678 / 0.663; CV macro F1 unchanged (0.610 / 0.399 / 0.495 vs 0.606 / 0.398 / 0.491).
      `src/caption_zeroshot.py`; results in `results/linear_probe/biomedclip_a1_seed0_*`.
- [ ] Step 2: label-efficiency grid, N = 50 / 100 / 200 / 500 / 2,042, 3 seeds, A0 vs A1 vs A2
      (`src/label_efficiency.py`, `src/subsets.py`; A2 via `notebooks/03_label_efficiency_a2.ipynb`)
  - [x] A0 and A1 done locally (`results/label_efficiency/`). Clean accuracy, mean of 3 seeds,
        A0 → EXP 0.671 / 0.685 / 0.706 / 0.751 / 0.779, ICM 0.590 / 0.607 / 0.605 / 0.652 / 0.678,
        TE 0.488 / 0.534 / 0.565 / 0.604 / 0.663 for N = 50 / 100 / 200 / 500 / all.
        **A1 is within 2 points of A0 at every N** (seed sd 0.01–0.04): caption-only adaptation
        does not change label efficiency. With 500 labels A0 keeps ~96% of full EXP/ICM accuracy.
  - [x] A2 on Colab (13 runs, ~1 h 15 min on a T4). Summary + plot: `src/plot_label_efficiency.py` →
        `results/label_efficiency/curves.png`, `summary.csv`. Clean accuracy, A0 → A2:
        N=50 0.671→0.655 / 0.590→0.597 / 0.488→0.509; N=100 0.685→0.676 / 0.607→0.596 / 0.534→0.535;
        N=200 0.706→0.714 / 0.605→0.602 / 0.565→0.560; N=500 0.751→0.762 / 0.652→0.643 / 0.604→0.593;
        all (1 seed) 0.779→**0.805** / 0.678→**0.692** / 0.663→**0.674**.
        **Language does not reduce the labels needed** (A2 within ~2 points of A0 for N ≤ 500).
        At full labels A2 is our best result, above Xception on EXP and TE — but 1 seed, within the
        ±5-point test noise, and possibly due to fine-tuning rather than language.
        Zero-shot after A2 stays far below heads at every N (all labels: 0.557 / 0.555 / 0.535).
  - [ ] Control: same fine-tuning with a plain classification loss (no text), N = all, 3 seeds;
        plus A2 at N = all for seeds 1 and 2
- [ ] Cross-clinic test on the 53 held-out captions (38 state an expansion grade)
- [ ] Initial hyperparameters (20 epochs, lr 2e-5, batch 32) are untuned; tune on the validation split if step 1 is promising

## Phase 7 — Later, each measured against the Phase 5 baseline
- [ ] Encoder swap: BMCA-CLIP, DINOv2
- [ ] Ordinal loss: CORN, Kragh's ordinal cross-entropy
- [ ] Imbalance correction: logit adjustment
- [x] Track B, first analysis (`src/track_b.py`, `results/track_b/uncertainty_vs_agreement.csv`):
      our baseline + all 12 released models; uncertainty = 1 − max probability; "contested" = panel
      not unanimous or revised by discussion. Contested share: EXP 154/298, ICM 133/237, TE 130/232.
      **Errors concentrate on contested images** for every model (best published: EXP 0.93–0.98
      unanimous vs 0.66–0.75 contested; ICM 0.84–0.91 vs 0.49–0.62). **Uncertainty flags contested
      images moderately for EXP (AUROC 0.65–0.79), weakly for ICM (0.60–0.70), not at all for TE
      (0.47–0.58) — for every model.** Ours is the weakest (0.674 / 0.596 / 0.545); the released models
      use SWA-Gaussian averaging. Released `convnextv2_tiny` has no TE probabilities (all NaN).
      Next: bootstrap CIs; test calibration fixes (temperature scaling, ensembling, soft labels).
