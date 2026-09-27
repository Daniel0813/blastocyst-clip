# Embryo Quality (Gardner Grade) Prediction — Research Context

## Goal and scope
The user leads this project's direction; decide scope questions with them.

1. **Main goal:** predict the three Gardner grades of a day-5 blastocyst from
   one static image: expansion (EXP 1–5), inner cell mass (ICM A/B/C/ND) and
   trophectoderm (TE A/B/C/ND).
2. **Stretch goal (only after the main goal works):** predict pregnancy or live
   birth using predicted Gardner grades plus clinical fields, on the 754 images
   with outcomes. Do not train end-to-end on images for this; 754 samples is
   too few.
3. **Out of scope:** time-lapse stage classification (another member's work).

**Evaluation: report both splits every time.**
- **Official split** (2,042 train after dropping the conflicting `838_02.png`
  pair / 300 gold test) is the headline number. It is directly comparable with
  Xception, Swin and the human experts.
- **Patient-level split** is the leakage-free check. Use the same 300 gold test
  images, but drop from training every image whose patient prefix also appears
  in the test set: 2,042 → 1,413 training images (ICM-C 10, TE-C 35). Keeping
  only unseen-patient test images instead was rejected because it leaves just
  58 test images.
- Metrics: accuracy (for comparison with published numbers) plus per-class and
  macro F1, because the rare grades (ICM-C, TE-C, EXP 5) are hidden by
  accuracy alone.

## Data
**Kromp et al. 2023 human blastocyst dataset** (Scientific Data; CC BY 4.0).
Stored in `data/blastocyst/`.
Source: https://figshare.com/articles/figure/Blastocyst_dataset_zip/20123153 ·
paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10175281 ·
scripts: https://github.com/software-competence-center-hagenberg/Blastocyst-Dataset

- `Images/`: 2,344 static day-5 blastocyst PNGs, 512×384 RGB, single focal
  plane, Olympus IX50 microscope (not time-lapse, not EmbryoScope).
- `Gardner_train_silver.csv` (`;`-separated): 2,044 rows, graded by one Gardner expert.
- `Gardner_test_gold_onlyGardnerScores.csv`: 300 images, consensus of an
  international expert panel. `Gardner_test_gold.xlsx` has the same labels plus
  per-image annotator agreement.
- `Clincial_annotations.csv` (the misspelling is in the original filename;
  latin-1, `;`): 754 fresh-transfer images with clinical fields and outcomes.
- `prediction_xception.csv` + `evaluate_prediction_results.py`: the authors'
  baseline predictions and evaluation script.

**Label encoding**
- Expansion (EXP) `0–4` = Gardner grade 1–5.
- ICM and TE: `0/1/2` = A/B/C. "Not defined" is `3` in the train file but `ND`
  in the test file.
- Test EXP values are stored as floats (`2.0`), so compare labels numerically.

**Hard rule, holds with zero exceptions:** EXP grade 1–2 ⇔ ICM and TE are
not defined.

**Class counts (train / test)**
| | Train | Test |
|---|---|---|
| EXP 1/2/3/4/5 | 197/194/429/1081/143 | 23/31/86/153/5 |
| ICM A/B/C/ND | 1305/332/16/391 | 153/68/17/54 |
| TE A/B/C/ND | 1081/525/47/391 | 122/82/30/54 |
ICM-C (16 train images) and TE-C (47) are very rare.

**Baselines to beat (official test set)**
- Xception (reproduced from the provided predictions): EXP 0.785 / ICM 0.723 / TE 0.670 accuracy.
- Swin (paper): 0.85 / 0.65 / 0.62.
- Human experts vs. consensus (paper): 0.78 / 0.74 / 0.70.

**Known data problems**
- **Train and test share patients.** The filename prefix appears to be a patient
  or cycle ID, and 200 prefixes appear in both splits. The published baselines
  have the same leakage.
- 21 test images are missing at least one gold label. Exclude them per grade.
- `838_02.png` appears twice in train with conflicting EXP (grade 4 vs 5).
- `846_01.png` is on disk but in neither split.
- Filenames on disk mix padded and unpadded styles (`0175_05.png`, `420_02.png`).
  Match raw names exactly; never re-pad them.
- Clinical file: `187_1.png` does not match any file (disk has `0187_01/02`).
  `AMH` is corrupted by Excel date auto-formatting (501 of 754 values look like
  dates, 165 missing), so treat it as unreliable. `Fond` is undocumented.
- Outcomes (column meanings inferred, not defined in the file): SS (likely
  biochemical pregnancy) 317/754, HA (likely fetal heart activity) 261, LB
  (live birth) 234. They nest as expected, with one LB=1/HA=0 exception.
  652 of these images are in train and 100 in test.

## Approach
Full checklist in `PLAN.md`; `findings-and-pivot.md` explains why.

- **Direction:** CLIP-based backbones (BiomedCLIP first) on embryo image–text
  pairs with Gardner grade labels. Accuracy alone is not the claim — prior work
  already reaches ~86 / 78 / 74. Tracks: (A) grading taught by expert language,
  measured by label efficiency and cross-clinic transfer; (B) confidence that
  matches per-image annotator agreement.
- **Baseline:** frozen BiomedCLIP ViT-B/16 + three linear heads (EXP 5, ICM 3,
  TE 3), plain cross-entropy. ICM/TE heads train on EXP ≥ 3 only; at prediction
  time EXP 1–2 ⇒ ICM = TE = ND. Every later change is measured against it.
- **Preprocessing:** pad to square, then the encoder's own resize (bicubic for CLIP).
- **Validation:** 15% of train, grouped by patient, stratified by EXP; linear
  heads tuned with 5-fold grouped CV. The test set is used once, at the end.
- **FEMI is excluded:** it was pretrained on this dataset's test images.
- **Compute:** this machine is CPU-only. Data prep and evaluation run locally;
  image encoding and fine-tuning run on Colab. Code syncs via a private GitHub
  repo; data lives on Google Drive and is never committed.

## Rules and lessons
_TODO._

## Open questions and next steps
_TODO._

## Related work (reference)
- Reading notes on the papers are in `papers/paper-notes.md`.
- Stage classification (time-lapse, Gomez dataset, constraint-aware Viterbi) is
  another team member's work. It is kept only for reference in
  `papers/stage-classification/`.
