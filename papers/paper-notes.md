# Paper Notes — PDFs in this folder

Notes pulled from the 5 PDFs, focused on what matters for the embryo
stage-classification pipeline (BiomedCLIP → per-frame probs → constrained Viterbi).

---

## 1. Constraint-Aware Decoding (CAD) — arXiv 2605.10149
Ee, Roy, Li, Zhang, Fernando (A*STAR / IIT Kharagpur / NTU), May 2026.
Code: https://github.com/LUNAProject22/CAD

**What it does.** A Viterbi pass after training, with no retraining needed. It works on
a T×C matrix of per-frame probabilities, so any backbone can feed it.

**Constraints (§3.1), all counted from training annotations:**
- Valid transitions 𝒯, where every transition seen in training is allowed, weighted by
  `Conf(A→B) = Count(A→B) / Count(A)` (Eq. 1). Unseen transitions are blocked.
- Valid start actions S and valid end actions E.
- Duration bounds per class `[d_min, d_max]`, **normalized by total video length T**.

**Objective (§3.2):** maximize `Σ_t log P[t,y_t] + Σ_t log Conf(y_{t-1}→y_t)` subject
to y_1∈S, y_T∈E, all transitions in 𝒯, and every segment's duration within bounds.

**Algorithm.** It tracks three matrices: V (score), D (length of the current segment)
and B (backpointer).
- Staying in class c is allowed only while `D[t-1,c]/T < d_max(c)`. It adds only
  log P, with **no Conf term**.
- Switching c′→c is allowed only if `D[t-1,c′]/T ≥ d_min(c′)` and (c′→c)∈𝒯.
  score = V[t-1,c′] + log Conf(c′→c) + log P[t,c]. D resets to 1.
- At the last frame, the best class is chosen from E with D/T ≤ d_max, then it backtracks.
- Cost is O(C·T): 2.81 s vs 148.9 s for the KARI grammar parser on 20 Breakfast videos.

**Results (Breakfast, ASFormer, F1@{10,25,50} / Edit / Acc):**
| | F1@10 | F1@25 | F1@50 | Edit | Acc |
|---|---|---|---|---|---|
| baseline | 74.1 | 68.7 | 55.5 | 72.8 | 72.4 |
| KARI | 77.3 | 71.6 | 57.0 | 78.3 | 74.5 |
| CAD | 78.8 | 73.1 | 58.0 | 77.9 | 74.6 |

**Ablations (Table 5, Breakfast):**
- Dropping **duration bounds** causes the largest F1 drop (F1@50 58.0 → 54.9).
- Dropping **transitions** mainly hurts Edit and Acc.
- Dropping start/end constraints costs little.
- **Hard constraints beat soft ones** (F1@10 78.8 vs 75.6). Soft constraints leave
  invalid paths a small nonzero probability.
- Transition and duration weights (0.3/0.7 vs 0.7/0.3) made almost no difference.
- Plain Viterbi (Conf only, no constraints) is much worse. On 50Salads semi-supervised,
  Edit is 39.2 vs 48.9.
- Multi-parent constraints (a1,a2,a3 → a4) gave no gain.

**Caveats for our use:**
- Gains show up mostly in **Edit/F1, not frame accuracy**. On 50Salads fully supervised,
  accuracy actually *fell* (85.0 → 82.9), and FACT's F1@50 fell (66.2 → 64.8). Report
  Edit and F1@50 too, since EmbryoDiff/EmbryosFormer use them.
- **Duration normalized by T is a poor fit for embryos.** Recordings stop at different
  times, so d/T means different things across embryos. Bounds in hours after
  fertilization (or frame counts at a fixed interval) are likely better. This ties
  directly to the open question about snapshot spacing.
- The DP stores only one duration per (t, c). That is a greedy approximation, not an
  exact segment-level (semi-Markov) search. With d_min enforced, some sequences can end
  with every state at −∞, so we need a fallback such as relaxing d_min.
- At the last frame, only d_max is checked, not d_min.
- The authors themselves say hard constraints get "overly restrictive when the training
  data is sparse." A skip that is legitimate but rare, and missing from our training
  split (e.g. t3→t5), will be **impossible** at test time. We could add smoothing or a
  small ε floor for adjacent-stage skips.
- The paper does not say whether Count(A) counts segments or frames. Use segment
  counts, or the Conf values for real transitions become tiny next to self-loops.
- Valid start/end sets fit embryos well: recording may start after tPB2, and arrested
  embryos never reach tB or tHB.

---

## 2. Expert-Annotated Embryo Image Dataset with Natural Language Descriptions — arXiv 2604.16528
Neu, Ebner, Primus, Schenkenfelder, Zefferer, Brunbauer, Kromp (SCCH Hagenberg +
Kepler Univ. Clinic Linz + Wunschkind Klinik Vienna), Apr 2026. Data descriptor, no model.

- **1,100 JPEG frames, a subset of Gomez et al. 2022** (the same 704-video,
  7-focal-plane EmbryoScope dataset we benchmark on). The paper does not say which
  focal plane.
- Captions cover cell cycle, stage (oocyte, zygote, cleavage, morula, blastocyst) and
  morphology: polar body, perivitelline space, zona, pronuclei, cytoplasm, cell
  number and evenness, fragmentation, compaction, expansion, ICM/TE. They follow the
  **Istanbul Consensus (2025 update)**. Annotation was done in Label Studio.
- **Gold set: 100 images.** Junior and senior embryologists annotated each one
  independently, then seniors consolidated. Stages are balanced. This is the test set.
- **Silver set: 1,000 images.** Junior and senior annotators, with more than 10% checked
  by a senior. Stage distribution is uneven. This is the training set.
  - The paper contradicts itself once ("gold-standard training set and a silver-standard
    test set"). The rest of the paper and InVitroVision both use 1,000 for training and
    100 for testing.
- Format: images plus two JSONL files (train/test).
  Figshare: https://doi.org/10.6084/m9.figshare.32024349.v1
- No baseline numbers in this paper. It points to InVitroVision.

**Relevance:** these captions could replace our hand-written stage prompts, or be used
to fine-tune BiomedCLIP's text side. **Leakage risk:** the frames come from Gomez videos.
If we train on these captions and evaluate on a Gomez split, remove videos that appear
in both. The IDs look like "AAL839-6 WELL6 RUN027".

---

## 3. InVitroVision — arXiv 2604.21061
Same group as #2, Apr 2026.

**Setup:**
- Base model: PaliGemma-2 with **LoRA on the language layers and the vision-language
  connector only. The vision encoder stays frozen.**
- Input is 448×448 with padding to keep the aspect ratio. Prompt: `<image>describe en`.
  Optimizer is AdamW with gradient accumulation.
- Models were trained on 400, 700 and 1,000 images (IVV-400/700/1000). Baselines are the
  base PaliGemma-2 and ChatGPT 5.2.

**Metric (custom, from experts).** BLEU, ROUGE and CIDEr were rejected because they
don't measure clinical correctness.
- ER, embryo recognized (0/1). If it is 0, the whole score is 0.
- ECC, cell cycle (0–3), weight α = 4.
- MD, morphological detail (0–4), weight β = 2.
- PD, positional detail (0–3), weight γ = 1. It is left out when not applicable.
- Score is the weighted sum of normalized parts, in [0, 1]. Two embryologists each
  scored 5 models on **50 images**.

**Results (Table 1):**
| Model | ECC /3 | MD /4 | PD /3 | Total |
|---|---|---|---|---|
| Base | 0.93 | 0.97 | 0.99 | 0.29 |
| IVV-400 | 2.04 | 1.89 | 0.98 | 0.57 |
| IVV-700 | 2.14 | 2.17 | 0.86 | 0.60 |
| IVV-1000 | 2.30 | 2.52 | 1.03 | 0.67 |
| ChatGPT 5.2 | 1.87 | 1.62 | 0.98 | 0.52 |

The text gives slightly different totals (0.66 / 0.56 / 0.50) than the table.

**Per-stage totals for IVV-1000 (Table 2):**
- Strongest: t2 0.79, t6 0.79, t7 0.77, t8 0.80, tEB 0.76.
- **Weakest: tM 0.48, tPB2 0.52, t4 0.55, tSB 0.60.**
- ChatGPT falls apart after compaction: tB 0.18, tEB 0.29.

**What this does and doesn't support for us:**
- It shows that small-data LoRA adaptation works *even with a frozen vision encoder*.
  That is a cheap first experiment for BiomedCLIP too (adapt text/projection only).
- It is **not a classification accuracy result**. "~1,000 pairs is enough" means a
  captioning score of 0.67/1 from 50 expert-scored images, not 16-stage accuracy.
  Don't cite it as a direct comparison with DLT-Embryo or EmbryoDiff.
- The same hard stages show up here (morula/compaction, t9+, tSB) as in the
  classification literature. Expect our confusions to cluster there.
- It mentions a follow-up (Kromp et al. 2026b) claiming IVV-1000 embeddings are good for
  downstream tasks. It could be an alternative encoder if we can get the weights.

---

## 4. Multimodal Learning for Embryo Viability Prediction — arXiv 2410.15581
Kim, Shi, Jeong, Knittel, Yang, Song, Li, Li, Ben-Yosef, Needleman, Pfister
(Harvard / CMU / Flatiron / Tel Aviv Sourasky), Oct 2024.

**Data:**
- 3,695 cycles and 24,027 embryos, imaged every 20 min for 5 days at 500×500 (about 6M
  images).
- Only **3,318 embryos in 1,700 cycles** have both video and EHR. Of those cycles,
  **260** were successful (at least one live birth).
- Split 8:1:1, stratified by success. Train: 2,617 embryos / 1,360 cycles.

**Model:**
- ViViT-style factorized transformer. The spatial part is **pretrained DeiT-Ti, frozen**;
  fine-tuning it made results *worse* because labels were scarce.
- 4 temporal/multimodal layers are trained from scratch.
- Video is capped at 360 frames and subsampled every 4th frame, giving 90 frames at
  224×224.
- EHR and other tabular features become extra tokens.
- Loss is Huber on viability = births / embryos transferred.
- A two-stage alternative feeds Embryo-vision and BlastAssist features into
  TabTransformer or TabNet.

**Results (embryo-level AUC):**
| Input | AUC |
|---|---|
| video only | 0.578 |
| video + EHR | 0.580 |
| video + Embryo-vision features | 0.676 |
| Embryo-vision features alone | 0.666 |
| Embryo-vision + EHR + BlastAssist (no raw video) | **0.688** |
| Two-stage, EHR + BlastAssist | ~0.63 |

At treatment level, Embryo-vision features alone score best (0.697).

**Relevance:**
- This is **independent evidence for dropping temporal attention**. Attention layers
  trained from scratch on a few thousand labeled sequences underperform, and adding raw
  video hurts.
- Embryo-vision's features include **stage classification**, and those structured
  outputs beat raw pixels. So a good stage classifier is a useful upstream feature if
  the project later moves to viability prediction.
- Their fix, self-supervised pretraining, is proposed here but not tested (see #5).

---

## 5. ⚠️ "Self-supervised Pre-training for Robust and Generic Spatial-Temporal Representations" — WRONG PAPER (PDF deleted)
This PDF was **Hu, Zhong, Zhang, Li, Xie, Jia, Zhou, Luo — IEEE ICDM 2023**
(DOI 10.1109/ICDM58522.2023.00024). It is about **taxi GPS trajectories in Shenzhen**
(driver identification, seeking vs. serving). It never mentions embryos. It is also
called "STPT", which is probably how it got mixed up.

The paper CLAUDE.md means is:
**arXiv 2506.17403 — "Spatial-Temporal Pre-Training for Embryo Viability Prediction
Using Time-Lapse Videos"**, by Zhiyi Shi, Junsik Kim, Helen Y. Yang, Yonghyun Song,
Hyun-Jic Oh, Dalit Ben-Yosef, Daniel Needleman, Hanspeter Pfister (June 2025). Same
Harvard group as #4. According to its abstract, it deals with memory limits and
temporal misalignment in long embryo videos, using over 23,000 videos (about 3,286
labeled). The wrong PDF has been deleted; download this one if needed.

For reference only, the ICDM paper has little to offer our project. It pretrains a
transformer on GPS sub-trajectory pairs with a same-source/different-source objective,
then fine-tunes. Accuracy is 83.1% on 8-driver classification and 77.9% on
seeking vs. serving.
