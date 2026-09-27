# Published baselines, reproduced

`kromp_models.csv` = all 12 models whose predictions the dataset authors released
(github.com/software-competence-center-hagenberg/Blastocyst-Dataset, `model_predictions/`),
scored by `src/evaluate.py` on the 300-image gold test set.

Regenerate:
```
python src/evaluate.py data/baselines/kromp_predictions/*/pred.csv --name-from-parent --out results/baselines/kromp_models.csv
```

## Reproduction check (Kromp et al. 2023, Table 2) — passes

| Model | Published acc EXP / ICM / TE | Ours, kromp protocol | Published weighted F1 | Ours |
|---|---|---|---|---|
| Xception | 0.78 / 0.69 / 0.62 | 0.780 / 0.688 / 0.617 | 0.77 / 0.65 / 0.56 | 0.77 / 0.65 / 0.56 |
| DeiT | 0.82 / 0.63 / 0.58 | 0.820 / 0.634 / 0.583 | 0.81 / 0.52 / 0.51 | 0.81 / 0.52 / 0.51 |
| Swin | 0.85 / 0.65 / 0.62 | 0.847 / 0.653 / 0.619 | 0.84 / 0.56 / 0.55 | 0.84 / 0.56 / 0.55 |

## The two protocols

**kromp** (the authors' `evaluate_prediction_results.py`): ICM/TE are scored only
where both the gold and the *predicted* EXP are grade ≥ 3, so each model is
scored on its own subset (Xception 240, Swin 239, DeiT 235 images). Missing gold
labels are scored as extra classes rather than excluded.

**clean** (ours): the same images for every model; missing gold labels excluded
per grade; ND is a class, so an EXP error that wrongly triggers ND counts
against ICM/TE. `*_given_exp` scores ICM/TE only where the gold EXP is grade ≥ 3.

**Cross-paper warning:** Embryo2live's quoted baselines (EXP 85.2, TE 67.4)
match our *clean* protocol for Swin (0.852, 0.674), not the published *kromp*
numbers. Its ICM baseline of 67.1 (attributed to Xception) matches DeiT under
the clean protocol (0.671), not Xception (0.723). So published "improvements"
over the Kromp baselines are not all measured the same way. MDMBG-Net's
protocol is unknown until its full text is read.

## Other findings

- **The repo holds stronger models than the paper reports.** Best under the
  kromp protocol: `convnext_tiny_384` 0.853 / 0.721 / 0.621 and
  `convnextv2_tiny_384` 0.840 / 0.726 / 0.629. Under the clean protocol,
  `convnext_tiny_384` reaches 0.859 / 0.753 / 0.674.
- **Macro F1 exposes the rare grades.** ICM/TE macro F1 is only 0.25–0.39 under
  the kromp protocol — models almost never predict grade C correctly — while
  accuracy looks reasonable. This is why we report macro F1 alongside accuracy.
