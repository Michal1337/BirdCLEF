# BirdCLEF 2026 Score Improvement Ideas (Code Review + Online Research)

Current baseline: **0.845 ROC AUC**  
Date: **2026-03-30**

## 1) Executive Summary

Your current pipeline is already solid for a custom-from-scratch baseline, but it is missing the techniques that repeatedly drove top BirdCLEF performance in 2024-2025:

1. **Iterative pseudo-labeling on unlabeled soundscapes** (highest expected gain).
2. **Class-imbalance-aware training** (sampling + loss), not plain BCE only.
3. **Stronger transfer learning backbones / embeddings** (BirdNET/Perch/BirdSet-style features or timm CNN backbones).
4. **OOF/GroupKFold validation and model diversity** to reduce LB shake.
5. **Inference-time postprocessing/calibration + TTA/ensembling**.

If you execute only the highest-ROI block first, a realistic improvement range is roughly **+0.01 to +0.03 ROC AUC** from 0.845, with upside if pseudo-labeling is done well.

## 2) Review of Current Code (What Helps, What Limits You)

### What is good

- Clean DDP scaffolding and BF16 autocast in training (`birdclef_example/train_ddp.py`).
- Reasonable spectrogram settings and transformer encoder head (`birdclef_example/model.py`).
- Correct BirdCLEF-style macro ROC AUC computation that ignores empty classes (`birdclef_example/utils.py`).
- Soundscape-aware validation split by filename (helps leakage control).

### Main bottlenecks in your current implementation

1. **No class-balancing mechanism**
- Training uses plain shuffled loader and plain `BCEWithLogitsLoss()`.
- BirdCLEF label distribution is heavily imbalanced; this usually under-trains rare species.

2. **No pseudo-labeling loop**
- You are training once on labeled data only.
- Top solutions repeatedly leverage unlabeled soundscapes via pseudo-labeling/distillation.

3. **Weak augmentation stack currently active**
- Dataset waveform augmentations exist in code, but in `train_ddp.py` they are effectively left at defaults (mostly zero-probability).
- You mostly rely on SpecAugment only.

4. **Single-split, no fold diversity**
- One holdout split can overfit split idiosyncrasies and reduce leaderboard robustness.

5. **Inference is minimal**
- Basic sigmoid predictions per 5-second chunk; no calibration, TTA, no co-occurrence/postprocessing.

6. **Loss does not handle noisy/fuzzy labels explicitly**
- Secondary labels and pseudo labels are noisy; BCE-only often underperforms BCE+Focal/ASL in this setup.

## 3) External Evidence (BirdCLEF 2024/2025 Patterns)

### A) Pseudo-label + distillation keeps showing up

- 3rd-place BirdCLEF 2024 team repos (Theo/CPMP) explicitly describe a two-stage setup: train first-level models, generate pseudo labels on unlabeled soundscapes, retrain second-level models with mixed real+pseudo data.
- They also report additive mixup and diverse model families.

### B) 2025 high leaderboard results relied on transfer + pseudo labels + postprocessing

From the BirdCLEF+ 2025 case study (Sydorskyi & Gonçalves):
- Class-imbalance-aware sampling (`gamma=-0.5` weighting scheme),
- BCE + Focal objective,
- transfer learning from large birdcall corpora,
- iterative pseudo-labeling,
- and postprocessing producing an additional uplift.

### C) CPU-time-constrained BirdCLEF pipelines emphasize inference optimization

2024/2025 writeups repeatedly mention model compilation/inference optimization (ONNX/OpenVINO/TFLite). Even if 2026 rules differ, fast inference increases freedom for ensembling/TTA.

## 4) Prioritized Improvement Roadmap (Highest ROI First)

## Phase 1 (Do first): High impact with moderate engineering

1. **Add class-balanced sampling + weighted loss**
- Implement per-class sample frequency and per-sample weights.
- Use `WeightedRandomSampler` (or custom balanced sampler in DDP).
- Replace BCE-only with **BCE + Focal** (or ASL).
- Keep one config flag to quickly ablate (`loss_mode = bce | bce_focal | asl`).

2. **Turn on real waveform augmentation policy**
- In `BirdCLEFDataset` call from `train_ddp.py`, pass non-zero probs:
  - `waveform_aug_prob`, `gain_prob`, `noise_prob`, `time_shift_prob`, `drop_segment_prob`.
- Add **mixup** (recommended: additive/union-label style as seen in strong BirdCLEF solutions).

3. **Validation hardening**
- Move from one split to `GroupKFold` by soundscape filename (or site/recording group if metadata allows).
- Use OOF AUC for model selection instead of single split AUC.

Expected effect from Phase 1 alone: often meaningful on rare species and LB stability.

## Phase 2: Pseudo-labeling pipeline (largest expected gain)

1. Train first-level model ensemble (3-5 diverse seeds/backbones).
2. Predict on unlabeled soundscapes in 5-second windows.
3. Keep pseudo labels with confidence filtering:
- per-class threshold or top-k strategy,
- optional temperature calibration before thresholding.
4. Retrain second-level model on **balanced mix of real + pseudo** data.
- Start with pseudo batch ratio around 0.3-0.6 and sweep.
5. Optionally run iterative refresh (I1 -> I2) of pseudo labels.

This is the single most proven path to break through a strong baseline.

## Phase 3: Backbone and feature upgrades

1. Add a `timm` backbone option (EfficientNetV2 / ConvNeXt / EfficientViT-small family).
2. Try embedding-based head from BirdNET/Perch style features as an additional model family.
3. Keep your current transformer head as one ensemble member for diversity.

## Phase 4: Inference/postprocessing

1. **TTA over time offsets** (e.g., center + shifted crops).
2. **Per-class calibration** with OOF predictions (Platt/temperature or isotonic-lite).
3. **Co-occurrence-aware smoothing** (light graph prior; do not over-regularize).
4. Ensembling with diversity-first weighting (different seeds/losses/backbones).

## 5) Concrete Changes to Your Existing Files

### `birdclef_example/train_ddp.py`

- Add config switches for:
  - `loss_mode`, `focal_gamma`, `focal_alpha`,
  - balanced sampling strategy,
  - pseudo-label dataset path + mixing ratio,
  - EMA, and optional SWA.
- Replace manual metadata sharding + shuffle with a proper DDP sampling strategy that supports balancing.
- Save OOF-friendly metadata/predictions for calibration and pseudo-label selection.

### `birdclef_example/data.py`

- Implement:
  - class-frequency statistics,
  - per-sample weights,
  - mixup/cutmix-like audio augmentation,
  - pseudo-label quality flags and filtering.
- Add support for confidence-weighted targets (soft labels for pseudo data).

### `birdclef_example/model.py`

- Keep current model, but add selectable backbones:
  - `simple_transformer` (current),
  - `timm_cnn` (EffNet/ConvNeXt/EfficientViT-like),
  - optional embedding-head mode.

### `birdclef_example/predict.py`

- Add:
  - TTA offsets,
  - ensembling across checkpoints,
  - calibration layer,
  - optional light postprocessing.

## 6) Suggested Experiment Matrix (Ordered)

1. **Exp A1**: BCE -> BCE+Focal (same everything else).
- Goal: improve rare-class recall/ranking.

2. **Exp A2**: Turn on waveform augs + time shift + background noise.
- Goal: domain robustness.

3. **Exp A3**: Balanced sampling in training loader.
- Goal: reduce head-class dominance.

4. **Exp B1**: 3-seed ensemble (same architecture), average logits.
- Goal: cheap diversity.

5. **Exp B2**: Pseudo-label v1 (single refresh, strict threshold).
- Goal: add unlabeled supervision.

6. **Exp B3**: Pseudo-label v2 (iterative refresh + confidence weighting).
- Goal: push beyond v1.

7. **Exp C1**: Add one timm CNN backbone and ensemble with current model.
- Goal: architecture diversity.

8. **Exp D1**: Inference calibration + TTA.
- Goal: leaderboard alignment and robust ranking.

Keep every run logged with:
- CV/OOF AUC,
- public LB AUC (if submitted),
- train time/inference time,
- keep/discard decision.

## 7) Practical Guardrails

- Avoid adding huge external datasets blindly; 2025 evidence shows too much additional data can hurt due to domain mismatch.
- Keep pseudo labels conservative first (high precision > high recall).
- Prefer 2-3 strong diverse models over many similar ones.
- Treat CPU inference budget as a first-class constraint if 2026 rules retain similar limits.

## 8) What I’d Do Next (Minimal High-Impact Sequence)

1. Implement **BCE+Focal + balanced sampler + active waveform augs**.
2. Switch to **GroupKFold/OOF** validation artifact generation.
3. Build **pseudo-label v1** (single iteration, strict confidence).
4. Add one **timm EfficientNetV2-S** branch for diversity.
5. Add **TTA + calibration** in inference.

If you want, I can implement this roadmap in your codebase in small safe steps, one experiment at a time, with an updated `experiment_log.md` template ready for your next run.

## Sources

- BirdCLEF+ 2025 case study (transfer + pseudo-labeling + postprocessing): https://ceur-ws.org/Vol-4038/paper_256.pdf
- BirdCLEF 2025 lightweight/efficiency notebook (CPU budget, transfer baselines): https://ceur-ws.org/Vol-4038/paper_249.pdf
- BirdCLEF 2025 one-detector-per-bird notebook (imbalance/calibration observations): https://ceur-ws.org/Vol-4038/paper_254.pdf
- BirdCLEF 2024 DS@GT transfer + pseudo multi-label notebook: https://ceur-ws.org/Vol-3740/paper-202.pdf
- BirdCLEF 2024 methods paper (augmentation + inference-time optimization): https://ceur-ws.org/Vol-3740/paper-193.pdf
- BirdCLEF 2024 3rd-place solution repo (Theo): https://github.com/TheoViel/kaggle_birdclef2024
- BirdCLEF 2024 3rd-place solution repo (CPMP): https://github.com/jfpuget/birdclef-2024
- BirdCLEF+ 2025 2nd-place solution repo: https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- BirdCLEF 2026 Kaggle overview page (accessible URL, content blocked in this environment): https://www.kaggle.com/competitions/birdclef-2026/overview

