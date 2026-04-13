# V18x Pipeline Review: Potential Issues and Improvements

## Context
This review targets the end-to-end approach in [birdclef_example/v18x_dmodel.py](birdclef_example/v18x_dmodel.py), with emphasis on competition practicality, generalization risk, and maintainability.

## 1) High-Impact Risks

### 1.1 Training happens inside the submission script
Issue:
- The script trains ProtoSSM, per-class MLP probes, and ResidualSSM during submission generation.

Why this is risky:
- This is expensive and can be fragile under Kaggle runtime constraints.
- Re-running training every submission increases variance and reproducibility issues.

Improvement:
- Split pipeline into two stages:
  - Offline train stage: fit ProtoSSM, probe models, ResidualSSM, scalers, PCA, priors.
  - Online inference stage: load frozen artifacts and run only test inference and post-processing.
- Save artifacts using clear versioned paths under outputs or models.

---

### 1.2 Potential overfitting from second-pass residual training
Issue:
- ResidualSSM learns corrections using first-pass predictions produced by models trained on the same cached dataset.

Why this is risky:
- Residual model may learn train-specific error patterns that do not transfer to hidden test.

Improvement:
- Train residual model on strict out-of-fold first-pass predictions only.
- Keep a held-out validation fold for residual stacking and report uplift before enabling at inference.

---

### 1.3 No explicit model artifact checkpointing for reuse
Issue:
- Trained models and preprocessing transforms are not persisted for stable reuse in future runs.

Why this is risky:
- Hard to reproduce a best configuration.
- Hard to compare changes cleanly across experiments.

Improvement:
- Persist:
  - ProtoSSM state_dict
  - ResidualSSM state_dict
  - probe models
  - StandardScaler and PCA
  - prior tables and class mappings
  - all config values used in the run

## 2) Statistical and Validation Concerns

### 2.1 Single-seed training and single final model
Issue:
- One seed and one trained temporal model are used.

Why this is risky:
- Results may be unstable from run to run.

Improvement:
- Add a lightweight seed ensemble (for example 2 to 3 seeds) for ProtoSSM and average logits.
- Keep only if mean validation improves versus compute cost.

---

### 2.2 Some configured knobs appear unused
Issue:
- Several config fields appear defined but not active in training logic.

Examples:
- proto_ssm.n_prototypes
- proto_ssm_train.val_ratio
- proto_ssm_train.proto_margin
- proto_ssm_train.swa_lr
- proto_ssm_train.use_cosine_restart
- proto_ssm_train.restart_period
- threshold_grid and optimize_per_class_thresholds are present but final thresholds are fixed at 0.5

Why this is risky:
- Tuning becomes confusing and may lead to incorrect assumptions about what changed.

Improvement:
- Remove dead knobs or implement them fully.
- Add a config audit printout listing active and inactive keys at startup.

---

### 2.3 Limited calibration strategy
Issue:
- Post-processing uses fixed class temperatures by coarse taxon group.

Why this is risky:
- Calibration may be suboptimal per class, especially with severe class imbalance.

Improvement:
- Fit per-class temperature or isotonic-like calibration on OOF predictions.
- Gate calibration by minimum positives to avoid unstable classes.

## 3) Data and Mapping Risks

### 3.1 Scientific-name mapping brittleness
Issue:
- Label mapping depends on exact scientific-name matching plus a manual map placeholder.

Why this is risky:
- Name changes, punctuation differences, or taxonomy updates can silently reduce mapped coverage.

Improvement:
- Add a mapping diagnostics report:
  - mapped count
  - unmapped active count
  - top unmapped classes by frequency
- Enforce fail-fast thresholds if mapped coverage drops unexpectedly.

---

### 3.2 Proxy class strategy may introduce bias
Issue:
- Unmapped classes use genus proxies from Perch labels.

Why this is risky:
- Genus-level substitutions can inflate related false positives and blur species boundaries.

Improvement:
- Validate proxy benefit class-by-class on OOF.
- Disable proxies that reduce per-class AUC or precision at top-k.

## 4) Engineering and Runtime Concerns

### 4.1 CPU-only path may be too slow for iterative work
Issue:
- The script forces CPU inference for TF and uses CPU torch training.

Why this matters:
- Good for compatibility, but iteration time can become large.

Improvement:
- Add a hardware mode flag:
  - safe_cpu mode for submission compatibility
  - accelerated mode for local/offline training when GPU is available

---

### 4.2 Large monolithic script increases maintenance risk
Issue:
- One file handles data setup, training, inference, fusion, and submission.

Why this is risky:
- Harder to test and reason about regressions.

Improvement:
- Modularize into components:
  - data preparation
  - Perch interface
  - temporal models
  - probe training
  - fusion and post-processing
  - artifact I/O

---

### 4.3 Repeated tensor conversions and full-batch training patterns
Issue:
- Training loops repeatedly convert large arrays to tensors and run effectively full-batch steps.

Why this is risky:
- Memory pressure and slower training for larger cache sizes.

Improvement:
- Move to minibatch DataLoader pipelines for ProtoSSM and ResidualSSM.
- Cache tensors once where practical.

## 5) Recommended Prioritized Roadmap

### Priority A: Make it competition-practical
1. Split train and inference scripts.
2. Save and reload all artifacts.
3. Keep submission script inference-only.

### Priority B: Improve generalization reliability
1. Train residual stacker on OOF first-pass predictions.
2. Add seed averaging for ProtoSSM if compute allows.
3. Add per-class calibration using OOF predictions.

### Priority C: Increase transparency and debuggability
1. Add mapping diagnostics and fail-fast checks.
2. Remove or activate dead config knobs.
3. Emit structured run summary JSON with all active settings and key metrics.

## 6) Quick Wins With Low Refactor Cost
- Add deterministic controls for TF where possible in addition to numpy/torch seeds.
- Save timing breakdown per phase to identify true bottlenecks.
- Report per-class uplift from each stage:
  - Perch baseline
  - +prior fusion
  - +ProtoSSM
  - +MLP probes
  - +ResidualSSM
- Disable components that do not show consistent OOF benefit.

## 7) Suggested Success Criteria for Next Revision
- Inference-only submission path completes within target runtime budget.
- Reproducible artifacts recreate identical predictions for a fixed seed and data snapshot.
- Residual stage shows positive OOF uplift before being enabled.
- Final ensemble improves macro AUC and is not driven by regressions on rare classes.
