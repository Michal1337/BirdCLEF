# BirdCLEF 2026 — Strategy v2

> **Status snapshot (2026-04-25)**: Legacy SSM notebook ([LB_093.ipynb](LB_093.ipynb)) at **0.92–0.93 public LB** (full-data trained, real). Single-fold SED at **0.738 public LB** despite v_anchor 0.89. Pseudo-label round 1 complete; SED 5-fold ensemble untested on LB. Soundscape memmap built; SED training infrastructure stable.
>
> **CRITICAL**: the 0.93 LB came from the *legacy* SSM stack (`lambda_prior=0.4`, `correction_weight=0.30`, `pca_dim=64`, adaptive smoothing + per-class threshold sharpening). The "cleaned" port in [birdclef/train/train_ssm_head.py](birdclef/train/train_ssm_head.py) (`lambda_prior=0`, `correction_weight=0`, `threshold=[0.5]`) has only been validated on OOF/V-anchor — **never on public LB**. Do not assume the two are interchangeable.

This supersedes [STRATEGY.md](STRATEGY.md). The pivot from v1: **SSM is the workhorse, SED is a candidate ensemble member, not a replacement.**

## 1. What changed since v1

### What worked
- **Site×date GroupKFold + V-anchor split** ([birdclef/data/splits.py](birdclef/data/splits.py)): cleaned up the leaky filename-grouping; gave honest per-fold/per-site metrics.
- **Legacy SSM submission notebook** ([LB_093.ipynb](LB_093.ipynb)): unchanged from the day it scored 0.93 LB. Pipeline = ProtoSSM (40 ep, lr=1e-3) + site/hour prior @ `lambda_prior=0.4` + MLP probes (pca_dim=64, hidden=(128,64)) blended at `ENSEMBLE_W=0.5` + ResidualSSM correction @ `correction_weight=0.30` + per-class temperatures + file-confidence + rank-aware + adaptive-delta smooth + isotonic+threshold sharpening over `[0.25..0.70]`.
- **SSM port cleanup ON OOF ONLY** ([birdclef/train/train_ssm_head.py](birdclef/train/train_ssm_head.py)): setting `lambda_prior=0`, `correction_weight=0`, `threshold_grid=[0.5]` lifted honest OOF from 0.71 → 0.83. **Not LB-validated** — the new port has never been deployed to Kaggle. Do not assume parity with the legacy notebook.
- **Pseudo-labeling pipeline** ([birdclef/train/pseudo_label.py](birdclef/train/pseudo_label.py)): SSM 3-seed ensemble teacher → SED student via `cache/pseudo/round1/`. End-to-end works.
- **SED DDP trainer** ([birdclef/train/train_sed_ddp.py](birdclef/train/train_sed_ddp.py)): EfficientNetV2-S, FocalBCE, mixup, EMA, fold/V-anchor periodic eval, bf16 on H100.
- **Soundscape memmap cache** ([birdclef/cache/build_soundscape_cache.py](birdclef/cache/build_soundscape_cache.py)): ~5× faster epochs. Necessary for round-2 iteration.

### What didn't
- **SED single-fold submission underperforms SSM by ~0.19 on public LB** (0.738 vs ~0.93). V-anchor over-estimated true generalization by ~0.16.
- **FP16 ONNX export tripped multiple bugs**: complex STFT, version-converter Pad, custom op_block_list overwriting defaults, BN-buffer EMA-load. We resolved all of them but the conclusion is "ship FP32".
- **My initial table treated 0.93 as "leaky"** — wrong. The submitted SSM model was trained on all labeled data; the OOF leakage only affected eval, not the deployed weights. Mea culpa.

### Key lesson
**Trust public LB over V-anchor when they disagree by more than ~0.05.** V-anchor is 13 small-distribution files; public LB is 600 broader-distribution files. For configs, V-anchor is fine for ranking. For *absolute* generalization, only LB tells the truth.

**Corollary applied to the SSM port**: V-anchor said `prior=0` and `correction=0` were fine. We extrapolated "no LB regression" from that. We have no actual LB measurement of the cleaned config, so we are flying blind on whether `prior=0.4 → 0` would cost 0.00 or 0.05 LB. Treat the legacy notebook as the load-bearing artifact.

## 2. Honest current numbers

| Pipeline | Train data | Honest V-anchor | Public LB |
|---|---|---:|---:|
| **SSM legacy ([LB_093.ipynb](LB_093.ipynb), prior=0.4, corr=0.30)** | all 59 labeled | not tracked | **0.92–0.93** |
| SSM cleaned port (`train_ssm_head.py`, prior=0, corr=0) | all 59 labeled | 0.83 (proxy) | untested — do not assume = 0.93 |
| SED fold 0 (round 1) | ~37 labeled + train_audio + pseudo | 0.89 | **0.738** |
| SED 5-fold ensemble (round 1) | per-fold | mean 0.865 | untested, project ~0.75–0.78 |
| SSM legacy + SED 5-fold blend | as above | untested | **target 0.93–0.95** |
| + Pseudo round 2 SED | + better pseudo-labels | untested | target 0.94–0.96 |

The honest ceiling for "what we can ship without major work" is roughly **0.93–0.95**. Beyond that needs a different backbone or many more pseudo rounds.

## 3. The pivot — SSM-primary, SED-supplementary

Why SED isn't competitive on its own:
1. **Data scale**: 46 labeled soundscapes is too few to train a CNN backbone from scratch even with 10 600 pseudo-labeled files. SSM piggy-backs on Perch's broad pretraining and only needs a small head.
2. **Domain mismatch**: SED trained 50/50 on Xeno-Canto focal recordings + soundscape pseudo-labels. Public test is soundscapes only — different acoustic statistics than focal.
3. **Pseudo-label noise**: round 1 pseudo-labels came from SSM; the SED student inherits SSM's biases without SSM's strengths (Perch features).

Why SED is still worth keeping:
1. **Different feature space** → decorrelated errors in an ensemble. Even at lower individual quality, blending often gives +0.02–0.05 when members are different model families.
2. **Round 2 ceiling**: with the SED 5-fold ensemble as a stronger teacher, round 2 pseudo-labels will be cleaner; the round-2 SED student might actually exceed SSM eventually.

## 4. Action plan, ranked by ROI

### A. Confirm SSM 0.93 still holds (30 min)

Re-submit [LB_093.ipynb](LB_093.ipynb) **as-is** (do not edit, do not refactor) to the *current* public test (the leaderboard might have been re-cut since your last SSM submission). Set `MODE = "submit"` (already the default), attach the BirdCLEF 2026 competition data + the Perch v2 model dataset (paths it expects: `/kaggle/input/competitions/birdclef-2026` and `/kaggle/input/models/google/bird-vocalization-classifier/...`), run, submit.

If LB confirms ≥ 0.92, **that's the floor**. Any future submission that scores below this is a regression.

Do NOT substitute the new `birdclef/train/train_ssm_head.py` pipeline for this step — that's a separate variable we have no LB data on.

### B. Test SED 5-fold ensemble locally (5 min)

Bold ensemble against V-anchor with the new convenience flag:

```bash
python -m birdclef.scripts._09b_test_submission_local \
    --sed-folds-glob 'birdclef/models_ckpt/sed/sed_v2s/fold*/best.onnx' \
    --source anchor
```

Expected: `macro_auc` in 0.88–0.91 range (mean fold v_anchor was 0.865, ensemble usually adds +0.01–0.02). If it lands here, the ensemble works as expected.

### C. Build SSM+SED ensemble inference template (~3–4 h)

The current Kaggle inference template only handles SED ONNX. Extend it to also load and run the SSM pipeline. **Reproduce the legacy LB_093.ipynb knobs verbatim** — anything else is an unverified variable.

1. **Perch** loads via the same ONNX session the legacy notebook uses (`perch_v2_cpu/1` SavedModel → ONNX upgrade) — gives 1536-d embeddings + 234-d Perch logits per 5-s window.
2. **Genus proxy mapping** for the unmapped species (cell 6 in `LB_093.ipynb`): max-aggregate Perch logits over genus matches. Persist the `proxy_map.json` once and bundle it.
3. **Site/hour prior tables** trained on labeled `train_soundscapes` (cell 11), applied with `lambda_prior=0.4`. Bundle as `prior_tables.npz`.
4. **MLP probes** trained per-species with `pca_dim=64`, `hidden=(128,64)`, `min_pos=5`, `alpha_blend=0.4` (cell 14). Persist via `joblib` to `probes.pkl`. Apply via the vectorized batched matmul (cell 15) for speed.
5. **LightProtoSSM** trained 40 ep, `lr=1e-3`, blended with prior+MLP-adjusted Perch at `ENSEMBLE_W=0.5`. TTA via circular shifts `[0, 1, -1, 2, -2]` (cell 20).
6. **ResidualSSM** trained on `(Y - sigmoid(first_pass))` MSE for 30 ep, applied with `correction_weight=0.30`.
7. Per-class temperature divide → sigmoid.
8. Post-processing: `file_confidence_scale(top_k=2, power=0.4)` → `rank_aware_scaling(power=0.4)` → `adaptive_delta_smooth(base_alpha=0.20)`.
9. Per-class threshold sharpening with the F1-optimised thresholds from isotonic calibration (grid `[0.25..0.70]`).
10. Blend final SSM probs with SED-ensemble probs at recipe weights.

Total Kaggle runtime estimate: ~35–45 min (Perch dominates; SSM head + MLP probes < 5 min; SED ensemble ~10–15 min).

Cheaper alternative: keep the legacy SSM pipeline as a **separate** notebook `submission_ssm_only.ipynb` (= today's `LB_093.ipynb`), and write a *combined* notebook that runs both `LB_093.ipynb`'s outputs AND the SED ensemble, then averages the per-row probabilities. This avoids re-implementing the SSM stack inside `birdclef/submit/`. Saves ~3 h of dev time at the cost of running Perch once per notebook (still well under the 8 h Kaggle budget if both get their own notebook).

Files to write (option 1 — full integration):
- `birdclef/submit/ssm_inference.py` — Perch + prior + MLP probes + ProtoSSM + ResidualSSM + post-proc, pure PyTorch CPU. Load knobs from a shared `birdclef/config/ssm_legacy.py` so the same constants are used to *retrain* and to *infer*.
- Update `birdclef/submit/inference_template.py` to call SSM, SED, then blend.
- Update `birdclef/submit/build_notebook.py` to bundle Perch ONNX + prior tables + MLP probes pkl + ProtoSSM .pt + ResidualSSM .pt + per-class thresholds + temperatures + SED ONNX paths.

Files to write (option 2 — keep notebooks separate, blend post-hoc):
- `birdclef/submit/blend_csvs.py` — runs both notebooks, averages their `submission.csv` row-wise. Bonus: easier to A/B which member helps.

### D. Search the SSM/SED blend weight on V-anchor (~30 min)

Once C is done, run blend-weight grid:

```bash
python -m birdclef.scripts._08_ensemble \
    --members <ssm_probs.npz> <sed_ensemble_probs.npz> \
    --y-true <vanchor_y.npy> \
    --meta <vanchor_meta.parquet>
```

Expected winning weights: 70–90% SSM, 10–30% SED. If best blend > 100% SSM by ≥0.005 V-anchor → submit it. Else stick with pure SSM.

### E. Pseudo round 2 — only if D shows SED+SSM blend helps (~10 h)

```bash
python -m birdclef.scripts._05_pseudo_label --teacher sed --round 2 --tau 0.5 \
    --ckpts birdclef/models_ckpt/sed/sed_v2s/fold*/best.pt
python -m birdclef.scripts._05b_refilter_pseudo --round 2 --tau 0.7 --inplace

for f in 0 1 2 3 4; do
  torchrun --standalone --nproc_per_node=2 \
    -m birdclef.scripts._06_train_sed_student \
    --config sed_v2s --fold $f --pseudo-round 2 \
    --override batch_size=64 lr=1.5e-3 amp_dtype=bf16 num_workers=3
done
```

With the soundscape memmap, each fold trains in ~1.5–2 h. Total ~10 h. Only worth it if the SED branch is contributing in (D).

### F. Train ECA-NFNet-L0 as a second SED backbone (~10 h, optional)

Cheap diversity addition. Same SED training pipeline, different config:

```bash
torchrun --standalone --nproc_per_node=2 \
  -m birdclef.scripts._06_train_sed_student \
  --config sed_v2s --fold 0 --pseudo-round 1 \
  --override backbone=eca_nfnet_l0 batch_size=64 lr=1.5e-3 amp_dtype=bf16 num_workers=3
```

Repeat for folds 1–4. 5-fold NFNet ensemble + 5-fold EffNet ensemble + SSM = 11 members. Worth doing only if (D) confirmed SED-family adds value.

## 5. Decision rules

- **If SSM 0.93 confirmed and bold SED ≥ 0.78 on V-anchor**: build SSM+SED blend (step C). Expected +0.01–0.04 LB.
- **If SED bold V-anchor < 0.85**: SED is a stronger drag than help; don't blend. Stick with pure SSM, optionally do round 2 to lift SED before re-trying blend.
- **If anything submitted regresses below 0.88 LB**: roll back to last-known-good SSM submission immediately.
- **Stop iterating on hyperparameters.** Three sweeps already showed the SSM pipeline is at its data-limited ceiling. Time goes into: (a) ensemble engineering, (b) more pseudo-label rounds, (c) different backbone.

## 6. Updated honest LB ceiling

With the moves above:

| Step | Cumulative LB target |
|---|---:|
| SSM only (re-confirmed) | 0.93 (anchored) |
| SSM + SED blend (round 1 SED) | 0.93–0.95 |
| + Pseudo round 2 SED + re-blend | 0.94–0.96 |
| + ECA-NFNet ensemble member | 0.94–0.97 |

Top public-leader notebooks were ~0.91 when last surveyed (2026-04-23 in [STRATEGY.md](STRATEGY.md) §1). Reaching 0.94+ would put you in the gold-medal zone if private LB tracks public.

## 7. What NOT to do

- **Don't keep V-anchor sweeping.** Each sweep at this scale is inside the noise band. Sweep only specific blend weights between SSM and SED, not config hyperparameters.
- **Don't try to "fix SED to match SSM."** Different model families with different inductive biases. The point of having both is *diversity*, not redundancy.
- **Don't spend time on FP16 ONNX.** FP32 fits the budget. Revisit only if Kaggle ever rejects bold for runtime.
- **Don't submit a bold/SED-only blend without SSM.** SSM is your floor.

## 8. What's already in the codebase

Everything you need is wired:

- **Load-bearing SSM submission**: [LB_093.ipynb](LB_093.ipynb) — frozen, scored 0.93 LB on the prior leaderboard cut. Driver code lives in [birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py](birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py).
- **Cleaned SSM port (OOF-only)**: [birdclef/train/train_ssm_head.py](birdclef/train/train_ssm_head.py) — V-anchor 0.83, no LB number. Different config from the legacy notebook (prior=0, correction=0, threshold=[0.5], pca=128). Useful for sweeping; not yet a submission candidate.
- **SED training**: [birdclef/train/train_sed_ddp.py](birdclef/train/train_sed_ddp.py).
- **Pseudo-label generator**: [birdclef/train/pseudo_label.py](birdclef/train/pseudo_label.py) (both SSM-teacher and SED-teacher paths).
- **Local Kaggle dry-run**: [birdclef/scripts/_09b_test_submission_local.py](birdclef/scripts/_09b_test_submission_local.py) — supports `--sed-folds-glob` for bold ensembles.
- **Submission builder**: [birdclef/scripts/_09_build_submission.py](birdclef/scripts/_09_build_submission.py).
- **Diagnostic toolkit** (kept for future regressions): `_09c_probe_onnx.py`, `_09d_onnx_first_nan.py`, `_05b_refilter_pseudo.py`.

The remaining gap is **SSM-in-the-Kaggle-notebook (combined with SED)**: the legacy SSM notebook works as a standalone submission, but the new `birdclef/submit/inference_template.py` doesn't call any SSM. Step C above proposes two options for closing that gap (full re-implementation vs. CSV-blend of two notebooks).

## 9. Tonight (2026-04-25) action items

1. **Re-submit [LB_093.ipynb](LB_093.ipynb) as-is** — confirm baseline holds. Do not edit. Do not refactor. Set `MODE = "submit"` (default), attach the BirdCLEF 2026 competition data + the Perch v2 `bird-vocalization-classifier` Kaggle dataset, run end-to-end, submit `submission.csv`.
2. **Run bold SED V-anchor locally**:
   ```bash
   python -m birdclef.scripts._09b_test_submission_local \
       --sed-folds-glob 'birdclef/models_ckpt/sed/sed_v2s/fold*/best.onnx' \
       --source anchor
   ```
3. **If bold V-anchor ≥ 0.88**: ping me and I'll write the blend code (step C — likely option 2 first: blend two `submission.csv`s post-hoc, since it's the cheapest path to validate that SSM+SED together actually beats SSM alone on LB).
4. **If bold V-anchor < 0.85**: pseudo round 2 first, blend later.

Send me both numbers (re-confirmed SSM LB + bold SED V-anchor). That's all we need to lock in tomorrow's plan.

## 10. Reference: what's in [LB_093.ipynb](LB_093.ipynb)

This is the load-bearing artifact. Knobs as inspected on 2026-04-25:

| Stage | Setting |
|---|---|
| Audio | 32 kHz mono, 60 s file → 12 × 5-s windows |
| Perch | Google `perch_v2_cpu/1` SavedModel, optionally upgraded to ONNX. 1536-d emb + 234-d logits per window. |
| Genus proxy | For unmapped species, max-aggregate Perch logits over same-genus matches (Amphibia, Insecta, Aves) |
| Site/hour prior | `lambda_prior=0.4`. Built from labeled `train_soundscapes` species frequencies, shrunk toward global mean with `n / (n+8)` weight. Added as logit (`log p − log(1−p)`) to raw Perch scores. |
| MLP probes | `pca_dim=64`, `hidden=(128,64)`, `max_iter=300`, `min_pos=5`, `alpha_blend=0.4`. Per-class oversampling capped at 8×. Features = PCA emb + raw score + prev/next/mean/max/std across 12 windows. |
| LightProtoSSM | Bidirectional selective SSM (`d_state=16`) with cross-attention + class prototypes initialised from positive-sample mean embeddings. 40 epochs, `lr=1e-3`, patience 8, focal+BCE+distill, mixup α=0.4, SWA. |
| TTA | Circular shifts `[0, 1, -1, 2, -2]` of the 12-window sequence. |
| First-pass blend | `ENSEMBLE_W = 0.5` × ProtoSSM + 0.5 × MLP-adjusted Perch. |
| ResidualSSM | 30 ep, `lr=1e-3`, patience 8, MSE on `(Y − sigmoid(first_pass))`. Output head zero-init. Applied with `correction_weight=0.30`. |
| Per-class temperature | Sharper for Amphibia/Insecta (T=0.95), softer for Aves (T=1.10). |
| Sigmoid → post-proc | `file_confidence_scale(top_k=2, power=0.4)` → `rank_aware_scaling(power=0.4)` → `adaptive_delta_smooth(base_alpha=0.20)` → clip → `apply_per_class_thresholds`. |
| Per-class thresholds | Isotonic regression per class on first-pass OOF probs, then F1-grid search over `[0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]`. Probabilities linearly sharpened around the per-class threshold. |

The `submit` branch of CFG (cell 3) sets the n_epochs/patience/oof_n_splits values used in the pipeline call. The pipeline call in cell 25 also overrides some of those (e.g. ProtoSSM `lr=1e-3` instead of CFG's `8e-4`). When porting, preserve the **call-site** values, not the CFG defaults.
