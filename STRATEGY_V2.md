# BirdCLEF 2026 — Strategy v2

> **Status snapshot (2026-04-25)**: SSM pipeline at **0.92–0.93 public LB** (full-data trained, real). Single-fold SED at **0.738 public LB** despite v_anchor 0.89. Pseudo-label round 1 complete; SED 5-fold ensemble untested on LB. Soundscape memmap built; SED training infrastructure stable.

This supersedes [STRATEGY.md](STRATEGY.md). The pivot from v1: **SSM is the workhorse, SED is a candidate ensemble member, not a replacement.**

## 1. What changed since v1

### What worked
- **Site×date GroupKFold + V-anchor split** ([birdclef/data/splits.py](birdclef/data/splits.py)): cleaned up the leaky filename-grouping; gave honest per-fold/per-site metrics.
- **SSM pipeline cleanup**: setting `lambda_prior=0`, `correction_weight=0`, `threshold_grid=[0.5]` lifted honest OOF from 0.71 → 0.83 with no LB regression.
- **Pseudo-labeling pipeline** ([birdclef/train/pseudo_label.py](birdclef/train/pseudo_label.py)): SSM 3-seed ensemble teacher → SED student via `cache/pseudo/round1/`. End-to-end works.
- **SED DDP trainer** ([birdclef/train/train_sed_ddp.py](birdclef/train/train_sed_ddp.py)): EfficientNetV2-S, FocalBCE, mixup, EMA, fold/V-anchor periodic eval, bf16 on H100.
- **Soundscape memmap cache** ([birdclef/cache/build_soundscape_cache.py](birdclef/cache/build_soundscape_cache.py)): ~5× faster epochs. Necessary for round-2 iteration.

### What didn't
- **SED single-fold submission underperforms SSM by ~0.19 on public LB** (0.738 vs ~0.93). V-anchor over-estimated true generalization by ~0.16.
- **FP16 ONNX export tripped multiple bugs**: complex STFT, version-converter Pad, custom op_block_list overwriting defaults, BN-buffer EMA-load. We resolved all of them but the conclusion is "ship FP32".
- **My initial table treated 0.93 as "leaky"** — wrong. The submitted SSM model was trained on all labeled data; the OOF leakage only affected eval, not the deployed weights. Mea culpa.

### Key lesson
**Trust public LB over V-anchor when they disagree by more than ~0.05.** V-anchor is 13 small-distribution files; public LB is 600 broader-distribution files. For configs, V-anchor is fine for ranking. For *absolute* generalization, only LB tells the truth.

## 2. Honest current numbers

| Pipeline | Train data | Honest V-anchor | Public LB |
|---|---|---:|---:|
| **SSM (`prior_off`, full-data)** | all 59 labeled | 0.83 (proxy) | **0.92–0.93** |
| SED fold 0 (round 1) | ~37 labeled + train_audio + pseudo | 0.89 | **0.738** |
| SED 5-fold ensemble (round 1) | per-fold | mean 0.865 | untested, project ~0.75–0.78 |
| SSM + SED 5-fold blend | as above | untested | **target 0.93–0.95** |
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

Re-submit the SSM pipeline to the *current* public test (the leaderboard might have been re-cut since your last SSM submission). Use the same notebook that hit 0.93, attach the latest competition data, run.

If LB confirms ≥ 0.92, **that's the floor**. Any future submission that scores below this is a regression.

### B. Test SED 5-fold ensemble locally (5 min)

Bold ensemble against V-anchor with the new convenience flag:

```bash
python -m birdclef.scripts._09b_test_submission_local \
    --sed-folds-glob 'birdclef/models_ckpt/sed/sed_v2s/fold*/best.onnx' \
    --source anchor
```

Expected: `macro_auc` in 0.88–0.91 range (mean fold v_anchor was 0.865, ensemble usually adds +0.01–0.02). If it lands here, the ensemble works as expected.

### C. Build SSM+SED ensemble inference template (~2 h)

The current Kaggle inference template only handles SED ONNX. Extend it to also load and run the SSM pipeline:

1. **Perch ONNX** loads (already wired in template, just unused) → gives 5-s embeddings + raw class logits.
2. **SSM head checkpoint** (small `.pt`, < 5 MB) loads in pure PyTorch CPU → maps embeddings to SSM logits.
3. **Per-class MLP probes** (sklearn `.pkl`) load → applied as a Linear in PyTorch.
4. Apply `lambda_prior=0`, post-processing, threshold pass-through.
5. Blend SSM probs with SED-ensemble probs at recipe weights.

Total Kaggle runtime estimate: ~25 min (Perch is slow; SSM head is trivial; SED ensemble dominates).

Files to write:
- `birdclef/submit/ssm_inference.py` — Perch + SSM head + MLP probes inference, pure PyTorch CPU.
- Update `birdclef/submit/inference_template.py` to call SSM, SED, then blend.
- Update `birdclef/submit/build_notebook.py` to bundle Perch ONNX + SSM head + SED ONNX paths.

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

- **SSM pipeline**: [birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py](birdclef_example/sota_oof_two_pass_ssm_advanced_pp.py) (legacy, scoring 0.93) AND [birdclef/train/train_ssm_head.py](birdclef/train/train_ssm_head.py) (clean port, V-anchor 0.83).
- **SED training**: [birdclef/train/train_sed_ddp.py](birdclef/train/train_sed_ddp.py).
- **Pseudo-label generator**: [birdclef/train/pseudo_label.py](birdclef/train/pseudo_label.py) (both SSM-teacher and SED-teacher paths).
- **Local Kaggle dry-run**: [birdclef/scripts/_09b_test_submission_local.py](birdclef/scripts/_09b_test_submission_local.py) — supports `--sed-folds-glob` for bold ensembles.
- **Submission builder**: [birdclef/scripts/_09_build_submission.py](birdclef/scripts/_09_build_submission.py).
- **Diagnostic toolkit** (kept for future regressions): `_09c_probe_onnx.py`, `_09d_onnx_first_nan.py`, `_05b_refilter_pseudo.py`.

The remaining gap is **SSM-in-the-Kaggle-notebook**: the legacy SSM submission notebook works as-is, but the new clean `birdclef/submit/inference_template.py` doesn't yet call the SSM head. Step C above closes that gap.

## 9. Tonight (2026-04-25) action items

1. **Re-submit SSM** (existing 0.93 notebook) — confirm baseline holds.
2. **Run bold SED V-anchor locally**:
   ```bash
   python -m birdclef.scripts._09b_test_submission_local \
       --sed-folds-glob 'birdclef/models_ckpt/sed/sed_v2s/fold*/best.onnx' \
       --source anchor
   ```
3. **If bold V-anchor ≥ 0.88**: ping me and I'll write the SSM-in-template code (step C).
4. **If bold V-anchor < 0.85**: pseudo round 2 first, blend later.

Send me both numbers (re-confirmed SSM LB + bold SED V-anchor). That's all we need to lock in tomorrow's plan.
