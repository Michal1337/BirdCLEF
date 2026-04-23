# BirdCLEF 2026 — `birdclef/` pipeline

Clean rewrite of the strategy in [STRATEGY.md](../STRATEGY.md). No imports from
the old `birdclef_example/`. Two worlds kept strictly apart:

- **Training world** (`birdclef/train/`, heavy, uses torch/timm/torchaudio).
- **Submission world** (`birdclef/submit/`, CPU-only, ONNX only) — enforced by
  `birdclef/submit/_import_guard.py`.

## File map — what lives where

| Path | Purpose |
|---|---|
| [config/paths.py](config/paths.py) | Single source of truth for all paths. `BIRDCLEF_PATH_MODE=kaggle` swaps to `/kaggle/input/...` |
| [config/ssm_configs.py](config/ssm_configs.py) | Sweep grids for Perch+SSM pipeline. |
| [config/sed_configs.py](config/sed_configs.py) | Sweep grids for SED DDP trainer. |
| [config/pp_configs.py](config/pp_configs.py) | Post-processing-only sweep grids. |
| [data/soundscapes.py](data/soundscapes.py) | Parse `train_soundscapes_labels.csv` → per-window metadata + filename regex (site, date, hour). |
| [data/train_audio.py](data/train_audio.py) | Parse `train.csv`, multi-hot label matrix, rare/frequent class support. |
| [data/splits.py](data/splits.py) | Build site×date GroupKFold + V-anchor hold-out; persist to `splits/`. |
| [data/datasets.py](data/datasets.py) | `SEDTrainDataset` (memmap reader), `InferenceDataset` (OGG streamer). |
| [data/augment.py](data/augment.py) | `SpecAugment`, `WaveformAug`, `mixup(max)`, `background_mix`. |
| [models/losses.py](models/losses.py) | `FocalBCE`, `BCEFocalMean`, `BCEPosWeight`. Secondary-label masking. |
| [models/ssm.py](models/ssm.py) | `LightProtoSSM` + `ResidualSSM` (clean port, no CFG globals). |
| [models/perch.py](models/perch.py) | Perch v2 ONNX wrapper + genus proxy mapping. |
| [models/sed.py](models/sed.py) | SED: timm backbone + attention pool + log-mel frontend. |
| [cache/build_perch_cache.py](cache/build_perch_cache.py) | Regenerate `cache/perch/{meta.parquet,arrays.npz,labels.npy,proxy_map.json}`. |
| [cache/build_waveform_cache.py](cache/build_waveform_cache.py) | Decode `train_audio` → single memmap `.npy` (float16 32 kHz mono) + index parquet. |
| [eval/metrics.py](eval/metrics.py) | Macro-AUC + per-site / per-hour / rare / frequent subgroup AUC. |
| [eval/oof.py](eval/oof.py) | Generic fold-safe OOF runner. |
| [eval/v_anchor.py](eval/v_anchor.py) | V-anchor scoring (primary selection metric). |
| [postproc/smoothing.py](postproc/smoothing.py) | Gaussian `[0.1,0.2,0.4,0.2,0.1]` + adaptive-delta smoothing. |
| [postproc/boost.py](postproc/boost.py) | Hard soundscape-wide probability boost (2024 #1 trick). |
| [postproc/tta.py](postproc/tta.py) | Sub-window waveform shift TTA (±1.25 s, ±2.5 s) + legacy window-roll. |
| [postproc/calibration.py](postproc/calibration.py) | Isotonic threshold calibration, prior shift, prior tables. |
| [sweep/runner.py](sweep/runner.py) | Generic sweep runner (resume, atomic CSV writes). |
| [sweep/writer.py](sweep/writer.py) | Lean summary CSV + per-config JSON + hparams-diff CSV. |
| [sweep/schema.py](sweep/schema.py) | Declares lean-CSV columns and rounding rules. |
| [train/train_ssm_head.py](train/train_ssm_head.py) | Perch+SSM head trainer — fold-safe OOF + V-anchor eval. |
| [train/train_sed_ddp.py](train/train_sed_ddp.py) | DDP SED trainer with EMA + AMP + periodic V-anchor eval. |
| [train/pseudo_label.py](train/pseudo_label.py) | Teacher-SED ensemble → pseudo-labels for all soundscapes. |
| [ensemble/blend.py](ensemble/blend.py) | Weighted/rank blending, weight search on V-anchor, correlation check. |
| [submit/inference_template.py](submit/inference_template.py) | Kaggle CPU inference pipeline (inlined into the notebook). |
| [submit/build_notebook.py](submit/build_notebook.py) | Assemble `submission_bold.ipynb` / `submission_safe.ipynb`. |
| [submit/export_onnx.py](submit/export_onnx.py) | Export trained SED checkpoint to ONNX (FP16 optional). |
| [submit/_import_guard.py](submit/_import_guard.py) | Trips if training deps leak into submit/. |
| [scripts/_01_build_caches.py](scripts/_01_build_caches.py) .. [_09_build_submission.py](scripts/_09_build_submission.py) | Thin CLI wrappers in the order below. |

## Filesystem layout

```
data/                       # UNMODIFIED — Kaggle raw data
birdclef/
├── cache/
│   ├── perch/{meta.parquet, arrays.npz, labels.npy, proxy_map.json}
│   ├── waveforms/{train_audio_f16_32k.npy, train_audio_index.parquet}
│   └── pseudo/round{N}/{probs.npz, meta.parquet, info.json}
├── splits/{folds_site_date.parquet, v_anchor_files.txt}
├── models_ckpt/
│   └── sed/<name>/fold{k}/{best.pt, best.onnx, best.fp16.onnx, final_metrics.json}
└── outputs/
    ├── sweep/<name>_summary.csv          # lean, sorted by primary desc
    ├── sweep/<name>_hparams.csv          # varied hparams only
    ├── sweep/<name>/<config>.json        # full per-fold/per-subgroup breakdown
    └── submit/{submission_bold.ipynb, submission_safe.ipynb}
```

## Run order

**Prerequisites**: Python 3.10+, PyTorch 2.1+, `onnxruntime`, `timm`,
`torchaudio`, `soundfile`, `librosa` (optional), `scikit-learn`, `tqdm`,
`psutil`, `pandas`, `pyarrow`, `scipy`, `onnxconverter-common` (for FP16
export). All commands run from the repository root.

> On Windows, replace `torchrun` with `python -m torch.distributed.run`.

```powershell
# From repo root
$env:BIRDCLEF_PATH_MODE = "local"
```

### 0 · Sanity import

```bash
python -c "import birdclef; from birdclef.config.paths import DATA_ROOT; print(DATA_ROOT)"
```

### 1 · Build caches

```bash
# Dry run (5 files per stage) — seconds
python -m birdclef.scripts._01_build_caches --stage perch --dry-run-files 5
python -m birdclef.scripts._01_build_caches --stage waveform --dry-run-files 5

# Full build — HOURS on CPU
python -m birdclef.scripts._01_build_caches --stage perch
python -m birdclef.scripts._01_build_caches --stage waveform
```

### 2 · Build site×date folds + V-anchor

```bash
python -m birdclef.scripts._02_build_splits
```
Inspect the stratification summary it prints. Should show each site × hour-bucket
present in both non-anchor and V-anchor columns.

### 3 · Regression sanity — reproduce current 0.91 OOF with new splits

```bash
python -m birdclef.scripts._04_train_ssm_head --sweep-name ssm_sanity
```
Expect `outputs/sweep/ssm_sanity_summary.csv` to show `macro_auc ≈ 0.91`
(within ±0.002 of the old sweep4 baseline). **Stop and investigate if the gap
is larger.**

### 4 · Cheap-wins sweep

```bash
python -m birdclef.scripts._07_oof_sweep --sweep cheap_wins
```
Outputs:
- `outputs/sweep/cheap_wins_summary.csv` (sorted by primary desc)
- `outputs/sweep/cheap_wins_hparams.csv` (varied hparams only)
- `outputs/sweep/cheap_wins/<config>.json` (full metrics per config)

You should see `focal_bce_waveform_shift` and/or `gaussian_plus_boost` rank
above baseline.

### 5 · SED smoke test (small DDP, <30 min)

```bash
# Pick 2 GPUs
python -m torch.distributed.run --standalone --nproc_per_node=2 \
    -m birdclef.train.train_sed_ddp --config sed_v2s --fold 0 --dry-run-steps 200
```
Verify: loss goes down, rank 0 writes a checkpoint under
`birdclef/models_ckpt/sed/sed_v2s/fold0/best.pt`.

### 6 · SED full training (one fold per run, repeat for folds 0..4)

```bash
python -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS \
    -m birdclef.train.train_sed_ddp --config sed_v2s --fold 0

python -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS \
    -m birdclef.train.train_sed_ddp --config sed_v2s --fold 1
# ... folds 2, 3, 4
```

Tune via `--override key=value` (JSON-parsed). Examples:
```
--override epochs=30 batch_size=48 mixup_alpha=0.8
```

Export each fold to ONNX:
```bash
python -m birdclef.submit.export_onnx \
    --ckpt birdclef/models_ckpt/sed/sed_v2s/fold0/best.pt \
    --out  birdclef/models_ckpt/sed/sed_v2s/fold0/best.onnx
```

### 7 · Pseudo-labeling round 1

```bash
python -m birdclef.scripts._05_pseudo_label \
    --round 1 \
    --ckpts birdclef/models_ckpt/sed/sed_v2s/fold0/best.pt \
            birdclef/models_ckpt/sed/sed_v2s/fold1/best.pt \
    --tau 0.5 --topk-per-species 2
```
Writes `birdclef/cache/pseudo/round1/{probs.npz, meta.parquet, info.json}`.

### 8 · SED student with pseudo-label mix

```bash
python -m torch.distributed.run --standalone --nproc_per_node=$N_GPUS \
    -m birdclef.scripts._06_train_sed_student \
    --config sed_v2s --fold 0 --pseudo-round 1
```
Repeat for folds 1..4. Only go for round 2 if V-anchor macro-AUC improved
by ≥ 0.003.

### 9 · Ensemble weight search (on V-anchor predictions)

Produce member prob files for each candidate (SSM pipeline + each SED fold).
Then:
```bash
python -m birdclef.scripts._08_ensemble \
    --members  outputs/members/ssm_baseline.npz \
              outputs/members/sed_v2s_fold0.npz \
              outputs/members/sed_v2s_fold1.npz \
    --y-true   outputs/members/v_anchor_y.npy \
    --meta     outputs/members/v_anchor_meta.parquet \
    --blend    sigmoid \
    --out      outputs/sweep/ensemble_final/best.json
```

### 10 · Build submission notebooks

```bash
python -m birdclef.scripts._09_build_submission \
    --recipe   outputs/sweep/ensemble_final/best.json \
    --sed-onnx /kaggle/input/birdclef26-artifacts/models_ckpt/sed/sed_v2s/fold0/best.fp16.onnx \
               /kaggle/input/birdclef26-artifacts/models_ckpt/sed/sed_v2s/fold1/best.fp16.onnx \
    --perch-onnx /kaggle/input/birdclef26-artifacts/models/perch_onnx/perch_v2.onnx \
    --variant bold

python -m birdclef.scripts._09_build_submission \
    --recipe   outputs/sweep/ensemble_final/safe.json \
    --sed-onnx /kaggle/input/birdclef26-artifacts/models_ckpt/sed/sed_v2s/fold0/best.fp16.onnx \
    --perch-onnx /kaggle/input/birdclef26-artifacts/models/perch_onnx/perch_v2.onnx \
    --variant safe
```
Outputs go to `birdclef/outputs/submit/submission_{bold,safe}.ipynb`. Upload
them + artifacts as a Kaggle dataset and submit. Submit the **safe** variant
first; escalate to bold only after a clean run.

## Env-var overrides

| Var | Effect |
|---|---|
| `BIRDCLEF_PATH_MODE=kaggle` | Kaggle inference mode |
| `BIRDCLEF_DATA_ROOT=<path>` | Override raw data path |
| `BIRDCLEF_CACHE_ROOT=<path>` | Override cache path |
| `BIRDCLEF_MODEL_ROOT=<path>` | Override model-checkpoint path |
| `BIRDCLEF_OUTPUT_ROOT=<path>` | Override sweep/submit output path |

## Troubleshooting

- `FileNotFoundError: cache/perch/meta.parquet` → run step 1 first.
- `RuntimeError: V-anchor file is empty` → run step 2 first.
- `MemoryError` from waveform cache → not enough RAM. Rerun with
  `BIRDCLEF_CACHE_ROOT=<fast SSD>` and accept page-cache speed instead of
  true RAM.
- SED training OOMs on GPU → `--override batch_size=32 grad_accum=2` to keep
  effective batch size.
- Any "import torchaudio" error inside `birdclef/submit/` → bug; the import
  guard should have caught it. Run `python -m birdclef.submit._import_guard`.
