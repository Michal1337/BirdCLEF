"""Patch LB_0931_seed.ipynb cell 26 — wrap AST inference with a 4-worker
multiprocessing pool. Each worker loads its own copy of the AST model and
processes a chunk of test files in parallel.
"""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

NEW_AST_CELL = '''# ── Cell 9c: AST (audio transformer) inference (OFFLINE + 4-worker MP) ─
# Loads the pre-frozen AST model from a Kaggle dataset (no Hub fetch) and
# runs inference using multiprocessing — 4 workers each handling ~25% of
# the test files. Each worker pins torch to 1 thread to avoid CPU
# oversubscription (4 workers × 4 threads = 16-way contention on Kaggle's
# 4-core CPU is much slower than 4 × 1).
#
# Build offline dir on Hopper:
#     python -m birdclef.scripts._freeze_ast_for_kaggle \\
#         --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
#         --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_offline
# Upload as Kaggle dataset birdclef-ast-best.

# === EDIT THIS PATH for your Kaggle dataset mount ===
AST_OFFLINE_DIR = Path("/kaggle/input/birdclef-ast-best/ast_lr3e-05_e15_offline")
if not AST_OFFLINE_DIR.exists():
    AST_OFFLINE_DIR = Path("/kaggle/input/birdclef-ast-best")
if not (AST_OFFLINE_DIR / "config.json").exists():
    raise SystemExit(
        f"AST offline dir missing or incomplete: {AST_OFFLINE_DIR}\\n"
        f"Expected config.json + model.safetensors inside.\\n"
        f"Run _freeze_ast_for_kaggle on Hopper and upload the dir."
    )

_wrapper_cfg_path = AST_OFFLINE_DIR / "wrapper_config.json"
if _wrapper_cfg_path.exists():
    with open(_wrapper_cfg_path, encoding="utf-8") as _f:
        _wrapper_cfg = json.load(_f)
else:
    _wrapper_cfg = {"input_sample_rate": 32000, "target_sample_rate": 16000,
                    "max_length": 512, "num_mel_bins": 128, "n_classes": N_CLASSES}
print(f"[ast] wrapper_cfg = {_wrapper_cfg}")


# Worker function — must be at module-level for the multiprocessing pool to
# pickle it. Each worker loads its own AST model copy from the local dir,
# processes a chunk of file indices, returns (file_indices, probs_array).
def _ast_worker(args):
    """Single-process AST inference for a chunk of test files."""
    file_indices, file_paths_str, offline_dir_str, cfg_dict, n_classes = args

    # Pin torch to 1 thread inside the worker. 4 workers × 4 torch threads
    # would contend; 4 workers × 1 thread cleanly maps to 4 cores.
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"

    import numpy as _np
    import torch as _torch
    import torch.nn as _nn
    import torchaudio as _ta
    import soundfile as _sf
    from transformers import ASTForAudioClassification as _ASTModel

    _torch.set_num_threads(1)

    target_sr   = int(cfg_dict["target_sample_rate"])
    input_sr    = int(cfg_dict["input_sample_rate"])
    max_length  = int(cfg_dict["max_length"])
    num_mel_bins = int(cfg_dict["num_mel_bins"])
    FBANK_MEAN, FBANK_STD = -4.2677, 4.5689

    FILE_SAMPLES   = 60 * input_sr
    N_WINDOWS_LOC  = 12
    WIN_SAMPLES    = FILE_SAMPLES // N_WINDOWS_LOC

    model = _ASTModel.from_pretrained(offline_dir_str, local_files_only=True).eval()
    resample = (_ta.transforms.Resample(input_sr, target_sr)
                if input_sr != target_sr else _nn.Identity())

    def _fbank_one(wav_1xT):
        fb = _ta.compliance.kaldi.fbank(
            wav_1xT, htk_compat=True, sample_frequency=target_sr,
            use_energy=False, window_type="hanning",
            num_mel_bins=num_mel_bins, dither=0.0,
            frame_shift=10, frame_length=25,
        )
        fb = (fb - FBANK_MEAN) / (FBANK_STD * 2.0)
        n = fb.size(0)
        if n < max_length:
            fb = _torch.nn.functional.pad(fb, (0, 0, 0, max_length - n))
        elif n > max_length:
            fb = fb[:max_length, :]
        return fb

    out = _np.zeros((len(file_paths_str) * N_WINDOWS_LOC, n_classes), dtype=_np.float32)
    for li, fpath in enumerate(file_paths_str):
        y, _sr = _sf.read(str(fpath), dtype="float32", always_2d=False)
        if y.ndim == 2:
            y = y.mean(axis=1)
        if y.shape[0] < FILE_SAMPLES:
            y = _np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
        else:
            y = y[:FILE_SAMPLES]
        wins_np = y.reshape(N_WINDOWS_LOC, WIN_SAMPLES).astype(_np.float32)
        wins = _torch.from_numpy(wins_np)
        wav = resample(wins)
        fb = _torch.stack([_fbank_one(wav[i:i+1]) for i in range(wav.size(0))], dim=0)
        with _torch.no_grad():
            logits = model(input_values=fb).logits
        probs = _torch.sigmoid(logits.float()).numpy().astype(_np.float32)
        out[li * N_WINDOWS_LOC:(li + 1) * N_WINDOWS_LOC] = probs
    return file_indices, out


# Submit to a process pool — 4 workers, file chunks balanced.
import concurrent.futures as _cf

N_AST_WORKERS = 4
n_test_files = len(test_paths)

# Split test files into N_AST_WORKERS contiguous chunks.
_chunk_size = (n_test_files + N_AST_WORKERS - 1) // N_AST_WORKERS
_chunks = []
for w in range(N_AST_WORKERS):
    start = w * _chunk_size
    end = min(start + _chunk_size, n_test_files)
    if start >= n_test_files:
        break
    indices = list(range(start, end))
    paths_str = [str(test_paths[i]) for i in indices]
    _chunks.append((indices, paths_str, str(AST_OFFLINE_DIR), _wrapper_cfg, N_CLASSES))

print(f"[ast] starting {len(_chunks)} workers × ~{_chunk_size} files each "
      f"on {n_test_files} total test files")
ast_test_probs = np.zeros((n_test_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
_t0_ast = time.time()
with _cf.ProcessPoolExecutor(max_workers=N_AST_WORKERS) as _ex:
    _futures = {_ex.submit(_ast_worker, ca): wi for wi, ca in enumerate(_chunks)}
    for _fut in _cf.as_completed(_futures):
        _wi = _futures[_fut]
        try:
            file_indices, chunk_probs = _fut.result()
        except Exception as _e:
            print(f"[ast] worker {_wi} FAILED: {_e}")
            raise
        for _li, _fi in enumerate(file_indices):
            _start_row = _fi * N_WINDOWS
            ast_test_probs[_start_row:_start_row + N_WINDOWS] = (
                chunk_probs[_li * N_WINDOWS:(_li + 1) * N_WINDOWS]
            )
        print(f"[ast]   worker {_wi} done — {len(file_indices)} files  "
              f"elapsed={time.time()-_t0_ast:.1f}s")

print(f"[ast] inference done  shape={ast_test_probs.shape}  "
      f"mean={ast_test_probs.mean():.4f}  total={time.time()-_t0_ast:.1f}s")
'''

nb["cells"][26]["source"] = NEW_AST_CELL.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 26 with 4-worker multiprocessing")
print(f"  cell 26 size: {len(NEW_AST_CELL)}b")
