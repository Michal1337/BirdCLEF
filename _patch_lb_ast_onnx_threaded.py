"""Patch LB_0931_seed.ipynb cell 26 — ONNX AST inference with ThreadPoolExecutor
prefetch for I/O + multi-file batched forwards.

Same shape as the Perch ONNX inference cell (Cell 5): a 4-thread I/O pool
prefetches the next batch's audio while the ONNX runtime processes the
current batch. Batched forward (4 files = 48 windows per session.run)
amortizes Python overhead and gives bigger matmuls for better SIMD util.
"""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

NEW_AST_CELL = '''# ── Cell 9c: AST inference via ONNX Runtime + threaded I/O prefetch ───
# Loads pre-exported ONNX of fine-tuned AST + runs CPU-optimized inference
# with overlapping disk I/O. Same architecture as the Perch ONNX cell:
#   - 4 I/O threads prefetch the next batch's audio while ORT processes
#     the current batch
#   - Multi-file batched forward (4 files × 12 windows = 48 per session.run)
#     gives bigger matmuls → better MKL/SIMD utilization than 1 file at a time
#
# Build ONNX once on Hopper:
#     python -m birdclef.scripts._freeze_ast_to_onnx \\
#         --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
#         --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx \\
#         --validate
# Upload as Kaggle dataset birdclef-ast-onnx.

import torch.nn as _nn
import torchaudio as _ta
import concurrent.futures as _cf
try:
    import onnxruntime as _ort
except ImportError:
    raise SystemExit("onnxruntime not installed — should be in Kaggle base image.")

# === EDIT THIS PATH for your Kaggle dataset mount ===
AST_ONNX_DIR = Path("/kaggle/input/birdclef-ast-onnx/ast_lr3e-05_e15_onnx")
if not AST_ONNX_DIR.exists():
    AST_ONNX_DIR = Path("/kaggle/input/birdclef-ast-onnx")
AST_ONNX_PATH = AST_ONNX_DIR / "model.onnx"
if not AST_ONNX_PATH.exists():
    raise SystemExit(
        f"AST ONNX file missing: {AST_ONNX_PATH}\\n"
        f"Run _freeze_ast_to_onnx on Hopper and upload the dir."
    )

_FBANK_MEAN = -4.2677
_FBANK_STD  = 4.5689

_wrapper_cfg_path = AST_ONNX_DIR / "wrapper_config.json"
if _wrapper_cfg_path.exists():
    with open(_wrapper_cfg_path, encoding="utf-8") as _f:
        _wrapper_cfg = json.load(_f)
else:
    _wrapper_cfg = {"input_sample_rate": 32000, "target_sample_rate": 16000,
                    "max_length": 512, "num_mel_bins": 128, "n_classes": N_CLASSES}
print(f"[ast] wrapper_cfg = {_wrapper_cfg}")

# Build the ORT session. intra_op = all CPU cores for the matmuls; with
# the I/O thread pool above this, the *I/O threads* are mostly idle waiting
# on disk so they don't compete with intra_op work.
_n_cpus = int(os.environ.get("ORT_INTRA_OP_THREADS", str(os.cpu_count() or 4)))
_so = _ort.SessionOptions()
_so.intra_op_num_threads = _n_cpus
_so.inter_op_num_threads = 1
_so.execution_mode = _ort.ExecutionMode.ORT_SEQUENTIAL
_so.graph_optimization_level = _ort.GraphOptimizationLevel.ORT_ENABLE_ALL
print(f"[ast] loading {AST_ONNX_PATH} (CPU, intra_op_threads={_n_cpus})...")
_ast_sess = _ort.InferenceSession(
    str(AST_ONNX_PATH), sess_options=_so, providers=["CPUExecutionProvider"],
)
_ast_input_name  = _ast_sess.get_inputs()[0].name   # "input_values"
_ast_output_name = _ast_sess.get_outputs()[0].name  # "logits"
print(f"[ast] loaded — n_classes={_wrapper_cfg.get('n_classes', '?')}, "
      f"trained_epoch={_wrapper_cfg.get('trained_epoch', '?')}, "
      f"best_val_auc_focal_seen={_wrapper_cfg.get('best_val_auc_focal_seen', '?')}")

# Preprocessing helpers (kaldi-fbank stays in PyTorch — fast, simple).
_input_sr   = int(_wrapper_cfg["input_sample_rate"])
_target_sr  = int(_wrapper_cfg["target_sample_rate"])
_max_length = int(_wrapper_cfg["max_length"])
_n_mels     = int(_wrapper_cfg["num_mel_bins"])
_resample = (_ta.transforms.Resample(_input_sr, _target_sr)
             if _input_sr != _target_sr else _nn.Identity())


def _ast_read_60s(fpath):
    """Read one 60s soundscape, mono-mix, pad/truncate to FILE_SAMPLES.
    Returns numpy float32 (FILE_SAMPLES,). Releases GIL during disk read,
    so safe to run in a ThreadPoolExecutor."""
    import soundfile as _sf
    y, _sr = _sf.read(str(fpath), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    else:
        y = y[:FILE_SAMPLES]
    return y.astype(np.float32, copy=False)


def _ast_fbank_one(wav_1xT):
    """Kaldi-style fbank for one (1, T) waveform at the target sample rate."""
    fb = _ta.compliance.kaldi.fbank(
        wav_1xT, htk_compat=True, sample_frequency=_target_sr,
        use_energy=False, window_type="hanning",
        num_mel_bins=_n_mels, dither=0.0,
        frame_shift=10, frame_length=25,
    )
    fb = (fb - _FBANK_MEAN) / (_FBANK_STD * 2.0)
    n = fb.size(0)
    if n < _max_length:
        fb = torch.nn.functional.pad(fb, (0, 0, 0, _max_length - n))
    elif n > _max_length:
        fb = fb[:_max_length, :]
    return fb


# Inference: 4 files per ONNX forward (48 windows), I/O for the next 4
# files prefetched while the current 4 are being processed.
_AST_BATCH_FILES = 4
n_test_files = len(test_paths)
ast_test_probs = np.zeros((n_test_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
_t0_ast = time.time()
print(f"[ast] {_AST_BATCH_FILES} files/batch, prefetching with 4 I/O threads")

with _cf.ThreadPoolExecutor(max_workers=4) as _io:
    # Submit reads for the first batch.
    _next_start = 0
    _next_paths = test_paths[_next_start : _next_start + _AST_BATCH_FILES]
    _futures = [_io.submit(_ast_read_60s, p) for p in _next_paths]

    for _batch_start in range(0, n_test_files, _AST_BATCH_FILES):
        # Collect this batch's audio (was prefetched in previous iteration).
        _batch_paths = _next_paths
        _batch_audio = [f.result() for f in _futures]
        _batch_n = len(_batch_audio)

        # Kick off prefetch for the next batch BEFORE we do CPU work.
        _next_start = _batch_start + _AST_BATCH_FILES
        if _next_start < n_test_files:
            _next_paths = test_paths[_next_start : _next_start + _AST_BATCH_FILES]
            _futures = [_io.submit(_ast_read_60s, p) for p in _next_paths]

        # Build (batch * N_WINDOWS, max_length, n_mels) fbank tensor.
        # 4 files × 12 windows = 48 rows per ORT forward — much better
        # MKL utilization than 1 file at a time (12 rows).
        _all_wins_np = np.stack(
            [a.reshape(N_WINDOWS, WINDOW_SAMPLES) for a in _batch_audio], axis=0,
        ).reshape(_batch_n * N_WINDOWS, WINDOW_SAMPLES)
        _wins = torch.from_numpy(_all_wins_np.astype(np.float32))
        _wav = _resample(_wins)
        _fb = torch.stack(
            [_ast_fbank_one(_wav[i:i+1]) for i in range(_wav.size(0))], dim=0,
        )
        _fb_np = _fb.numpy().astype(np.float32)                         # (batch*12, max_length, n_mels)
        _logits = _ast_sess.run([_ast_output_name], {_ast_input_name: _fb_np})[0]
        _probs = (1.0 / (1.0 + np.exp(-np.clip(_logits, -30, 30)))).astype(np.float32)
        ast_test_probs[_batch_start * N_WINDOWS : (_batch_start + _batch_n) * N_WINDOWS] = _probs

        if _batch_start % (_AST_BATCH_FILES * 10) == 0 or _batch_start + _batch_n >= n_test_files:
            elapsed = time.time() - _t0_ast
            done = _batch_start + _batch_n
            eta = elapsed / max(1, done) * max(0, n_test_files - done)
            print(f"[ast]   {done}/{n_test_files}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

print(f"[ast] inference done  shape={ast_test_probs.shape}  "
      f"mean={ast_test_probs.mean():.4f}  total={time.time()-_t0_ast:.1f}s")
'''

nb["cells"][26]["source"] = NEW_AST_CELL.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 26 with threaded I/O prefetch + batched ONNX forwards")
print(f"  cell 26 size: {len(NEW_AST_CELL)}b")
