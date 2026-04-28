"""Patch LB_0931_seed.ipynb cell 26 — ONNX AST inference, Perch-cell style.

One ONNX forward = one file = 12 windows. ThreadPoolExecutor with 4 workers
prefetches the next files' audio while the current file is processed by ORT.
Mirrors the Perch inference cell (Cell 5)'s I/O-prefetch + per-file forward
pattern.
"""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

NEW_AST_CELL = '''# ── Cell 9c: AST inference via ONNX Runtime + I/O prefetch (Perch-style) ─
# One forward pass per file (12 windows). ThreadPoolExecutor with 4 workers
# prefetches the next files' audio while ORT processes the current file —
# same pattern as the Perch ONNX inference cell. Disk I/O is hidden behind
# the model forward; ORT uses all CPU cores for the matmuls.
#
# Build ONNX once on Hopper:
#     python -m birdclef.scripts._freeze_ast_to_onnx \\
#         --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
#         --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx \\
#         --validate

import torch.nn as _nn
import torchaudio as _ta
import concurrent.futures as _cf
from collections import deque as _deque
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

# Build the ORT session. intra_op = all CPU cores; the I/O thread pool
# above this is mostly idle waiting on disk so it doesn't compete.
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
_ast_input_name  = _ast_sess.get_inputs()[0].name
_ast_output_name = _ast_sess.get_outputs()[0].name
print(f"[ast] loaded — n_classes={_wrapper_cfg.get('n_classes', '?')}, "
      f"trained_epoch={_wrapper_cfg.get('trained_epoch', '?')}, "
      f"best_val_auc_focal_seen={_wrapper_cfg.get('best_val_auc_focal_seen', '?')}")

# Preprocessing helpers (kaldi fbank stays in PyTorch — fast).
_input_sr   = int(_wrapper_cfg["input_sample_rate"])
_target_sr  = int(_wrapper_cfg["target_sample_rate"])
_max_length = int(_wrapper_cfg["max_length"])
_n_mels     = int(_wrapper_cfg["num_mel_bins"])
_resample = (_ta.transforms.Resample(_input_sr, _target_sr)
             if _input_sr != _target_sr else _nn.Identity())


def _ast_read_60s(fpath):
    """Read 60s soundscape, mono-mix, pad/truncate to FILE_SAMPLES.
    Releases GIL during disk read → safe to run in ThreadPoolExecutor."""
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


# Inference: 1 file per ORT forward (12 windows). 4 I/O threads keep a
# sliding window of prefetched files ready so the disk read never blocks
# the model forward.
n_test_files = len(test_paths)
ast_test_probs = np.zeros((n_test_files * N_WINDOWS, N_CLASSES), dtype=np.float32)
_t0_ast = time.time()
_PREFETCH_DEPTH = 4
print(f"[ast] 1 file / forward, prefetch depth={_PREFETCH_DEPTH}, {n_test_files} files")

with _cf.ThreadPoolExecutor(max_workers=4) as _io:
    # Prime the pipeline: submit the first PREFETCH_DEPTH reads.
    _q = _deque()
    _next_idx = min(_PREFETCH_DEPTH, n_test_files)
    for _i in range(_next_idx):
        _q.append(_io.submit(_ast_read_60s, test_paths[_i]))

    for _fi in range(n_test_files):
        # Pull the audio for the current file (already prefetched).
        y = _q.popleft().result()

        # Submit a new prefetch to keep the queue full.
        if _next_idx < n_test_files:
            _q.append(_io.submit(_ast_read_60s, test_paths[_next_idx]))
            _next_idx += 1

        # Process this file: 12 windows → 1 ORT forward.
        wins = torch.from_numpy(
            y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32)
        )
        wav = _resample(wins)
        fb = torch.stack(
            [_ast_fbank_one(wav[i:i+1]) for i in range(wav.size(0))], dim=0,
        )
        fb_np = fb.numpy().astype(np.float32)                           # (12, max_length, n_mels)
        logits = _ast_sess.run([_ast_output_name], {_ast_input_name: fb_np})[0]
        ast_test_probs[_fi * N_WINDOWS:(_fi + 1) * N_WINDOWS] = (
            1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
        ).astype(np.float32)

        if _fi % 50 == 0 or _fi == n_test_files - 1:
            elapsed = time.time() - _t0_ast
            done = _fi + 1
            eta = elapsed / done * (n_test_files - done)
            print(f"[ast]   {done}/{n_test_files}  elapsed={elapsed:.1f}s  eta={eta:.1f}s")

print(f"[ast] inference done  shape={ast_test_probs.shape}  "
      f"mean={ast_test_probs.mean():.4f}  total={time.time()-_t0_ast:.1f}s")
'''

nb["cells"][26]["source"] = NEW_AST_CELL.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 26 — Perch-style 1-file-per-forward ONNX inference with I/O prefetch")
print(f"  cell 26 size: {len(NEW_AST_CELL)}b")
