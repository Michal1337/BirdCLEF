"""Patch LB_0931_seed.ipynb cell 26 — replace the PyTorch+multiprocessing
AST inference with single-process ONNX runtime inference.

ONNX Runtime on CPU is typically 3-5x faster than PyTorch eager mode for
transformers AND uses far less RAM than 4 worker processes each loading
their own model copy. With proper thread tuning (intra_op = available
cores), single-process ONNX should beat the 4-worker PyTorch setup.
"""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

NEW_AST_CELL = '''# ── Cell 9c: AST inference via ONNX Runtime (CPU-optimized) ──────────
# Loads a pre-exported ONNX of the fine-tuned AST model. ONNX Runtime
# on CPU is 3-5x faster than PyTorch eager for transformers (graph fusion,
# vectorized matmul, MKL). Single-process inference, ORT-internal threading
# tuned to all available cores.
#
# Build the ONNX file once on Hopper:
#     python -m birdclef.scripts._freeze_ast_to_onnx \\
#         --ast-ckpt birdclef_example/outputs/ast/ast_lr3e-05_e15/best_model.pt \\
#         --out-dir  birdclef_example/outputs/ast/ast_lr3e-05_e15_onnx \\
#         --validate
# Upload the resulting directory as a Kaggle dataset (e.g. birdclef-ast-onnx).

import torch.nn as _nn
import torchaudio as _ta
try:
    import onnxruntime as _ort
except ImportError:
    raise SystemExit("onnxruntime not installed — should be in Kaggle's base "
                     "image; if not, `pip install onnxruntime`.")

# === EDIT THIS PATH for your Kaggle dataset mount ===
AST_ONNX_DIR = Path("/kaggle/input/birdclef-ast-onnx/ast_lr3e-05_e15_onnx")
if not AST_ONNX_DIR.exists():
    AST_ONNX_DIR = Path("/kaggle/input/birdclef-ast-onnx")
AST_ONNX_PATH = AST_ONNX_DIR / "model.onnx"
if not AST_ONNX_PATH.exists():
    raise SystemExit(
        f"AST ONNX file missing: {AST_ONNX_PATH}\\n"
        f"Run _freeze_ast_to_onnx on Hopper, then upload the dir as "
        f"a Kaggle dataset and edit AST_ONNX_DIR above."
    )

_FBANK_MEAN = -4.2677
_FBANK_STD  = 4.5689

# Wrapper-level params (max_length, sample rates) saved next to model.onnx.
_wrapper_cfg_path = AST_ONNX_DIR / "wrapper_config.json"
if _wrapper_cfg_path.exists():
    with open(_wrapper_cfg_path, encoding="utf-8") as _f:
        _wrapper_cfg = json.load(_f)
else:
    _wrapper_cfg = {"input_sample_rate": 32000, "target_sample_rate": 16000,
                    "max_length": 512, "num_mel_bins": 128, "n_classes": N_CLASSES}
print(f"[ast] wrapper_cfg = {_wrapper_cfg}")

# Build the ORT session. intra_op = all CPU cores (threads inside a single
# matmul); inter_op = 1 (graph-level parallelism — limited gain for
# transformer forward, costs RAM if >1).
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


# Inference: each test soundscape (60s @ 32kHz) → 12 windows of 5s →
# resample → fbank → ONNX forward → sigmoid.
import soundfile as _sf
_t0_ast = time.time()
n_test_files = len(test_paths)
ast_test_probs = np.zeros((n_test_files * N_WINDOWS, N_CLASSES), dtype=np.float32)

for fi, fpath in enumerate(test_paths):
    y, _sr = _sf.read(str(fpath), dtype="float32", always_2d=False)
    if y.ndim == 2:
        y = y.mean(axis=1)
    if y.shape[0] < FILE_SAMPLES:
        y = np.pad(y, (0, FILE_SAMPLES - y.shape[0]))
    else:
        y = y[:FILE_SAMPLES]
    wins = torch.from_numpy(y.reshape(N_WINDOWS, WINDOW_SAMPLES).astype(np.float32))
    wav = _resample(wins)
    fb = torch.stack([_ast_fbank_one(wav[i:i+1]) for i in range(wav.size(0))], dim=0)
    fb_np = fb.numpy().astype(np.float32)                             # (12, max_length, n_mels)
    logits = _ast_sess.run([_ast_output_name], {_ast_input_name: fb_np})[0]
    # sigmoid in numpy (skips a torch round-trip)
    ast_test_probs[fi * N_WINDOWS:(fi + 1) * N_WINDOWS] = (
        1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))
    ).astype(np.float32)
    if fi % 50 == 0:
        elapsed = time.time() - _t0_ast
        print(f"[ast]   {fi+1}/{n_test_files}  elapsed={elapsed:.1f}s  "
              f"eta={elapsed/(fi+1)*(n_test_files-fi-1):.1f}s")

print(f"[ast] inference done  shape={ast_test_probs.shape}  "
      f"mean={ast_test_probs.mean():.4f}  total={time.time()-_t0_ast:.1f}s")
'''

nb["cells"][26]["source"] = NEW_AST_CELL.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 26 with ONNX Runtime AST inference")
print(f"  cell 26 size: {len(NEW_AST_CELL)}b")
