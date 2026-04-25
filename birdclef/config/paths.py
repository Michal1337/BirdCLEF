"""Single source of truth for all filesystem paths.

Env var BIRDCLEF_PATH_MODE switches between local training and Kaggle inference:
    BIRDCLEF_PATH_MODE=local   (default)
    BIRDCLEF_PATH_MODE=kaggle  (swaps roots to /kaggle/input/...)

Kaggle-mode layout assumes the user uploads their `birdclef/models_ckpt/`
and `birdclef/cache/perch/` as a Kaggle dataset named "birdclef26-artifacts".
Override via env vars BIRDCLEF_KAGGLE_DATA / BIRDCLEF_KAGGLE_ARTIFACTS.
"""
from __future__ import annotations

import os
from pathlib import Path


_MODE = os.environ.get("BIRDCLEF_PATH_MODE", "local").lower()


def _env_path(name: str, default: Path) -> Path:
    raw = os.environ.get(name)
    return Path(raw) if raw else default


_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2]
BCLEF_ROOT = REPO / "birdclef"


if _MODE == "kaggle":
    DATA_ROOT = _env_path("BIRDCLEF_KAGGLE_DATA", Path("/kaggle/input/birdclef-2026"))
    ARTIFACT_ROOT = _env_path(
        "BIRDCLEF_KAGGLE_ARTIFACTS", Path("/kaggle/input/birdclef26-artifacts")
    )
    CACHE_ROOT = ARTIFACT_ROOT / "cache"
    MODEL_ROOT = ARTIFACT_ROOT / "models_ckpt"
    OUTPUT_ROOT = _env_path("BIRDCLEF_KAGGLE_OUTPUT", Path("/kaggle/working"))
    SPLIT_ROOT = ARTIFACT_ROOT / "splits"
else:
    DATA_ROOT = _env_path("BIRDCLEF_DATA_ROOT", REPO / "data")
    CACHE_ROOT = _env_path("BIRDCLEF_CACHE_ROOT", BCLEF_ROOT / "cache")
    MODEL_ROOT = _env_path("BIRDCLEF_MODEL_ROOT", BCLEF_ROOT / "models_ckpt")
    OUTPUT_ROOT = _env_path("BIRDCLEF_OUTPUT_ROOT", BCLEF_ROOT / "outputs")
    SPLIT_ROOT = _env_path("BIRDCLEF_SPLIT_ROOT", BCLEF_ROOT / "splits")


TRAIN_AUDIO = Path("/mnt/evafs/groups/re-com/mgromadzki/data/train_audio")
SOUNDSCAPES = DATA_ROOT / "train_soundscapes"
TEST_SC = DATA_ROOT / "test_soundscapes"
TRAIN_CSV = Path("/mnt/evafs/groups/re-com/mgromadzki/data/train.csv")
SCLABEL_CSV = DATA_ROOT / "train_soundscapes_labels.csv"
TAXONOMY = DATA_ROOT / "taxonomy.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
RECORDING_LOCATION = DATA_ROOT / "recording_location.txt"

PERCH_DIR = CACHE_ROOT / "perch"
PERCH_META = PERCH_DIR / "meta.parquet"
PERCH_NPZ = PERCH_DIR / "arrays.npz"
PERCH_LABELS = PERCH_DIR / "labels.npy"
PERCH_PROXY_MAP = PERCH_DIR / "proxy_map.json"

WAVEFORM_DIR = CACHE_ROOT / "waveforms"
WAVEFORM_NPY = WAVEFORM_DIR / "train_audio_f16_32k.npy"
WAVEFORM_INDEX = WAVEFORM_DIR / "train_audio_index.parquet"

# Soundscape cache: every soundscape OGG decoded once to a single fixed-shape
# memmap (n_files, FILE_SAMPLES) float16 32 kHz mono. SEDTrainDataset reads
# from this when present, falls back to on-the-fly OGG decode otherwise.
SOUNDSCAPE_CACHE_DIR = CACHE_ROOT / "soundscapes"
SOUNDSCAPE_NPY = SOUNDSCAPE_CACHE_DIR / "soundscapes_f16_32k.npy"
SOUNDSCAPE_INDEX = SOUNDSCAPE_CACHE_DIR / "soundscapes_index.parquet"

PSEUDO_DIR = CACHE_ROOT / "pseudo"

def folds_path(n_splits: int) -> Path:
    """Path to the static StratifiedKFold parquet for `n_splits` folds.

    Materialized by `python -m birdclef.scripts._02_build_splits`.
    Both 5-fold and 10-fold variants live alongside each other; downstream
    scripts pick via `--n-splits`.
    """
    return SPLIT_ROOT / f"folds_{int(n_splits)}_strat.parquet"

SUBMIT_DIR = OUTPUT_ROOT / "submit"

# Existing shared artifacts we still rely on (Perch ONNX lives under the
# original repo models/ tree to avoid a 2 GB re-copy).
EXISTING_MODELS_ROOT = _env_path("BIRDCLEF_EXISTING_MODELS", REPO / "models")
ONNX_PERCH_PATH = EXISTING_MODELS_ROOT / "perch_onnx" / "perch_v2.onnx"
PERCH_TF_SAVEDMODEL = EXISTING_MODELS_ROOT / "perch_v2_cpu" / "1"
PERCH_TF_LABELS = PERCH_TF_SAVEDMODEL / "assets" / "labels.csv"


# Audio constants
SR = 32_000
WINDOW_SEC = 5
WINDOW_SAMPLES = SR * WINDOW_SEC
FILE_SAMPLES = 60 * SR
N_WINDOWS = 12


def ensure_dirs() -> None:
    """Create writable dirs (no-op on Kaggle where /kaggle/input is RO)."""
    if _MODE == "kaggle":
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        return
    for p in [CACHE_ROOT, PERCH_DIR, WAVEFORM_DIR, SOUNDSCAPE_CACHE_DIR, PSEUDO_DIR, MODEL_ROOT,
              OUTPUT_ROOT, SPLIT_ROOT, SUBMIT_DIR]:
        p.mkdir(parents=True, exist_ok=True)


def mode() -> str:
    return _MODE
