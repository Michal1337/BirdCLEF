"""Reproducibility utilities.

`seed_everything(seed)` seeds every random source we touch (python `random`,
numpy, torch CPU + all CUDA devices) so two runs with the same seed produce
identical results, modulo a few caveats:

- cuDNN convolution algorithms can still be non-deterministic; we do NOT
  enable `torch.use_deterministic_algorithms(True)` because it (a) requires
  `CUBLAS_WORKSPACE_CONFIG=:4096:8` set at process start and (b) slows conv
  ops. Repeatability on CPU is exact; on GPU it is statistical-stable.
- DataLoader workers have their own RNGs; for single-process CPU training
  in train_ssm_head this doesn't matter. The DDP trainer already seeds
  `torch.manual_seed(42 + rank)` in `train_one_fold`.
- `PYTHONHASHSEED` cannot be reliably set from within Python for an already-
  running process. Set it in the shell (e.g. `export PYTHONHASHSEED=0`) if
  dict-ordering determinism matters.

Import cost is near-zero; safe to call inside hot loops.
"""
from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int) -> int:
    """Seed python/numpy/torch RNGs. Returns the seed for logging."""
    seed = int(seed)
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
    return seed
