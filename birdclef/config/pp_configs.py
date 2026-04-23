"""Post-processing-only sweep grids (operate on an existing probability tensor)."""
from __future__ import annotations

SWEEP_PP = [
    {"name": "baseline", "smoothing": "adaptive", "use_boost": False},
    {"name": "gaussian_only", "smoothing": "gaussian", "use_boost": False},
    {"name": "boost_only", "smoothing": "none", "use_boost": True, "boost_lift": 0.25},
    {"name": "gaussian_boost", "smoothing": "gaussian", "use_boost": True,
     "boost_lift": 0.25, "boost_threshold": 0.5},
    {"name": "gaussian_boost_soft", "smoothing": "gaussian", "use_boost": True,
     "boost_lift": 0.15, "boost_threshold": 0.6},
]
