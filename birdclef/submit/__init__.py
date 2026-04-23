"""Submit-side modules. Must remain import-clean of training deps.

Any accidental import from birdclef.train or of torchaudio/timm should
fail the import-guard test in birdclef/submit/_import_guard.py.
"""
from . import _import_guard  # noqa: F401
