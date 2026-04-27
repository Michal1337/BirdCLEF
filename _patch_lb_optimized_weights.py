"""Patch LB_0931_seed.ipynb cell 27 with the per-taxon weights from
`_09_blend_search.py` output."""
import json
from pathlib import Path

NB = Path("LB_0931_seed.ipynb")
nb = json.loads(NB.read_text(encoding="utf-8"))

# Hand-picked weights (what's currently in the cell)
old_block = (
    "_TAXON_WEIGHTS = {\n"
    "    \"Aves\":     (0.60, 0.40),\n"
    "    \"Amphibia\": (0.55, 0.45),\n"
    "    \"Insecta\":  (0.30, 0.70),\n"
    "    \"Mammalia\": (0.30, 0.70),\n"
    "    \"Reptilia\": (0.30, 0.70),\n"
    "}"
)
# Optimized from val OOF + AST val predictions (708 rows, step=0.1).
# Aves' 0.10/0.90 is non-obvious — SSM alone is higher (0.895) but the
# blend captures decorrelated errors and wins (0.908). If LB shows this
# is over-aggressive (Aves drops more than other taxa lift), bump the
# SSM weight back up — Aves had only 25 evaluable classes in val so
# some overfit risk is real.
new_block = (
    "_TAXON_WEIGHTS = {\n"
    "    # From _09_blend_search.py (per-taxon blend on 708 OOF rows, step=0.1):\n"
    "    #   per-taxon blend overall val_auc_seen = 0.8643\n"
    "    #   vs SSM alone 0.8065  /  vs AST alone 0.7906  /  global α blend 0.8580\n"
    "    \"Aves\":     (0.10, 0.90),  # AST 90% — counterintuitive, blend captures decorrelated errors (0.895 SSM, 0.876 AST → 0.908 blend)\n"
    "    \"Amphibia\": (0.80, 0.20),  # Perch's genus proxy dominates (0.890 SSM vs 0.768 AST)\n"
    "    \"Insecta\":  (0.20, 0.80),  # AST much stronger on AudioSet-trained insects (0.647 SSM, 0.701 AST → 0.784 blend, +0.13)\n"
    "    \"Mammalia\": (0.00, 1.00),  # AST alone wins outright (0.975 vs 0.872)\n"
    "    \"Reptilia\": (0.70, 0.30),  # Single class (Caiman yacare); Perch genus-proxy nailed it on val\n"
    "}"
)

src27 = "".join(nb["cells"][27]["source"])
assert old_block in src27, "could not locate the hand-picked _TAXON_WEIGHTS block"
src27 = src27.replace(old_block, new_block)
nb["cells"][27]["source"] = src27.splitlines(keepends=True)
NB.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
print("✓ patched cell 27 with optimized per-taxon weights")
