"""Final prose pass (Kurt's ruling, 2026-10-26): the prose-rule items flagged on the confirming run are fixed.

Run-status sentences, timing promises and unlabelled expectations are removed or recast
as records of named runs.
Fails on PR #10 head 36507a4.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "sam_vit_segmentation_colab.ipynb"


def _markdown(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_final_prose_pass():
    md = _markdown(NB)
    for gone in (
        'from one GPU run to the next',
        'within a few hundredths of the numbers below',
        'No CPU run of the whole path has been recorded',
    ):
        assert gone not in md, gone
    for kept in (
        'so the validation curve is noisy. The cell first',
        'About 1,160 image encodes',
    ):
        assert kept in md, kept
