"""Prose numbers stay true across every recorded hosted run (Kurt's ruling, 2026-10-25).

Runs: Colab T4 of blob edfa7ad6 (2026-10-09), Colab T4 of blob 2b2e4b2b (2026-10-25),
Kaggle T4 of blob 0112e790 (2026-09-20).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "sam_vit_segmentation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_single_run_claims_that_failed_a_later_run_are_gone():
    md = _markdown()
    for stale in ("[0.955, 1.012, 0.979]", "roughly 0.65–0.71", "within about 0.01 of frozen",
                  "both recorded runs gained about a tenth", "912 s in the recorded run",
                  "worked answer from the recorded Kaggle T4 run", "defaults reproduce the recorded run"):
        assert stale not in md, stale


def test_varying_quantities_are_stated_as_ranges_over_the_named_runs():
    md = _markdown()
    assert "`[0.957, 1.0116, 0.982]`" in md
    assert "0.784 → 0.768 to 0.793" in md
    assert ("epoch 6 in the first, epoch 2 in the second, epoch 6 in the third and epoch 3 in the Kaggle T4 run of blob `0112e790`"
            in md)
    assert "each recorded hosted run gained (by 0.051 to 0.124)" in md
    assert "738.7 s to 912.1 s across the four recorded hosted runs" in md
