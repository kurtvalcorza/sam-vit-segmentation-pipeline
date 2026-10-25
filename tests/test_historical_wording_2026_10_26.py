"""Run-derived prose is a record of named runs, never an expectation (Kurt's ruling, 2026-10-26).

Runs: Colab T4 of blobs edfa7ad6 (2026-10-09), 2b2e4b2b (2026-10-25) and 9a507580 (2026-10-25);
Kaggle T4 of blob 0112e790 (2026-09-20).
"""
import json
from pathlib import Path

NB = Path(__file__).resolve().parents[1] / "tutorials" / "sam_vit_segmentation_colab.ipynb"


def _markdown():
    nb = json.loads(NB.read_text(encoding="utf-8"))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def test_expectations_and_generalisations_are_gone():
    md = _markdown()
    for stale in ("expect roughly 0.61–0.69", "it slipped in every recorded hosted run",
                  "`above box-fill` and `worse` for the box", "seat 0.026 → 0.067 to 0.361",
                  "takes about 12 to 15 minutes", "Budget about 12 to 15 minutes",
                  "expect your own kept epoch to differ", "the point prompt barely moves"):
        assert stale not in md, stale


def test_confirming_run_of_9a507580_is_recorded():
    md = _markdown()
    assert "hosted Colab T4 run of blob `9a507580` (2026-10-25)" in md
    assert "0.784 → 0.768 to 0.793 (−0.016 to +0.009)" in md
    assert "the run of `9a507580` gave `held or improved`" in md
    assert "seat 0.026 → 0.067 to 0.428" in md
    assert "took between 12.3 and 15.2 minutes" in md
