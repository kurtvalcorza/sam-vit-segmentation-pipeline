"""Tier C review findings T78-1..T78-4 (2026-10-25): worked-answer sources, DAT9 overlap, spec 2.5 items."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "sam_vit_segmentation_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    return "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")


def test_t78_1_worked_answers_quote_a_labelled_hosted_run():
    md = _markdown()
    for stale in ("the recorded Kaggle T4 run kept epoch 3", "moved only 0.402 → 0.419", "+0.099",
                  "In the recorded Kaggle T4 run", "recorded Kaggle Tesla T4 run of the previous version"):
        assert stale not in md, stale
    assert "hosted Colab T4 run of blob `edfa7ad6` (2026-10-09)" in md
    assert "0.564 → 0.616 to 0.688 (+0.051 to +0.124" in md and "wall 0.402 → 0.415 to 0.548" in md


def test_t78_2_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "SA-1B" in md
    assert "inferred to be unlikely, not verified" in md


def test_t78_3_preflight_is_dated():
    assert "RTX 5070 Ti pre-flight (local build record, 2026-09-20, an earlier revision)" in _markdown()


def test_t78_4_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] A. Kirillov" in md and "[Online]. Available:" in md
    assert "`DIMER_ISOLATED_ENV`" in md
