"""Regression tests for the 2026-10-02 Notebook Review Framework v1 findings (SAM-M1..M5, SAM-m1..m3).

The sweep's own tests (tests/test_fleet_sweep_fixes.py) cover the isolated environment, the frozen-decoder
restore and the BYOD path field; these cover what the review asked for beyond them. No torch is needed.
"""

# ruff: noqa: E501  -- long assertion strings and probe values read better on one line
from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam_vit_segmentation_pipeline import samples as sm

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "tutorials" / "sam_vit_segmentation_colab.ipynb"


def _nb() -> dict:
    return json.loads(NB_PATH.read_text(encoding="utf-8"))


def _code(nb: dict) -> list[str]:
    return [c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _markdown(nb: dict) -> str:
    return "\n".join(c["source"] if isinstance(c["source"], str) else "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")


def _cell(marker: str) -> str:
    found = [c for c in _code(_nb()) if marker in c]
    assert len(found) == 1, marker
    return found[0]


def _byod_zip(path: Path, n: int) -> Path:
    rng = np.random.default_rng(0)
    with zipfile.ZipFile(path, "w") as archive:
        for i in range(n):
            pixels = rng.integers(0, 255, size=(48, 64, 3), dtype=np.uint8)  # distinct images: no de-duplication
            mask = np.zeros((48, 64), dtype=np.uint8)
            mask[10:38, 12:52] = 255
            for name, array in ((f"img{i:03d}.png", pixels), (f"img{i:03d}_mask.png", mask)):
                buffer = path.parent / "tmp.png"
                Image.fromarray(array).save(buffer)
                archive.write(buffer, name)
    return path


# --- SAM-M3: the BYOD minimum is computed, stated, and a short split is refused by name ------------------------------


def test_m3_byod_minimum_is_fifty_under_the_default_fractions():
    assert sm.byod_minimum_records() == 50
    assert sm.MIN_RECORDS == 8


@pytest.mark.parametrize("n", [8, 49])
def test_m3_a_short_split_is_refused_naming_the_split_its_size_and_the_total(n):
    records = [_record(i) for i in range(n)]
    splits = sm.split_dataset(records, seed=42)
    with pytest.raises(ValueError, match=r"the (test|validation) split has \d+ records but every split needs at least 8") as info:
        sm.check_split_sizes(splits)
    assert "Supply at least 50 distinct images" in str(info.value)


def test_m3_fifty_images_pass_every_split():
    splits = sm.split_dataset([_record(i) for i in range(50)], seed=42)
    assert sm.check_split_sizes(splits) == {"test": 10, "validation": 8, "train": 32}
    for part in splits.values():
        sm.validate_dataset(part)


def _record(i: int) -> dict:
    rng = np.random.default_rng(i)
    image = Image.fromarray(rng.integers(0, 255, size=(40, 40, 3), dtype=np.uint8))
    mask = np.zeros((40, 40), dtype=bool)
    mask[8:32, 8:32] = True
    return {"id": f"r{i:03d}", "image": image, "mask": mask, "point": sm.interior_point(mask), "box": sm.mask_box(mask)}


def _run_section_4(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, zip_path: Path) -> dict:
    source = _cell("records = load_byod_dataset(byod_path)")
    source = source.replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_PATH = ''", f"BYOD_PATH = {str(zip_path)!r}", 1)
    monkeypatch.chdir(tmp_path)
    namespace = {"os": os, "Path": Path, **{name: getattr(sm, name) for name in sm.__dict__ if not name.startswith("__")}}
    exec(compile(source, "<section 4>", "exec"), namespace)
    return namespace


def test_m3_section_4_runs_a_fifty_image_zip_and_refuses_forty_nine(tmp_path, monkeypatch, capsys):
    """The notebook's own Section 4 cell with USE_BYOD on: 50 images pass every split; 49 are refused by name."""
    ok = _run_section_4(tmp_path, monkeypatch, _byod_zip(tmp_path / "fifty.zip", 50))
    assert {k: len(v) for k, v in ok["splits"].items()} == {"test": 10, "validation": 8, "train": 32}
    assert "'byod_split_sizes': {'test': 10, 'validation': 8, 'train': 32}" in capsys.readouterr().out
    with pytest.raises(ValueError, match=r"the validation split has 7 records.*Supply at least 50 distinct images"):
        _run_section_4(tmp_path, monkeypatch, _byod_zip(tmp_path / "short.zip", 49))


def test_m3_the_stated_minimum_is_fifty_everywhere():
    markdown = _markdown(_nb())
    assert "at least eight images" not in markdown
    assert "at least **50 distinct images**" in markdown
    assert "needs at least **50** distinct images" in markdown


# --- SAM-M4: quality expectations are loud on the sample path only ---------------------------------------------------


def _tail(source: str, start: str) -> str:
    return source[source.index(start):]


@pytest.mark.parametrize("use_byod", [False, True])
def test_m4_section_6_frozen_box_check_is_loud_on_the_sample_only(use_byod):
    tail = _tail(_cell("frozen_box = pipe.evaluate(test_records, prompt='box')"), "frozen_verdict = ")
    # Round objects: the area-given disk beats the box prompt (the review's probe: 0.9964 vs 0.9886).
    ns = {"frozen_box": {"iou": 0.9886}, "baseline_disk": {"iou": 0.9964}, "USE_BYOD": use_byod}
    if use_byod:
        exec(tail, ns)
        assert ns["frozen_verdict"] == "not above the centre-disk baseline"
    else:
        with pytest.raises(RuntimeError, match="Sample-path expectation broken"):
            exec(tail, ns)
    ok = {"frozen_box": {"iou": 0.784}, "baseline_disk": {"iou": 0.442}, "USE_BYOD": False}
    exec(tail, ok)
    assert ok["frozen_verdict"] == "above the centre-disk baseline"


@pytest.mark.parametrize("use_byod", [False, True])
def test_m4_section_8_adaptation_check_is_loud_on_the_sample_only(use_byod):
    s8 = _cell("adapted_point = pipe.evaluate(test_records, prompt='point')")
    verdicts = s8[s8.index("delta_point = "):s8.index("for key, row in comparison.items():")]
    check = s8[s8.index("if not USE_BYOD and adaptation_verdict"):]
    ns = {"adapted_point": {"iou": 0.5}, "frozen_point": {"iou": 0.5}, "baseline_box": {"iou": 0.6}, "adapted_box": {"iou": 0.8},
          "frozen_box": {"iou": 0.8}, "frozen_verdict": "above the centre-disk baseline", "comparison": {}, "USE_BYOD": use_byod}
    exec(verdicts, ns)
    assert ns["comparison"]["verdicts"]["adapted_point_vs_frozen_point_iou"] == "no gain"
    if use_byod:
        exec(check, ns)
    else:
        with pytest.raises(RuntimeError, match="Sample-path expectation broken"):
            exec(check, ns)
    assert s8.index("json.dump(evaluation_report_payload") < s8.index("if not USE_BYOD and adaptation_verdict")


# --- SAM-M2 / SAM-M5: exact re-run steps and a complete activity ------------------------------------------------------


def test_m2_m5_activity_and_rerun_steps():
    markdown = _markdown(_nb())
    for step in ("*Predict:*", "*Change:*", "*Run:*", "*Observe:*", "*Explain:*"):
        assert step in markdown
    assert "run Sections 7, 8 and 9 again, in that order" in markdown
    assert "another `SPLIT_SEED` (Sections 4–9" in markdown
    assert "run Sections 4 to 9 again in order" in markdown
    assert "re-run from that cell" not in markdown
    assert "after this notebook you can: explain why one click" in markdown
    assert markdown.count("**Predict:**") >= 2 and markdown.count("Check your reasoning") >= 2
    for section in ("## Troubleshooting", "## Glossary", "## Conclusion (your notes)", "How to use this notebook", "Roadmap"):
        assert section in markdown


# --- SAM-m1 / SAM-m3: numbers name their run, CPU time is an estimate, a GPU is recommended ---------------------------


def test_m1_expectations_name_their_run():
    markdown = _markdown(_nb())
    assert "That is the claim" not in markdown
    assert "wall, building, cabinet, seat — gain the most" not in markdown
    assert "Wall, building, cabinet and seat gained most" not in markdown
    assert "moved only 0.402 → 0.419" in markdown
    for match in re.finditer(r"0\.691", markdown):
        window = markdown[max(0, match.start() - 200): match.end() + 200]
        assert "pre-flight" in window, window


def test_m3_cpu_time_is_an_estimate_and_a_gpu_is_recommended():
    markdown = _markdown(_nb())
    assert "an hour or more" not in markdown
    assert "Choose a GPU runtime (Colab T4)" in markdown or "Choose a **GPU** runtime" in markdown
    assert "*estimated* (not measured on a hosted CPU)" in markdown
    assert "(an estimate, see Prerequisites)" in markdown


# --- SAM-S2 / SAM-S5 ---------------------------------------------------------------------------------------------------


def test_s2_declares_spec_2_2_and_s5_drops_the_pillow_mode_argument():
    nb = _nb()
    assert nb["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert "mode='RGB'" not in "\n".join(_code(nb))
