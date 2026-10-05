"""Regression tests for the 2026-10-05 fleet-sweep fixes (SWP-R restart guard, SWP-G guided layer and the
repository-specific SWP-A / SWP-F / SWP-B fixes recorded in docs/reviews/2026-10-05-fleet-sweep/).

Every test needs only CI's dependencies. The notebooks' own cell sources are executed with stand-ins; no model, no
network and no torch are needed.
"""
# ruff: noqa: E501

from __future__ import annotations

import functools
import hashlib
import importlib.util
import json
import re
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ['sam_vit_segmentation_colab']
LOCK = ROOT / 'tutorials/requirements-colab.lock.txt'
MIN_PREDICT = {'sam_vit_segmentation_colab': 6}


@functools.cache
def _nb_text(name: str) -> str:
    return (ROOT / "tutorials" / f"{name}.ipynb").read_text(encoding="utf-8")


def _nb(name: str) -> dict:
    return json.loads(_nb_text(name))


def _code_cells(notebook: dict) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == "code"]


def _cell(notebook: dict, marker: str) -> str:
    found = [c["source"] for c in _code_cells(notebook) if marker in c["source"]]
    assert len(found) == 1, f"expected one code cell containing {marker!r}, found {len(found)}"
    return found[0]


def _build():
    spec = importlib.util.spec_from_file_location("_sweep_build_notebook", ROOT / "tools" / "build_notebook.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    return build


# --- SWP-R: no in-kernel install, no restart, idempotent Section 1 (shared by every notebook) -------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_nothing_is_pip_installed_into_the_kernel_and_no_restart_is_requested(name):
    notebook = _nb(name)
    code = "\n".join(c["source"] for c in _code_cells(notebook))
    assert "pip install" not in code and "'-m', 'pip'" not in code
    assert "restart the runtime" not in json.dumps(notebook).lower()
    kernel = [c for c in _code_cells(notebook) if "# dimer: kernel cell" in c["source"]]
    assert len(kernel) == 1, "exactly one cell may run in the kernel"
    source = kernel[0]["source"]
    for needed in ("'--require-hashes', '--only-binary', ':all:'", "'--managed-python'", "UV_SHA256", "LOCK_SHA256"):
        assert needed in source
    # The worker gets a clean interpreter environment and a non-interactive matplotlib backend.
    for needed in ('MPLBACKEND="Agg"', '"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"'):
        assert needed in source
    assert notebook["metadata"]["dimer"]["environment"].startswith("isolated hash-locked uv environment")


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_r_carried_lock_is_the_committed_lock_and_pins_every_runtime_pin(name):
    source = _cell(_nb(name), "# dimer: kernel cell")
    lock_text = LOCK.read_text(encoding="utf-8")
    digest = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    assert digest == hashlib.sha256(lock_text.encode("utf-8")).hexdigest()
    assert f"LOCK_TEXT = r'''{lock_text}'''" in source
    build = _build()
    build.check_lock(build._pins(ROOT), lock_text)  # raises SystemExit on any drift


class _Shell:
    def __init__(self) -> None:
        self.input_transformers_cleanup: list = []


def test_swp_r_section_1_is_idempotent_and_keeps_the_live_worker(tmp_path, monkeypatch, capsys):
    """Re-running the Section 1 cell reuses the matching environment (no download) and keeps the live worker, so the
    variables later cells created survive and the cells after it are not stranded."""
    source = _cell(_nb(NOTEBOOKS[0]), "# dimer: kernel cell")
    lock_sha = re.search(r"^LOCK_SHA256 = '([0-9a-f]{64})'$", source, re.M).group(1)
    env = tmp_path / "env"
    (env / "bin").mkdir(parents=True)
    (env / "bin" / "python").symlink_to(sys.executable)  # stand-in interpreter for the isolated environment
    (env / ".dimer-lock-sha256").write_text(lock_sha + "\n", encoding="utf-8")
    monkeypatch.setenv("DIMER_ISOLATED_ENV", str(env))
    monkeypatch.delenv("DIMER_NOTEBOOK_CI_PREINSTALLED", raising=False)
    shell = _Shell()
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: shell
    ipython_display = types.ModuleType("IPython.display")
    ipython_display.display = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "IPython", ipython)
    monkeypatch.setitem(sys.modules, "IPython.display", ipython_display)

    def no_download(*args, **kwargs):
        raise AssertionError("a matching environment must be reused, not downloaded again")

    monkeypatch.setattr("urllib.request.urlopen", no_download)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(source, "<section 1>", "exec"), namespace)
    runtime = namespace["_DIMER_ISOLATED_RUNTIME"]
    try:
        assert "'reused': True" in capsys.readouterr().out
        runtime.run("learner_value = 41 + 1\n")
        exec(compile(source, "<section 1 again>", "exec"), namespace)  # the learner re-runs Section 1 on its own
        assert namespace["_DIMER_ISOLATED_RUNTIME"] is runtime and runtime.alive()
        assert [t.__name__ for t in shell.input_transformers_cleanup] == ["_route_to_isolated_runtime"]
        runtime.run("print('value', learner_value)\n")
        assert "value 42" in capsys.readouterr().out
        assert namespace["_route_to_isolated_runtime"](["x = 1\n"]) == ["_DIMER_ISOLATED_RUNTIME.run('x = 1\\n')\n"]
        assert namespace["_route_to_isolated_runtime"]([source]) == [source]  # the kernel cell itself stays in the kernel
        with pytest.raises(RuntimeError, match="ZeroDivisionError"):
            runtime.run("1 / 0\n")
    finally:
        runtime.close()


# --- SWP-G: the guided layer and infrastructure labelling (shared) ----------------------------------------------------


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_guided_layer_is_present(name):
    notebook = _nb(name)
    markdown = "\n".join(c["source"] for c in notebook["cells"] if c["cell_type"] == "markdown")
    for heading in (
        "**Who this notebook is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
        "## Change one thing (next experiments)",
    ):
        assert heading in markdown, heading
    assert markdown.count("**Predict:**") >= MIN_PREDICT[name]
    assert markdown.count("<details><summary>Check your reasoning</summary>") >= MIN_PREDICT[name]
    assert "Run all completes in one pass" in markdown


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_infrastructure_cells_are_labelled_and_collapsed(name):
    cells = _code_cells(_nb(name))
    infra = [c for c in cells if c["metadata"].get("cellView") == "form"]
    assert any("# dimer: kernel cell" in c["source"] for c in infra)
    assert any(c["metadata"].get("dimer", {}).get("embedded_module") for c in infra)
    assert any(c["source"].startswith("# @title Infrastructure: stage and digest-verify") for c in infra)
    learner = [c for c in cells if c["metadata"].get("cellView") != "form"]
    assert learner and all("# @title Infrastructure" not in c["source"] for c in learner)


@pytest.mark.parametrize("name", NOTEBOOKS)
def test_swp_g_no_template_placeholders_leak(name):
    notebook = _nb(name)
    text = "\n".join(
        c["source"] for c in notebook["cells"] if not c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    )
    for leftover in ("{{", "{MODEL_ID}", "{stem}", "@P:"):
        assert leftover not in text, leftover


def _colab(monkeypatch, upload) -> None:
    google = types.ModuleType("google")
    google.__path__ = []
    colab_mod = types.ModuleType("google.colab")
    files = types.ModuleType("google.colab.files")
    files.upload = upload
    colab_mod.files = files
    google.colab = colab_mod
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab_mod)
    monkeypatch.setitem(sys.modules, "google.colab.files", files)


def _no_colab(monkeypatch) -> None:
    monkeypatch.setitem(sys.modules, "google.colab", None)  # import fails as it does on Kaggle / Jupyter


# --- repository-specific: SWP-G checkpoint numbers, SWP-A verdicts, SWP-F frozen decoder snapshot, SWP-B BYOD path ----

import os  # noqa: E402

NB = NOTEBOOKS[0]
RECORD = ROOT / "docs" / "release-verification.md"


def test_swp_g_checkpoint_answers_quote_the_recorded_runs():
    """Every number in a Check-your-reasoning answer is in the recorded 2026-09-20 Kaggle T4 run or the recorded pre-flights."""
    markdown = "\n".join(c["source"] for c in _nb(NB)["cells"] if c["cell_type"] == "markdown")
    checks = re.findall(r"<details><summary>Check your reasoning</summary>(.*?)</details>", markdown, re.S)
    assert len(checks) == 6
    record = RECORD.read_text(encoding="utf-8")
    hosted = ("0.442", "0.614", "0.564", "0.663", "0.784", "0.778", "0.286", "0.657", "0.857", "0.929", "0.099", "0.049", "-0.006", "912.1 s")
    preflight = ("0.691", "0.788", "0.629", "0.645", "0.694", "0.654", "0.639", "0.715", "[0.955, 1.012, 0.979]", "1.000", "8/8")
    for number in hosted + preflight:
        assert number in record, number
    assert all(n in checks[2] for n in ("0.442", "0.564", "0.614", "0.784"))
    assert all(n in checks[4] for n in ("0.564", "0.663", "0.614", "0.049", "0.784", "0.778", "0.691", "0.788"))
    assert "0.629" in checks[3] and "0.715" in checks[3]


def test_swp_a_no_quality_assert_remains_and_verdicts_reach_the_report():
    code = "\n".join(c["source"] for c in _code_cells(_nb(NB)))
    assert "assert frozen_box['iou'] > baseline_disk['iou']" not in code
    assert "assert adapted_point['iou'] > frozen_point['iou']" not in code
    quality = [line for line in code.splitlines() if line.strip().startswith("assert ") and re.search(r"\['(iou|hit_rate)'\]", line)]
    assert quality == []
    assert "assert parity['identical_masks'] == parity['of']" in code  # reload parity stays a hard check
    s6 = _cell(_nb(NB), "frozen_point = pipe.evaluate(test_records, prompt='point')")
    frozen_line = next(line for line in s6.splitlines() if line.startswith("frozen_verdict = "))
    s8 = _cell(_nb(NB), "adapted_point = pipe.evaluate(test_records, prompt='point')")
    block = s8[s8.index("delta_point = ") : s8.index("for key, row in comparison.items():")]
    for frozen, disk, adapted, box_fill, fbox, abox, expected in (
        (0.784, 0.442, 0.663, 0.614, 0.784, 0.778, ("above the centre-disk baseline", "improved", "above box-fill", "worse")),
        (0.3, 0.442, 0.5, 0.614, 0.3, 0.3, ("not above the centre-disk baseline", "no gain", "not above box-fill", "held or improved")),
        (0.784, 0.442, 0.4, 0.614, 0.784, 0.79, ("above the centre-disk baseline", "worse", "not above box-fill", "held or improved")),
    ):
        ns = {"frozen_box": {"iou": frozen}, "baseline_disk": {"iou": disk}, "adapted_point": {"iou": adapted}, "frozen_point": {"iou": 0.5}, "baseline_box": {"iou": box_fill}, "adapted_box": {"iou": abox}, "comparison": {}}
        ns["frozen_box"] = {"iou": fbox}
        ns["frozen_verdict"] = None
        exec(frozen_line, {**ns, "frozen_box": {"iou": frozen}}, ns)
        exec(block, ns)
        verdicts = ns["comparison"]["verdicts"]
        assert (verdicts["frozen_box_vs_centre_disk"], verdicts["adapted_point_vs_frozen_point_iou"], verdicts["adapted_point_vs_box_fill"], verdicts["adapted_box_vs_frozen_box_iou"]) == expected
    assert s8.index("comparison['verdicts'] = ") < s8.index("json.dump(evaluation_report_payload")
    assert "'comparison': comparison," in _cell(_nb(NB), "result_payload = {")


class _Tensor:
    def __init__(self, value, device="cuda", dtype="float32"):
        self.value, self.device, self.dtype = value, device, dtype

    def detach(self):
        return self

    def cpu(self):
        return _Tensor(self.value, "cpu", self.dtype)

    def clone(self):
        return _Tensor(self.value, self.device, self.dtype)

    def to(self, device, dtype):
        return _Tensor(self.value, device, dtype)


class _Model:
    def __init__(self):
        self.state = {"mask_decoder.head.weight": _Tensor(0.0), "vision_encoder.weight": _Tensor(5.0)}
        self.loaded = []

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, part, strict=True):
        assert not strict and set(part) <= set(self.state)
        self.state.update(part)
        self.loaded.append({k: v.value for k, v in part.items()})

    def named_parameters(self):
        return [(k, v) for k, v in self.state.items()]

    def eval(self):
        return self


def _section_6_head(nb: dict) -> str:
    source = _cell(nb, "def restore_frozen_decoder():")
    return source[: source.index("METRICS = ('iou', 'hit_rate')")]


def test_swp_f_section_6_snapshots_the_frozen_decoder_once_and_restores_it_on_a_rerun(capsys):
    head = _section_6_head(_nb(NB))
    model = _Model()
    pipe = types.SimpleNamespace(_require_model=lambda: (model, None), adapter=None)
    ns = {"pipe": pipe, "_trainable_names": lambda m: [n for n, _ in m.named_parameters() if n.startswith("mask_decoder.")]}
    exec(compile(head, "<section 6>", "exec"), ns)
    assert set(ns["frozen_decoder_state"]) == {"mask_decoder.head.weight"} and ns["frozen_decoder_state"]["mask_decoder.head.weight"].device == "cpu"
    assert "'frozen_decoder_snapshot': 'taken'" in capsys.readouterr().out
    model.state["mask_decoder.head.weight"] = _Tensor(3.0)  # Section 7 trained in place
    pipe.adapter = {"best_epoch": 6}
    exec(compile(head, "<section 6 again>", "exec"), ns)
    assert model.state["mask_decoder.head.weight"].value == 0.0 and model.state["vision_encoder.weight"].value == 5.0
    assert model.loaded[-1] == {"mask_decoder.head.weight": 0.0} and pipe.adapter is None
    assert "'frozen_decoder_snapshot': 'restored'" in capsys.readouterr().out


def test_swp_f_section_7_restores_the_frozen_decoder_before_training():
    s7 = _cell(_nb(NB), "adapt_result = pipe.adapt(")
    assert s7.index("restore_frozen_decoder()") < s7.index("adapt_result = pipe.adapt(")


def _byod_source(nb: dict):
    source = _cell(nb, "def byod_source(")
    helper = source[source.index("def byod_source(") : source.index("if USE_BYOD:")]
    ns = {"os": os, "Path": Path}
    exec(helper, ns)
    return ns["byod_source"]


def test_swp_b_byod_path_accepts_a_zip_or_directory_and_refusals_name_the_file(tmp_path, monkeypatch):
    helper = _byod_source(_nb(NB))
    good = tmp_path / "mine.zip"
    good.write_bytes(b"PK")
    assert helper(str(good), "zip") == good
    folder = tmp_path / "masks"
    folder.mkdir()
    assert helper(str(folder), "zip") == folder
    with pytest.raises(FileNotFoundError, match="missing.zip"):
        helper(str(tmp_path / "missing.zip"), "zip")
    bad = tmp_path / "wrong.tar"
    bad.write_bytes(b"x")
    with pytest.raises(ValueError, match=r"wrong.tar: expected a zip"):
        helper(str(bad), "zip")
    _no_colab(monkeypatch)
    with pytest.raises(RuntimeError, match="only in Google Colab"):
        helper("", "zip")


def test_swp_b_cancelled_upload_is_named_and_one_upload_is_saved(tmp_path, monkeypatch):
    helper = _byod_source(_nb(NB))
    _colab(monkeypatch, lambda: {})
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="Upload exactly one"):
        helper("", "zip")
    _colab(monkeypatch, lambda: {"up.zip": b"1"})
    saved = helper("", "zip")
    assert saved.read_bytes() == b"1" and saved.parent.name == "work"


def test_swp_b_byod_gate_is_off_by_default_and_has_a_path_field():
    code = "\n".join(c["source"] for c in _code_cells(_nb(NB)))
    assert "USE_BYOD = False  # @param" in code and "BYOD_PATH = ''  # @param" in code
    assert "file_name, payload = next(iter(uploaded.items()))" not in code
