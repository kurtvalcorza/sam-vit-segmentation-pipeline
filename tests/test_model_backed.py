"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): point- and box-prompt
evaluation with the per-category breakdown, a one-epoch adaptation of the mask decoder on a dozen drawn scenes, the
artifact round trip, the loader's scope check, the transactional guarantee and — where CUDA is visible — the same path
on the accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sam_vit_segmentation_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MASK_DECODER_PARAMETERS,
    PARAMETER_COUNT,
    WEIGHT_FILE,
    SAMViTSegmentationPipeline,
    interior_point,
    mask_box,
)

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHT_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _record(i, size=128):
    image = Image.new("RGB", (size, size), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, size * 2 // 3, size, size], fill=(60, 179, 75))
    cx, cy, r = size // 2 + (i % 3) * 8, size * 5 // 12, size // 5 + (i % 2) * 6
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=COLOURS[i % 6])
    image.putpixel((i % size, 0), (i % 256, 0, 0))
    yy, xx = np.ogrid[:size, :size]
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r**2
    return {"id": f"s{i:02d}", "image": image, "mask": mask, "point": interior_point(mask), "box": mask_box(mask), "category": COLOURS[i % 6]}


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(16)]


@pytest.fixture(scope="module")
def pipe():
    return SAMViTSegmentationPipeline.from_pretrained(device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_identity_and_frozen_evaluation_with_the_breakdown(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith("mask_decoder.")) == MASK_DECODER_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    point = pipe.evaluate(records[:8], prompt="point")
    box = pipe.evaluate(records[:8], prompt="box")
    assert point["n"] == 8 and 0.0 <= point["iou"] <= 1.0 and point["adapted"] is False and point["prompt"] == "point"
    assert set(point["per_category"]) == set(COLOURS) and point["verdict"] == "measured-small-sample"
    assert box["iou"] > 0.5  # a box around a solid disc on a plain background is easy for the frozen model
    mask = pipe.predict_mask(records[0], prompt="box")
    assert mask.shape == (128, 128) and mask.dtype == np.bool_ and mask.any()


def test_one_epoch_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], records[12:], epochs=1, prompts="mixed")
    assert result["n_trainable"] == MASK_DECODER_PARAMETERS and result["n_total"] == PARAMETER_COUNT
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"iou", "hit_rate", "n"} and result["embedding_seconds"] > 0.0
    assert all(n.startswith("mask_decoder.") for n in result["trainable_names"])
    assert not any(n.startswith(("vision_encoder.", "prompt_encoder.")) for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["adapter"]["prompts"] == "mixed" and manifest["metadata"] == {"note": "test"}
    reloaded = SAMViTSegmentationPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    a = [pipe.predict_mask(r) for r in records[:4]]
    b = [reloaded.predict_mask(r) for r in records[:4]]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[:4])["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:12], None, epochs=2, prompts="point")
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = SAMViTSegmentationPipeline.from_artifact(artifact, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])
    assert reloaded.adapter["prompts"] == "point"


def test_adapt_refuses_bad_hyperparameters(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:12], None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:12], None, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="prompts"):
        pipe.adapt(records[:12], None, epochs=1, prompts="scribble")
    with pytest.raises(ValueError, match="8..5000"):
        pipe.adapt(records[:4], None, epochs=1)
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:12], None, epochs=1)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        SAMViTSegmentationPipeline.from_artifact(fewer, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["mask_decoder.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        SAMViTSegmentationPipeline.from_artifact(extra, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)
    encoder = tmp_path / "encoder"
    shutil.copytree(artifact, encoder)
    (encoder / "manifest.json").write_text(json.dumps({**manifest, "tensors": [*manifest["tensors"], "vision_encoder.patch_embed.projection.weight"]}))
    with pytest.raises(ValueError, match="mask decoder"):
        SAMViTSegmentationPipeline.from_artifact(encoder, device="cpu", weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:12], None, epochs=2, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_evaluate_adapt_and_reload_run_on_a_cuda_device(records, tmp_path):
    cuda = SAMViTSegmentationPipeline.from_pretrained(device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    assert cuda.device == "cuda:0"
    metrics = cuda.evaluate(records[:8], prompt="box")
    assert metrics["iou"] > 0.5
    result = cuda.adapt(records[:12], records[12:], epochs=1)
    assert result["best_epoch"] in (0, 1) and result["history"][1]["train_loss"] > 0.0
    artifact = cuda.save_artifact(tmp_path / "cuda")
    reloaded = SAMViTSegmentationPipeline.from_artifact(artifact, device="cuda:0", weights_dir=DEFAULT_WEIGHTS_DIR)
    a = [cuda.predict_mask(r) for r in records[:4]]
    b = [reloaded.predict_mask(r) for r in records[:4]]
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
