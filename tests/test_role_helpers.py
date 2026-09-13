"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sam_vit_segmentation_pipeline import (
    INPUT_SCHEMA,
    MAX_IMAGE_SIDE,
    MAX_PROMPTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    NUM_MULTIMASK_OUTPUTS,
    evaluation_report,
    validate_inputs,
)

RECTANGLE = [40, 60, 140, 180]


def _image(width: int = 320, height: int = 240) -> Image.Image:
    return Image.new("RGB", (width, height), (128, 128, 128))


def _reference(width: int = 320, height: int = 240) -> np.ndarray:
    mask = Image.new("1", (width, height), 0)
    ImageDraw.Draw(mask).rectangle(RECTANGLE, fill=1)
    return np.asarray(mask, dtype=np.bool_)


def _result(masks: np.ndarray, scores: list[float]) -> dict:
    return {
        "masks": masks,
        "iou_scores": scores,
        "multimask": masks.shape[0] > 1,
        "points": [[90.0, 120.0]],
        "point_labels": [1],
        "box": None,
        "width": masks.shape[2],
        "height": masks.shape[1],
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(
        _image(), points=[[90, 120]], point_labels=[1], names=["synthetic_scene_320x240.png"]
    )
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["points"] == [1, MAX_PROMPTS]
    assert manifest["schema"]["multimask_outputs"] == NUM_MULTIMASK_OUTPUTS
    assert manifest["inputs"] == [
        {
            "id": "synthetic_scene_320x240.png",
            "mode": "RGB",
            "size": [320, 240],
            "n_points": 1,
            "has_box": False,
        }
    ]
    assert (manifest["points"], manifest["point_labels"]) == ([[90.0, 120.0]], [1])
    assert manifest["box"] is None and manifest["multimask"] is True
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_box_prompt_default_id() -> None:
    manifest = validate_inputs(_image(), box=[200, 80, 281, 161], multimask=False)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["inputs"][0]["has_box"] is True
    assert manifest["inputs"][0]["n_points"] == 0
    assert manifest["box"] == [200.0, 80.0, 281.0, 161.0]
    assert manifest["multimask"] is False


def test_validate_inputs_rejects_like_segment() -> None:
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        validate_inputs(_image(MAX_IMAGE_SIDE + 1, 64), points=[[1, 1]], point_labels=[1])
    with pytest.raises(TypeError, match="PIL.Image.Image"):
        validate_inputs("not an image", points=[[1, 1]], point_labels=[1])
    with pytest.raises(ValueError, match="at least one of points or box"):
        validate_inputs(_image())
    with pytest.raises(ValueError, match="outside image"):
        validate_inputs(_image(), points=[[400, 10]], point_labels=[1])
    with pytest.raises(ValueError, match="one 0/1 entry per point"):
        validate_inputs(_image(), points=[[10, 10]])
    with pytest.raises(TypeError, match="multimask must be a bool"):
        validate_inputs(_image(), points=[[10, 10]], point_labels=[1], multimask="yes")
    with pytest.raises(ValueError, match="names must have exactly one entry"):
        validate_inputs(_image(), points=[[10, 10]], point_labels=[1], names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference_mask() -> None:
    masks = np.zeros((3, 240, 320), dtype=np.bool_)
    report = evaluation_report(_result(masks, [0.01, 0.99, 0.40]))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_masks"] == 3
    assert report["best_candidate"] == 1
    assert "mask_iou" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference_mask() -> None:
    reference = _reference()
    masks = np.stack([np.zeros_like(reference), reference, np.ones_like(reference)])
    report = evaluation_report(_result(masks, [0.01, 0.99, 0.40]), reference, sample_kind="synthetic")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "synthetic"
    assert {metric["id"] for metric in report["metrics"]} == {"mask_iou"}
    values = {metric["candidate"]: metric["value"] for metric in report["metrics"]}
    assert values[1] == pytest.approx(1.0)
    assert values[0] == 0.0
    assert [metric["selected"] for metric in report["metrics"]] == [False, True, False]
    assert report["reference_area_px"] == int(reference.sum())
    assert all(metric["estimation"] for metric in report["metrics"])


def test_evaluation_report_states_that_predicted_iou_may_exceed_one() -> None:
    masks = np.zeros((1, 240, 320), dtype=np.bool_)
    report = evaluation_report(_result(masks, [1.012]))
    assert "may exceed 1.0" in report["score_semantics"]
    assert report["iou_scores_model_predicted"] == [1.012]
    assert report["mask_areas_px"] == [0]
