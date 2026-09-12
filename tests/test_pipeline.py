import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from sam_vit_segmentation_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_IMAGE_SIDE,
    MAX_PROMPTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    NUM_MULTIMASK_OUTPUTS,
    SAMViTSegmentationPipeline,
    mask_iou,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "facebook/sam-vit-base"
    assert NUM_MULTIMASK_OUTPUTS == 3
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "sam"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "sam"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_from_pretrained_refuses_without_snapshot_or_download(tmp_path):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        SAMViTSegmentationPipeline.from_pretrained(weights_dir=tmp_path, allow_download=False)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def _fake_pipeline(calls: list | None = None) -> SAMViTSegmentationPipeline:
    def runner(image, points, labels, box, multimask):
        if calls is not None:
            calls.append((image.mode, points, labels, box, multimask))
        k = NUM_MULTIMASK_OUTPUTS if multimask else 1
        masks = np.zeros((k, image.height, image.width), dtype=np.bool_)
        masks[:, : image.height // 2, :] = True
        return masks, [0.9, 0.5, 0.1][:k]

    return SAMViTSegmentationPipeline(runner, "cpu")


def test_segment_points_output_fields():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.segment(Image.new("L", (40, 30)), points=[(10, 5), [20, 25]], point_labels=[1, 0])
    assert result["masks"].shape == (3, 30, 40) and result["masks"].dtype == np.bool_
    assert result["iou_scores"] == [0.9, 0.5, 0.1] and result["multimask"] is True
    assert result["points"] == [[10.0, 5.0], [20.0, 25.0]] and result["point_labels"] == [1, 0]
    assert result["box"] is None and (result["width"], result["height"]) == (40, 30)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert calls == [("RGB", [[10.0, 5.0], [20.0, 25.0]], [1, 0], None, True)]


def test_segment_box_single_mask():
    pipe = _fake_pipeline()
    result = pipe.segment(Image.new("RGB", (40, 30)), box=[5, 5, 35, 25], multimask=False)
    assert result["masks"].shape == (1, 30, 40) and result["iou_scores"] == [0.9]
    assert result["box"] == [5.0, 5.0, 35.0, 25.0] and result["points"] is None


def test_segment_rejects_bad_inputs():
    pipe = _fake_pipeline()
    img = Image.new("RGB", (40, 30))
    with pytest.raises(TypeError):
        pipe.segment(np.zeros((30, 40, 3), dtype=np.uint8), box=[0, 0, 10, 10])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.segment(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)), box=[0, 0, 5, 5])
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.segment(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)), box=[0, 0, 5, 5])
    with pytest.raises(ValueError, match="at least one"):
        pipe.segment(img)
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        pipe.segment(img, points=[(1, 1)] * (MAX_PROMPTS + 1), point_labels=[1] * (MAX_PROMPTS + 1))
    with pytest.raises(ValueError, match="point_labels"):
        pipe.segment(img, points=[(1, 1)])
    with pytest.raises(ValueError, match="outside image"):
        pipe.segment(img, points=[(40, 1)], point_labels=[1])
    with pytest.raises(ValueError, match="label must be 0 or 1"):
        pipe.segment(img, points=[(1, 1)], point_labels=[2])
    with pytest.raises(ValueError, match="xyxy"):
        pipe.segment(img, box=[10, 10, 5, 20])
    with pytest.raises(ValueError, match="xyxy"):
        pipe.segment(img, box=[0, 0, 41, 20])
    with pytest.raises(TypeError, match="multimask"):
        pipe.segment(img, box=[0, 0, 10, 10], multimask="yes")


def test_segment_rejects_backend_shape_mismatch():
    pipe = SAMViTSegmentationPipeline(lambda *_: (np.zeros((2, 30, 40), dtype=np.bool_), [0.5, 0.5]), "cpu")
    with pytest.raises(RuntimeError):
        pipe.segment(Image.new("RGB", (40, 30)), box=[0, 0, 10, 10])


def test_mask_iou():
    a = np.zeros((4, 4), dtype=np.bool_)
    a[:2] = True
    b = np.zeros((4, 4), dtype=np.bool_)
    b[1:3] = True
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == pytest.approx(4 / 12)
    assert mask_iou(np.zeros((2, 2), dtype=np.bool_), np.zeros((2, 2), dtype=np.bool_)) == 0.0
    with pytest.raises(ValueError):
        mask_iou(a, b[:2])
    with pytest.raises(TypeError):
        mask_iou(a.astype(np.uint8), b)
