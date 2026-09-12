"""Promptable image segmentation over the pinned ``facebook/sam-vit-base`` (SAM v1, ViT-B) checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``segment``: one object per call from point
clicks and/or one box, returning boolean masks at input resolution plus the model's predicted IoU per mask.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "facebook/sam-vit-base"
MODEL_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "sam-vit-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Input ceilings. The processor resizes the longest edge to 1024 and pads to 1024x1024 (see
# preprocessor_config.json), so model cost is fixed; the caller's resolution only sets the size of the
# up-sampled output masks. Prompts are one object per call: up to MAX_PROMPTS point clicks (label 1 =
# foreground, 0 = background) and/or one xyxy box.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
NUM_MULTIMASK_OUTPUTS = 3  # config.json mask_decoder_config.num_multimask_outputs
MASK_THRESHOLD = 0.0  # logits above this become True in the binarised masks (processor default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks of identical shape; the primitive behind any mIoU."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise TypeError("mask_iou expects boolean arrays")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def validate_prompts(
    width: int,
    height: int,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[int] | None,
    box: Sequence[float] | None,
) -> tuple[list[list[float]] | None, list[int] | None, list[float] | None]:
    """Check one object's prompts: points inside the image with 0/1 labels, and/or one xyxy box inside it."""
    if points is None and box is None:
        raise ValueError("provide at least one of points or box")
    clean_points = clean_labels = None
    if points is not None:
        if isinstance(points, str) or not isinstance(points, Sequence):
            raise TypeError("points must be a sequence of [x, y] pairs")
        if not 1 <= len(points) <= MAX_PROMPTS:
            raise ValueError(f"point count {len(points)} outside 1..MAX_PROMPTS {MAX_PROMPTS}")
        if point_labels is None or len(point_labels) != len(points):
            raise ValueError("point_labels must be given with one 0/1 entry per point")
        clean_points, clean_labels = [], []
        for (x, y), label in zip(points, point_labels, strict=True):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"point ({x}, {y}) outside image {width}x{height}")
            if label not in (0, 1) or isinstance(label, bool):
                raise ValueError(f"point label must be 0 or 1, got {label!r}")
            clean_points.append([float(x), float(y)])
            clean_labels.append(int(label))
    elif point_labels is not None:
        raise ValueError("point_labels given without points")
    clean_box = None
    if box is not None:
        if len(box) != 4:
            raise ValueError("box must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(v) for v in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"box {box!r} is not a non-empty xyxy box inside image {width}x{height}")
        clean_box = [x0, y0, x1, y1]
    return clean_points, clean_labels, clean_box


@dataclass
class SAMViTSegmentationPipeline:
    """Promptable image segmentation (points/box -> masks) over the pinned SAM ViT-B checkpoint.

    Prompted mode only (`SamModel` + `SamProcessor`); automatic grid-prompt mask generation is not exposed."""

    _runner: Callable[..., tuple[np.ndarray, list[float]]]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SAMViTSegmentationPipeline:
        import torch
        from transformers import SamModel, SamProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        processor = SamProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = SamModel.from_pretrained(
            source, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image, points, labels, box, multimask) -> tuple[np.ndarray, list[float]]:
            prompt_kwargs: dict[str, Any] = {}
            if points is not None:
                prompt_kwargs["input_points"] = [[points]]
                prompt_kwargs["input_labels"] = [[labels]]
            if box is not None:
                prompt_kwargs["input_boxes"] = [[box]]
            inputs = processor(images=image, return_tensors="pt", **prompt_kwargs).to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs, multimask_output=multimask)
            # SAM v1 pads the resized image to 1024x1024; reshaped_input_sizes lets post-processing strip it.
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                mask_threshold=MASK_THRESHOLD,
            )[0]
            return masks[0].numpy().astype(np.bool_), [float(v) for v in outputs.iou_scores[0, 0].tolist()]

        return cls(runner, resolved_device)

    def segment(
        self,
        image: Image.Image,
        *,
        points: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        box: Sequence[float] | None = None,
        multimask: bool = True,
    ) -> dict[str, Any]:
        """Segment one object; returns K boolean masks (K = 3 with multimask, else 1) at input resolution."""
        rgb = validate_image(image)
        clean = validate_prompts(rgb.width, rgb.height, points, point_labels, box)
        clean_points, clean_labels, clean_box = clean
        if not isinstance(multimask, bool):
            raise TypeError("multimask must be a bool")
        masks, iou_scores = self._runner(rgb, clean_points, clean_labels, clean_box, multimask)
        masks = np.asarray(masks)
        expected = (NUM_MULTIMASK_OUTPUTS if multimask else 1, rgb.height, rgb.width)
        if masks.dtype != np.bool_ or masks.shape != expected or len(iou_scores) != expected[0]:
            raise RuntimeError(f"backend returned {masks.shape} {masks.dtype}, {len(iou_scores)} scores")
        return {
            "masks": masks,
            "iou_scores": [float(v) for v in iou_scores],
            "multimask": multimask,
            "points": clean_points,
            "point_labels": clean_labels,
            "box": clean_box,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
