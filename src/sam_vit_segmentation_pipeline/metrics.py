"""Corpus-level prompt-segmentation measures and two non-neural baselines, in numpy.

``segmentation_metrics`` scores one predicted boolean mask per record against ``record['mask']``: mean IoU
(``pipeline.mask_iou``) and the fraction of records with IoU at least 0.5 (``hit_rate``), overall and per
``category``. The baselines answer from the prompts alone — the filled box, or a disk around the point with the
target's area — and are scored by the same function.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .pipeline import mask_iou

METRIC_DEFINITIONS = {
    "iou": "mean intersection-over-union between the predicted mask and the record's target mask; in 0..1, higher is better",
    "hit_rate": "fraction of records whose predicted mask reaches IoU >= 0.5 with the target; in 0..1",
}
HIT_THRESHOLD = 0.5


def segmentation_metrics(masks: Sequence[np.ndarray], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Score one boolean mask per record against `record['mask']`; per-record rows, means overall and per category.
    Raises when the lengths differ or nothing is scored."""
    if len(masks) != len(records) or not masks:
        raise ValueError("masks and records must be non-empty and the same length")
    rows = []
    for mask, record in zip(masks, records, strict=True):
        iou = mask_iou(np.asarray(mask, dtype=bool), np.asarray(record["mask"], dtype=bool))
        rows.append({"id": record["id"], "category": record.get("category"), "iou": iou, "hit": iou >= HIT_THRESHOLD})

    def _mean(items: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
        return {"n": len(items), "iou": float(np.mean([r["iou"] for r in items])), "hit_rate": float(np.mean([r["hit"] for r in items]))}

    categories = sorted({r["category"] for r in rows if r["category"] is not None})
    return {
        **_mean(rows),
        "per_category": {c: _mean([r for r in rows if r["category"] == c]) for c in categories},
        "per_record": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def box_fill_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The prompt as the answer: every pixel inside the box."""
    masks = []
    for record in records:
        target = np.asarray(record["mask"], dtype=bool)
        mask = np.zeros_like(target)
        x0, y0, x1, y1 = (int(round(v)) for v in record["box"])
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
        masks.append(mask)
    return {**segmentation_metrics(masks, records), "baseline": "the box prompt filled"}


def centre_disk_baseline(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """A disk around the point prompt with the target's own area — the size is given away, the shape is not."""
    masks = []
    for record in records:
        target = np.asarray(record["mask"], dtype=bool)
        radius = float(np.sqrt(target.sum() / np.pi))
        yy, xx = np.ogrid[: target.shape[0], : target.shape[1]]
        x, y = record["point"]
        masks.append((yy - y) ** 2 + (xx - x) ** 2 <= radius**2)
    return {**segmentation_metrics(masks, records), "baseline": "a disk around the point prompt with the target's area"}
