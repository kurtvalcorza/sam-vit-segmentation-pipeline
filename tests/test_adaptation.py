"""Offline checks of the adaptation contract: the prompt-segmentation record contract and its refusals, target
selection and prompt derivation, the pinned-corpus refusals and the draw, splitting, the BYOD loader, the metrics
and baselines, `evaluate` through the injected runner, the artifact-manifest checks, and the model-free refusals of
`adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image, ImageDraw

from sam_vit_segmentation_pipeline import (
    ADE_CLASSES,
    ARTIFACT_FORMAT,
    MODEL_ID,
    MODEL_REVISION,
    ROW_GROUP_PINS,
    SAMPLE_SPLIT,
    SAMViTSegmentationPipeline,
    box_fill_baseline,
    build_sample_dataset,
    centre_disk_baseline,
    check_split_disjoint,
    dataset_digest,
    fetch_corpus,
    image_digest,
    interior_point,
    load_byod_dataset,
    mask_box,
    read_corpus,
    segmentation_metrics,
    select_target,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from sam_vit_segmentation_pipeline import pipeline as pl
from sam_vit_segmentation_pipeline import samples as sm

COLOURS = ["red", "green", "blue", "yellow", "white", "black"]


def _scene(i, size=96):
    image = Image.new("RGB", (size, size), (135, 206, 235))
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, size * 2 // 3, size, size], fill=(60, 179, 75))
    draw.ellipse([size // 4, size // 6, size * 3 // 4, size * 2 // 3], fill=COLOURS[i % 6])
    image.putpixel((i % size, 0), (i % 256, 0, 0))
    return image


def _mask(size=96):
    mask = np.zeros((size, size), dtype=bool)
    yy, xx = np.ogrid[:size, :size]
    cx, cy, r = size // 2, size * 5 // 12, size // 4
    mask[(yy - cy) ** 2 + (xx - cx) ** 2 <= r**2] = True
    return mask


def _record(i, size=96, category=None):
    mask = _mask(size)
    return {"id": f"r{i:03d}", "image": _scene(i, size), "mask": mask, "point": interior_point(mask), "box": mask_box(mask), "category": category or COLOURS[i % 6]}


def _records(n=12):
    return [_record(i) for i in range(n)]


class _ScriptedRunner:
    """Returns the box filled as every mask (records the calls)."""

    def __init__(self):
        self.calls = 0

    def __call__(self, image, points, labels, box, multimask):
        self.calls += 1
        k = 3 if multimask else 1
        mask = np.zeros((image.height, image.width), dtype=bool)
        if box is not None:
            x0, y0, x1, y1 = (int(round(v)) for v in box)
            mask[y0 : y1 + 1, x0 : x1 + 1] = True
        else:
            x, y = points[0]
            mask[max(0, int(y) - 20) : int(y) + 20, max(0, int(x) - 20) : int(x) + 20] = True
        return np.stack([mask] * k), [0.9] * k


# --- targets and record contract ---------------------------------------------------------------------------------


def test_select_target_picks_the_largest_qualifying_component():
    ann = np.zeros((100, 100), dtype=np.uint8)
    ann[:, :] = 1  # a wall covering everything: too large
    ann[10:30, 10:30] = 2  # 4 %: qualifies
    ann[50:60, 50:60] = 3  # 1 %: too small
    ann[70:90, 60:95] = 2  # 7 %, same class, larger component
    mask, cls = select_target(ann)
    assert cls == 2 and mask.sum() == 20 * 35 and not mask[10, 10]
    assert select_target(np.ones((50, 50), dtype=np.uint8)) is None
    point = interior_point(mask)
    assert mask[int(point[1]), int(point[0])]
    assert mask_box(mask) == [60.0, 70.0, 94.0, 89.0]
    with pytest.raises(ValueError, match="empty"):
        interior_point(np.zeros((4, 4), dtype=bool))


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    manifest = validate_dataset(_records())
    assert manifest["n_records"] == 12 and manifest["category_counts"] == {c: 2 for c in COLOURS}
    assert manifest["image_side"] == {"min": 96, "max": 96} and 0.19 < manifest["mask_fraction"]["min"] < 0.21
    assert len(manifest["digest"]) == 64 and manifest["model_id"] == MODEL_ID


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.__setitem__("id", "bad id"), "id must match"),
        (lambda r: r.__setitem__("mask", _mask(64)), "does not match image"),
        (lambda r: r.__setitem__("mask", np.zeros((96, 96), dtype=bool)), "mask is empty"),
        (lambda r: r.__setitem__("mask", np.zeros((96, 96), dtype=np.uint8)), "boolean array"),
        (lambda r: r.__setitem__("point", [1.0, 1.0]), "not inside the mask"),
        (lambda r: r.__setitem__("point", [999.0, 1.0]), "outside image"),
        (lambda r: r.__setitem__("box", [40.0, 40.0, 50.0, 50.0]), "does not enclose"),
        (lambda r: r.__setitem__("image", Image.new("RGB", (8, 8))), "MIN_IMAGE_SIDE"),
        (lambda r: r.__setitem__("category", "x" * 40), "category must be"),
        (lambda r: r.pop("box"), "missing 'box'"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[0])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_and_unique_ids():
    with pytest.raises(ValueError, match="8..5000"):
        validate_dataset(_records(4))
    records = _records()
    records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(records)
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError):
        validate_dataset(_records(3))
    validate_dataset(_records())


def test_digests_and_split_disjointness():
    records = _records(24)
    assert dataset_digest(records) == dataset_digest(list(reversed(records)))
    assert image_digest(records[0]["image"]) != image_digest(records[1]["image"])
    splits = split_dataset(records, seed=1)
    assert sum(len(v) for v in splits.values()) == 24 and all(splits.values())
    assert check_split_disjoint(splits) == {k: len(v) for k, v in splits.items()}
    leaked = {**splits, "test": [*splits["test"], splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaked)
    duplicated = [*records, {**records[0], "id": "copy"}]
    assert sum(len(v) for v in split_dataset(duplicated, seed=1).values()) == 24
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.9)


# --- pinned corpus ----------------------------------------------------------------------------------------------


def test_pins_and_class_table():
    assert sorted(ROW_GROUP_PINS) == [0, 1, 2] and all(len(v[0]) == 64 for v in ROW_GROUP_PINS.values())
    assert len(ADE_CLASSES) == 150 and ADE_CLASSES[1] == "wall" and ADE_CLASSES[150] == "flag"
    assert sum(SAMPLE_SPLIT.values()) == 295


def _fake_group(n=8, size=200):
    rows = []
    for i in range(n):
        image = io.BytesIO()
        _scene(i, size).save(image, format="JPEG")
        ann = np.ones((size, size), dtype=np.uint8) * 3  # sky everywhere ...
        ann[40:120, 40:140] = 8  # ... with a bed covering 20 %
        annotation = io.BytesIO()
        Image.fromarray(ann).save(annotation, format="PNG")
        rows.append({"image": image.getvalue(), "annotation": annotation.getvalue()})
    return rows


def test_read_corpus_builds_one_target_per_qualifying_image():
    records = read_corpus({0: _fake_group()})
    assert len(records) == 8 and records[0]["category"] == "bed" and records[0]["ade_class_id"] == 8
    assert records[0]["mask"].sum() == 80 * 100 and records[0]["id"] == "ade-val-0"
    manifest = validate_dataset(records)
    assert manifest["category_counts"] == {"bed": 8}
    splits = build_sample_dataset(records, sizes={"train": 4, "validation": 2, "test": 2})
    assert {k: len(v) for k, v in splits.items()} == {"train": 4, "validation": 2, "test": 2}
    assert build_sample_dataset(records, sizes={"train": 4, "validation": 2, "test": 2}) == splits
    with pytest.raises(ValueError, match="need"):
        build_sample_dataset(records)


def test_fetch_corpus_refuses_a_row_group_that_does_not_match_its_pin(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    rows = _fake_group()
    table = pa.table({"image": [{"bytes": r["image"], "path": ""} for r in rows], "annotation": [{"bytes": r["annotation"], "path": ""} for r in rows]})
    path = tmp_path / "fake.parquet"
    pq.write_table(table, path)
    opener = lambda url: io.BytesIO(path.read_bytes())  # noqa: E731
    with pytest.raises(ValueError, match="rows, pinned"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[0], opener=opener)
    with pytest.raises(ValueError, match="no pin"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[7], opener=opener)
    assert not (tmp_path / "cache" / "validation-rg0.parquet").exists()


def test_default_draw_matches_the_pinned_digest_when_the_row_groups_are_cached():
    cached = sm.DEFAULT_CACHE_DIR
    if not all((cached / f"validation-rg{g}.parquet").is_file() for g in ROW_GROUP_PINS):
        pytest.skip("ADE20K row groups not cached")
    splits = build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cached)))
    assert {k: len(v) for k, v in splits.items()} == SAMPLE_SPLIT
    assert check_split_disjoint(splits)
    assert dataset_digest([r for part in splits.values() for r in part]) == sm.SAMPLE_DIGEST


# --- BYOD ---------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_images_with_masks_from_a_zip_or_directory(tmp_path):
    zip_path = tmp_path / "photos.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for i in range(9):
            buffer = io.BytesIO()
            _scene(i, 120).save(buffer, format="PNG")
            archive.writestr(f"photo{i}.png", buffer.getvalue())
            mask = io.BytesIO()
            Image.fromarray((_mask(120) * 255).astype(np.uint8)).save(mask, format="PNG")
            archive.writestr(f"photo{i}_mask.png", mask.getvalue())
        archive.writestr("labels.csv", "id,file,category\n" + "".join(f"p{i},photo{i}.png,{COLOURS[i % 6]}\n" for i in range(9)))
    records = load_byod_dataset(zip_path)
    assert len(records) == 9 and {r["id"] for r in records} == {f"p{i}" for i in range(9)}
    assert validate_dataset(records)["n_records"] == 9
    folder = tmp_path / "folder"
    folder.mkdir()
    for i in range(8):
        _scene(i, 120).save(folder / f"img{i}.png")
        Image.fromarray((_mask(120) * 255).astype(np.uint8)).save(folder / f"img{i}_mask.png")
    assert len(load_byod_dataset(folder)) == 8
    _scene(9, 120).save(folder / "lonely.png")
    with pytest.raises(ValueError, match="has no lonely_mask"):
        load_byod_dataset(folder)
    with pytest.raises(ValueError, match="neither"):
        load_byod_dataset(tmp_path / "nothing")
    csv_path = write_dataset_csv(records, tmp_path / "train.csv")
    assert csv_path.read_text(encoding="utf-8").startswith("id,category,width,height,mask_fraction")


# --- metrics, baselines, evaluate ----------------------------------------------------------------------------------


def test_segmentation_metrics_and_baselines():
    records = _records()
    perfect = segmentation_metrics([r["mask"] for r in records], records)
    assert perfect["iou"] == 1.0 and perfect["hit_rate"] == 1.0 and set(perfect["per_category"]) == set(COLOURS)
    empty = segmentation_metrics([np.zeros_like(r["mask"]) for r in records], records)
    assert empty["iou"] == 0.0 and empty["hit_rate"] == 0.0
    box = box_fill_baseline(records)
    disk = centre_disk_baseline(records)
    assert 0.7 < box["iou"] < 0.85 and disk["iou"] > 0.9  # the target is a disk; the box over-covers it
    with pytest.raises(ValueError, match="same length"):
        segmentation_metrics([records[0]["mask"]], records)


def test_evaluate_runs_every_record_through_segment_and_scores_it():
    runner = _ScriptedRunner()
    pipe = SAMViTSegmentationPipeline(runner, "cpu")
    result = pipe.evaluate(_records(), prompt="box")
    assert runner.calls == 12 and result["n"] == 12 and result["adapted"] is False and result["prompt"] == "box"
    assert result["verdict"] == "measured-small-sample" and result["iou"] == pytest.approx(box_fill_baseline(_records())["iou"])
    point = pipe.evaluate(_records(), prompt="point")
    assert 0.0 < point["iou"] < 1.0
    with pytest.raises(ValueError, match="prompt must be"):
        pipe.evaluate(_records(), prompt="scribble")
    with pytest.raises(ValueError):
        pipe.evaluate([{"id": "x"}] * 8)


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = SAMViTSegmentationPipeline(_ScriptedRunner(), "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records())
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "adapter": {"prompts": "mixed"},
        "tensors": ["mask_decoder.transformer.layers.0.self_attn.q_proj.weight", "mask_decoder.iou_prediction_head.proj_out.bias"],
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="mask decoder"):
        pl._check_artifact_manifest(_manifest(tmp_path, tensors=["vision_encoder.patch_embed.projection.weight"]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="prompts"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"prompts": "scribble"}), tmp_path, "base-digest")


def test_trainable_names_selects_the_mask_decoder_only():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["vision_encoder.patch_embed.w", "prompt_encoder.point_embed.w", "mask_decoder.transformer.w", "mask_decoder.iou_prediction_head.w"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model()) == ["mask_decoder.transformer.w", "mask_decoder.iou_prediction_head.w"]


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.3, "val": {"iou": 0.7, "hit_rate": 0.8, "n": 45}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["iou"] == 0.7
