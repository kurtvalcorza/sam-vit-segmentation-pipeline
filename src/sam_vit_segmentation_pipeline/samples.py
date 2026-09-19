"""Labelled prompt-segmentation datasets for the adaptation contract: the digest-pinned ADE20K sample, the record
contract and its structural validation, image-disjoint splitting, and the BYOD loader.

A record is ``{id, image, mask, point, box}`` where ``image`` is a PIL image (sides within the pipeline's ceilings),
``mask`` the boolean target (same height and width), ``point`` one ``[x, y]`` foreground click inside the mask and
``box`` the ``[x0, y0, x1, y1]`` box around it — the two prompts the model is asked to turn into that mask. An
optional ``category`` (free text, at most 32 characters) is carried into the per-category breakdown.

The default sample is drawn from the ADE20K scene-parsing validation set (Zhou et al. 2017/2019, **BSD-3-Clause**)
as converted to parquet by the Hugging Face Hub at an immutable revision: the first ``CORPUS_ROW_GROUPS`` row groups
of the validation shard are read with HTTPS range requests (about 4.8 MB each; the shard's declared size is checked
first and every row group's decoded content is refused unless its SHA-256 matches the pin), and each image yields
one target — the largest connected component of one semantic class covering 3..35 % of the image — with an interior
point (the distance-transform maximum) and the tight box. The class name is the record's ``category``.
"""
# ruff: noqa: E501  -- record and pin literals are kept on single lines

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import re
import urllib.request
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .pipeline import MODEL_ID, validate_image, validate_prompts

CORPUS_NAME = "ADE20K scene parsing (validation), first three parquet row groups"
CORPUS_REPO = "zhoubolei/scene_parse_150"
CORPUS_REVISION = "e660d866a1351c70bcf07925d2600b60bd6e3bc3"  # refs/convert/parquet commit on the Hub
CORPUS_FILE = "scene_parsing/validation/0000.parquet"
CORPUS_SHA256 = "79742deaf2661c4740a75d7c26dd361b50209240f6f25aecd29883b96099f879"  # the whole shard (LFS oid)
CORPUS_BYTES = 89_146_778
CORPUS_ROWS = 2_000
CORPUS_ROW_GROUPS = 3  # of 20; 100 images each
CORPUS_LICENSE = "BSD-3-Clause (MIT CSAIL scene parsing benchmark, Zhou et al. 2017; https://github.com/CSAILVision/sceneparsing)"
CORPUS_URL = f"https://huggingface.co/datasets/{CORPUS_REPO}/resolve/{CORPUS_REVISION}/{CORPUS_FILE}"
# SHA-256 over the concatenated image + annotation bytes of each row group, in row order.
ROW_GROUP_PINS: dict[int, tuple[str, int]] = {
    0: ("6a056583387d31c659ef2ae0f10eed4184776ec3f70ec9250e9fb2ecf809e4f0", 4_600_209),
    1: ("020c24dc20b9bf6acbc2092cf94d7db8dc3e21516ec60e02591dd051ead63539", 5_007_242),
    2: ("07c4516da0c55b6d9cceecc534a18930337c52b5dc38e8a8dd559f39136d4c3c", 4_540_332),
}
DEFAULT_CACHE_DIR = Path("weights") / "ade20k"

TARGET_MIN_FRACTION = 0.03  # of the image area
TARGET_MAX_FRACTION = 0.35
SAMPLE_MAX_SIDE = 1024
SAMPLE_MIN_SIDE = 128
SAMPLE_SEED = 42
SAMPLE_SPLIT = {"train": 180, "validation": 45, "test": 70}  # of the ~296 records the three row groups yield
SAMPLE_DIGEST = "509007cac0b005f76294dd93c107ff3b8310b8a7a9ec4b69511641e686b223d0"  # dataset_digest over the three default splits together; tests pin it
MIN_RECORDS = 8
MAX_RECORDS = 5_000
MAX_CATEGORY_CHARS = 32
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

# objectInfo150.csv (CSAILVision/sceneparsing), first name of each row; index 0 is "other objects".
ADE_CLASSES: dict[int, str] = {
    1: "wall", 2: "building", 3: "sky", 4: "floor", 5: "tree", 6: "ceiling", 7: "road", 8: "bed", 9: "windowpane", 10: "grass",
    11: "cabinet", 12: "sidewalk", 13: "person", 14: "earth", 15: "door", 16: "table", 17: "mountain", 18: "plant", 19: "curtain", 20: "chair",
    21: "car", 22: "water", 23: "painting", 24: "sofa", 25: "shelf", 26: "house", 27: "sea", 28: "mirror", 29: "rug", 30: "field",
    31: "armchair", 32: "seat", 33: "fence", 34: "desk", 35: "rock", 36: "wardrobe", 37: "lamp", 38: "bathtub", 39: "railing", 40: "cushion",
    41: "base", 42: "box", 43: "column", 44: "signboard", 45: "chest of drawers", 46: "counter", 47: "sand", 48: "sink", 49: "skyscraper", 50: "fireplace",
    51: "refrigerator", 52: "grandstand", 53: "path", 54: "stairs", 55: "runway", 56: "case", 57: "pool table", 58: "pillow", 59: "screen door", 60: "stairway",
    61: "river", 62: "bridge", 63: "bookcase", 64: "blind", 65: "coffee table", 66: "toilet", 67: "flower", 68: "book", 69: "hill", 70: "bench",
    71: "countertop", 72: "stove", 73: "palm", 74: "kitchen island", 75: "computer", 76: "swivel chair", 77: "boat", 78: "bar", 79: "arcade machine", 80: "hovel",
    81: "bus", 82: "towel", 83: "light", 84: "truck", 85: "tower", 86: "chandelier", 87: "awning", 88: "streetlight", 89: "booth", 90: "television receiver",
    91: "airplane", 92: "dirt track", 93: "apparel", 94: "pole", 95: "land", 96: "bannister", 97: "escalator", 98: "ottoman", 99: "bottle", 100: "buffet",
    101: "poster", 102: "stage", 103: "van", 104: "ship", 105: "fountain", 106: "conveyer belt", 107: "canopy", 108: "washer", 109: "plaything", 110: "swimming pool",
    111: "stool", 112: "barrel", 113: "basket", 114: "waterfall", 115: "tent", 116: "bag", 117: "minibike", 118: "cradle", 119: "oven", 120: "ball",
    121: "food", 122: "step", 123: "tank", 124: "trade name", 125: "microwave", 126: "pot", 127: "animal", 128: "bicycle", 129: "lake", 130: "dishwasher",
    131: "screen", 132: "blanket", 133: "sculpture", 134: "hood", 135: "sconce", 136: "vase", 137: "traffic light", 138: "tray", 139: "ashcan", 140: "fan",
    141: "pier", 142: "crt screen", 143: "plate", 144: "monitor", 145: "bulletin board", 146: "shower", 147: "radiator", 148: "glass", 149: "clock", 150: "flag",
}


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _HttpRangeFile(io.RawIOBase):
    """A seekable read-only view of one HTTPS object served with `Range` requests (what `pyarrow` needs to read a
    parquet footer and a few row groups without downloading the file)."""

    def __init__(self, url: str, size: int) -> None:
        self.url, self.size, self.pos = url, size, 0
        self.fetched = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, offset: int, whence: int = 0) -> int:
        base = {0: 0, 1: self.pos, 2: self.size}[whence]
        self.pos = max(0, base + offset)
        return self.pos

    def read(self, n: int = -1) -> bytes:
        if n is None or n < 0:
            n = self.size - self.pos
        if n <= 0 or self.pos >= self.size:
            return b""
        end = min(self.size, self.pos + n) - 1
        request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}", "User-Agent": "sam-vit-segmentation-pipeline"})
        with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310 (pinned https URL)
            if response.status != 206:
                raise ValueError(f"{self.url}: server ignored the Range request (HTTP {response.status})")
            data = response.read()
        self.fetched += len(data)
        self.pos += len(data)
        return data

    def readinto(self, buffer: Any) -> int:
        data = self.read(len(buffer))
        buffer[: len(data)] = data
        return len(data)


def _declared_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "sam-vit-segmentation-pipeline"})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 (pinned https URL)
        length = response.headers.get("Content-Length")
    if length is None:
        raise ValueError(f"{url}: no Content-Length in the HEAD response")
    return int(length)


def _group_digest(rows: Sequence[Mapping[str, Any]]) -> tuple[str, int]:
    digest, total = hashlib.sha256(), 0
    for row in rows:
        for key in ("image", "annotation"):
            data = row[key]["bytes"]
            digest.update(data)
            total += len(data)
    return digest.hexdigest(), total


def fetch_corpus(
    *, cache_dir: str | Path | None = None, groups: Sequence[int] | None = None, opener: Any = None
) -> dict[int, list[dict[str, bytes]]]:
    """Return the pinned row groups as lists of `{image, annotation}` byte pairs, from the cache (one parquet file per
    row group) or the Hub (footer + the row groups it needs, over range requests). Every row group's decoded content is
    refused unless its SHA-256 and byte total match `ROW_GROUP_PINS`; a fresh fetch also checks the shard's declared size."""
    import pyarrow.parquet as pq

    cache = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
    cache.mkdir(parents=True, exist_ok=True)
    wanted = list(groups) if groups is not None else sorted(ROW_GROUP_PINS)
    out: dict[int, list[dict[str, bytes]]] = {}
    reader = None
    for group in wanted:
        if group not in ROW_GROUP_PINS:
            raise ValueError(f"row group {group} has no pin; pinned groups are {sorted(ROW_GROUP_PINS)}")
        local = cache / f"validation-rg{group}.parquet"
        rows: list[dict[str, Any]] | None = None
        if local.is_file():
            rows = pq.read_table(local).to_pylist()
            if _group_digest(rows) != ROW_GROUP_PINS[group]:
                rows = None  # stale or corrupt cache: refetch
        if rows is None:
            if reader is None:
                if opener is not None:
                    reader = pq.ParquetFile(opener(CORPUS_URL))
                else:
                    declared = _declared_size(CORPUS_URL)
                    if declared != CORPUS_BYTES:
                        raise ValueError(f"{CORPUS_FILE}: declared size {declared} != pinned {CORPUS_BYTES}")
                    reader = pq.ParquetFile(_HttpRangeFile(CORPUS_URL, CORPUS_BYTES))
                if reader.metadata.num_rows != CORPUS_ROWS:
                    raise ValueError(f"{CORPUS_FILE}: {reader.metadata.num_rows} rows, pinned {CORPUS_ROWS}")
            table = reader.read_row_group(group, columns=["image", "annotation"])
            rows = table.to_pylist()
            digest, total = _group_digest(rows)
            if (digest, total) != ROW_GROUP_PINS[group]:
                raise ValueError(f"{CORPUS_FILE} row group {group}: sha256 {digest} / {total} bytes != pinned {ROW_GROUP_PINS[group]}")
            pq.write_table(table, local)
        out[group] = [{"image": r["image"]["bytes"], "annotation": r["annotation"]["bytes"]} for r in rows]
    return out


# ---------------------------------------------------------------------------------------------------------
# Targets
# ---------------------------------------------------------------------------------------------------------


def connected_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels of a boolean mask (0 = background, 1..n = components)."""
    from scipy import ndimage

    labels, n = ndimage.label(np.asarray(mask, dtype=bool))
    return labels, int(n)


def interior_point(mask: np.ndarray) -> list[float]:
    """The `[x, y]` pixel of a boolean mask farthest from its boundary (the distance-transform maximum)."""
    from scipy import ndimage

    arr = np.asarray(mask, dtype=bool)
    if not arr.any():
        raise ValueError("mask is empty")
    distance = ndimage.distance_transform_edt(arr)
    y, x = np.unravel_index(int(distance.argmax()), distance.shape)
    return [float(x), float(y)]


def mask_box(mask: np.ndarray) -> list[float]:
    """The tight `[x0, y0, x1, y1]` box around a boolean mask."""
    ys, xs = np.where(np.asarray(mask, dtype=bool))
    if ys.size == 0:
        raise ValueError("mask is empty")
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def select_target(annotation: np.ndarray, *, min_fraction: float = TARGET_MIN_FRACTION, max_fraction: float = TARGET_MAX_FRACTION) -> tuple[np.ndarray, int] | None:
    """The largest 4-connected component of any non-background class whose area is within the fractions, as
    `(mask, class_id)`, or None when no component qualifies."""
    ann = np.asarray(annotation)
    area = ann.size
    best: tuple[np.ndarray, int, int] | None = None
    for cls in np.unique(ann):
        if int(cls) == 0:
            continue
        labels, n = connected_components(ann == cls)
        if n == 0:
            continue
        counts = np.bincount(labels.ravel())[1:]
        k = int(counts.argmax()) + 1
        size = int(counts[k - 1])
        if min_fraction * area <= size <= max_fraction * area and (best is None or size > best[2]):
            best = (labels == k, int(cls), size)
    return None if best is None else (best[0], best[1])


def read_corpus(groups: Mapping[int, Sequence[Mapping[str, bytes]]]) -> list[dict[str, Any]]:
    """Decode the verified row groups into records: one target per image that has a qualifying component."""
    out = []
    for group in sorted(groups):
        for index, row in enumerate(groups[group]):
            image = Image.open(io.BytesIO(row["image"]))
            image.load()
            if max(image.size) > SAMPLE_MAX_SIDE or min(image.size) < SAMPLE_MIN_SIDE:
                continue
            annotation = np.array(Image.open(io.BytesIO(row["annotation"])))
            if annotation.ndim != 2 or annotation.shape != (image.height, image.width):
                continue
            target = select_target(annotation)
            if target is None:
                continue
            mask, cls = target
            out.append(
                {
                    "id": f"ade-val-{group * 100 + index}",
                    "image": image.convert("RGB"),
                    "mask": mask,
                    "point": interior_point(mask),
                    "box": mask_box(mask),
                    "category": ADE_CLASSES.get(cls, f"class-{cls}"),
                    "ade_class_id": cls,
                    "source_row_group": group,
                }
            )
    return out


def build_sample_dataset(
    records: Sequence[Mapping[str, Any]], *, seed: int = SAMPLE_SEED, sizes: Mapping[str, int] | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Seeded image-level draw: shuffle the records and cut `sizes` (train / validation / test) in order."""
    sizes = dict(sizes or SAMPLE_SPLIT)
    pool = [dict(r) for r in records]
    random.Random(seed).shuffle(pool)
    needed = sum(sizes.values())
    if len(pool) < needed:
        raise ValueError(f"only {len(pool)} records available, need {needed}")
    out, cursor = {}, 0
    for name, count in sizes.items():
        out[name] = pool[cursor : cursor + count]
        cursor += count
    return out


def fetch_sample_dataset(*, cache_dir: str | Path | None = None, seed: int = SAMPLE_SEED) -> dict[str, list[dict[str, Any]]]:
    return build_sample_dataset(read_corpus(fetch_corpus(cache_dir=cache_dir)), seed=seed)


# ---------------------------------------------------------------------------------------------------------
# Record contract
# ---------------------------------------------------------------------------------------------------------


def _open(image: Any, where: str) -> Image.Image:
    if isinstance(image, str | Path):
        path = Path(image)
        if not path.is_file():
            raise ValueError(f"{where}: image file not found: {path}")
        image = Image.open(path)
        image.load()
    if not isinstance(image, Image.Image):
        raise ValueError(f"{where}: must be a PIL.Image.Image or a file path")
    return image


def _as_mask(mask: Any, where: str) -> np.ndarray:
    if isinstance(mask, str | Path):
        path = Path(mask)
        if not path.is_file():
            raise ValueError(f"{where}: mask file not found: {path}")
        mask = Image.open(path)
        mask.load()
    if isinstance(mask, Image.Image):
        arr = np.asarray(mask.convert("L")) > 127
    else:
        arr = np.asarray(mask)
        if arr.dtype != np.bool_:
            raise ValueError(f"{where}: mask must be a boolean array or a mask image")
    if arr.ndim != 2:
        raise ValueError(f"{where}: mask must be two-dimensional")
    if not arr.any():
        raise ValueError(f"{where}: mask is empty")
    return arr


def _check_record(record: Any, index: int) -> dict[str, Any]:
    where = f"records[{index}]"
    if not isinstance(record, Mapping):
        raise ValueError(f"{where} must be a mapping with id/image/mask/point/box")
    for key in ("id", "image", "mask", "point", "box"):
        if key not in record:
            raise ValueError(f"{where} is missing {key!r}")
    rid = record["id"]
    if not isinstance(rid, str) or not _ID_RE.match(rid):
        raise ValueError(f"{where}: id must match {_ID_RE.pattern}")
    try:
        image = validate_image(_open(record["image"], f"{where}.image"))
    except TypeError as exc:
        raise ValueError(f"{where}: {exc}") from exc
    mask = _as_mask(record["mask"], f"{where}.mask")
    if mask.shape != (image.height, image.width):
        raise ValueError(f"{where}: mask {mask.shape} does not match image {(image.height, image.width)}")
    try:
        points, _labels, box = validate_prompts(image.width, image.height, [record["point"]], [1], record["box"])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{where}: {exc}") from exc
    x, y = points[0]
    if not mask[min(int(round(y)), mask.shape[0] - 1), min(int(round(x)), mask.shape[1] - 1)]:
        raise ValueError(f"{where}: point {points[0]} is not inside the mask")
    ys, xs = np.where(mask)
    if not (box[0] <= xs.min() and box[1] <= ys.min() and box[2] >= xs.max() and box[3] >= ys.max()):
        raise ValueError(f"{where}: box {box} does not enclose the mask")
    item = {"id": rid, "image": image, "mask": mask, "point": [float(x), float(y)], "box": [float(v) for v in box]}
    category = record.get("category")
    if category is not None:
        if not isinstance(category, str) or not 1 <= len(category.strip()) <= MAX_CATEGORY_CHARS:
            raise ValueError(f"{where}: category must be a str of 1..{MAX_CATEGORY_CHARS} characters")
        item["category"] = category.strip()
    for key in ("ade_class_id", "source_row_group"):
        if key in record:
            item[key] = record[key]
    return item


def validate_dataset(
    records: Sequence[Mapping[str, Any]], *, min_records: int = MIN_RECORDS, max_records: int = MAX_RECORDS
) -> dict[str, Any]:
    """Structural validation of a prompt-segmentation dataset; raises ValueError before any model import."""
    if isinstance(records, Mapping) or not isinstance(records, Sequence) or isinstance(records, str | bytes):
        raise ValueError("records must be a list of {id, image, mask, point, box} mappings")
    if not min_records <= len(records) <= max_records:
        raise ValueError(f"{len(records)} records; {min_records}..{max_records} are required")
    checked, ids, categories = [], set(), {}
    for index, record in enumerate(records):
        item = _check_record(record, index)
        if item["id"] in ids:
            raise ValueError(f"duplicate id {item['id']!r}")
        ids.add(item["id"])
        if "category" in item:
            categories[item["category"]] = categories.get(item["category"], 0) + 1
        checked.append(item)
    fractions = [float(r["mask"].mean()) for r in checked]
    sides = [max(r["image"].size) for r in checked]
    return {
        "records": checked,
        "n_records": len(checked),
        "category_counts": dict(sorted(categories.items())),
        "image_side": {"min": min(sides), "max": max(sides)},
        "mask_fraction": {"min": round(min(fractions), 4), "max": round(max(fractions), 4)},
        "digest": dataset_digest(checked),
        "model_id": MODEL_ID,
    }


def image_digest(image: Image.Image) -> str:
    """SHA-256 of the decoded RGB pixels (size-prefixed) — the identity a split is made disjoint on."""
    rgb = image.convert("RGB")
    return _sha256_bytes(f"{rgb.width}x{rgb.height}:".encode() + rgb.tobytes())


def dataset_digest(records: Sequence[Mapping[str, Any]]) -> str:
    """Order-independent SHA-256 over (id, image digest, mask digest, point, box)."""
    parts = sorted(
        f"{r['id']}:{image_digest(r['image'])}:{_sha256_bytes(np.packbits(np.asarray(r['mask'], dtype=bool)).tobytes())}:{json.dumps(r['point'])}:{json.dumps(r['box'])}"
        for r in records
    )
    return _sha256_bytes("\n".join(parts).encode("utf-8"))


def check_split_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    """Assert no image (by decoded-pixel digest) appears in two splits (leakage check)."""
    seen: dict[str, str] = {}
    for name, records in splits.items():
        for record in records:
            key = image_digest(record["image"])
            if key in seen and seen[key] != name:
                raise ValueError(f"image {record['id']!r} appears in both {seen[key]} and {name}")
            seen[key] = name
    return {name: len(records) for name, records in splits.items()}


def split_dataset(
    records: Sequence[Mapping[str, Any]], *, val_fraction: float = 0.15, test_fraction: float = 0.2, seed: int = 0
) -> dict[str, list[dict[str, Any]]]:
    """Seeded shuffle of a BYOD dataset into train/validation/test after de-duplicating images."""
    if not (0.0 <= val_fraction < 1.0 and 0.0 < test_fraction < 1.0 and val_fraction + test_fraction < 1.0):
        raise ValueError("fractions must satisfy 0 <= val < 1, 0 < test < 1, val + test < 1")
    checked = validate_dataset(records)["records"]
    seen: set[str] = set()
    unique = []
    for record in checked:
        key = image_digest(record["image"])
        if key not in seen:
            seen.add(key)
            unique.append(record)
    random.Random(seed).shuffle(unique)
    n = len(unique)
    n_test = max(1, round(n * test_fraction))
    n_val = round(n * val_fraction)
    if n - n_test - n_val < 1:
        raise ValueError(f"{n} distinct images are too few to split into train/validation/test")
    return {"test": unique[:n_test], "validation": unique[n_test : n_test + n_val], "train": unique[n_test + n_val :]}


def load_byod_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Records from a directory or zip holding image files and, for each, a `<stem>_mask.png` (white = target)
    with an optional `labels.csv` (`id`, `file`, `category`); the point and box are derived from the mask."""
    source = Path(path)
    members: dict[str, bytes] = {}
    if source.is_dir():
        for file in sorted(source.rglob("*")):
            if file.is_file():
                members[file.name] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if not info.is_dir():
                    members[Path(info.filename).name] = archive.read(info)  # flattened; no extractall
    else:
        raise ValueError(f"{source} is neither a directory nor a zip file")
    labels: dict[str, tuple[str, str]] = {}
    if "labels.csv" in members:
        for row in csv.DictReader(io.StringIO(members["labels.csv"].decode("utf-8-sig"))):
            labels[str(row.get("file", "")).strip()] = (str(row.get("id", "")).strip(), str(row.get("category", "")).strip())
    out = []
    for name, data in members.items():
        stem = Path(name).stem
        if name == "labels.csv" or stem.endswith("_mask"):
            continue
        mask_name = next((m for m in members if Path(m).stem == f"{stem}_mask"), None)
        if mask_name is None:
            raise ValueError(f"BYOD image {name} has no {stem}_mask.* file")
        try:
            image = Image.open(io.BytesIO(data))
            image.load()
            mask = np.asarray(Image.open(io.BytesIO(members[mask_name])).convert("L")) > 127
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"BYOD file is not a decodable image: {name}") from exc
        if not mask.any():
            raise ValueError(f"BYOD mask {mask_name} is empty")
        rid, category = labels.get(name, ("", ""))
        record: dict[str, Any] = {
            "id": rid or re.sub(r"[^A-Za-z0-9_.:-]", "_", stem)[:64],
            "image": image.convert("RGB"),
            "mask": mask,
            "point": interior_point(mask),
            "box": mask_box(mask),
        }
        if category:
            record["category"] = category
        out.append(record)
    return out


def write_dataset_csv(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """A summary table (id, category, image size, mask fraction, point, box, provenance), not the BYOD format."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "category", "width", "height", "mask_fraction", "point_x", "point_y", "x0", "y0", "x1", "y1", "source_row_group"])
        for r in records:
            writer.writerow([r["id"], r.get("category", ""), r["image"].width, r["image"].height, round(float(np.asarray(r["mask"]).mean()), 4), *r["point"], *r["box"], r.get("source_row_group", "")])
    return out
