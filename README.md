# SAM ViT-B promptable segmentation pipeline

DIMER inference wrapper for **Segment Anything (SAM v1) ViT-B** (`facebook/sam-vit-base`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline segments one object per call from point clicks and/or a box and returns boolean masks at the input resolution with the model's predicted IoU per mask, and carries a bounded fine-tuning contract for the mask decoder on labelled point/box → mask records. Prompted mode only: automatic (grid-prompt) mask generation is not exposed. The SAM 2.1 sibling lives in `sam2-segmentation-pipeline` and shares this contract.

## Upstream alignment

- Model: `facebook/sam-vit-base`
- Revision: `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`
- Upstream weight license: Apache-2.0
- Upstream task: promptable image segmentation (points / boxes → masks)
- Repository adaptation: a bounded fine-tuning contract (`evaluate`, `adapt`, `save_artifact`, `from_artifact`) over the mask decoder with a BCE + soft-Dice loss under point, box or mixed prompts; the image and prompt encoders stay frozen and the base weights are never modified on disk

## Quick start

```python
import numpy as np
from PIL import Image
from sam_vit_segmentation_pipeline import SAMViTSegmentationPipeline, mask_iou

pipe = SAMViTSegmentationPipeline.from_pretrained()    # stages + verifies weights/sam-vit-base first
image = Image.open("photo.jpg")

# one foreground click -> three candidate masks, pick the one the model rates highest
result = pipe.segment(image, points=[(500, 375)], point_labels=[1])
best = result["masks"][int(np.argmax(result["iou_scores"]))]   # bool H x W

# a box (e.g. from grounding-dino-detection-pipeline) -> one mask
result = pipe.segment(image, box=[75, 275, 1725, 850], multimask=False)
mask = result["masks"][0]

# optional: score against a caller-supplied reference mask
# print(mask_iou(mask, reference_bool_mask))
```

Install into a Python 3.12 environment that already holds the pinned dependencies (including `scipy` and `pyarrow` for the sample corpus) with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the test suite (`tests/test_model_backed.py` runs only where the snapshot is staged and skips otherwise). On a fresh clone the manifest is committed but the weights are not: `SAMViTSegmentationPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/sam-vit-base/
  dimer-base-manifest.json      # modelId, revision, per-file bytes + SHA-256 (4 files, 374,993,239 bytes)
  config.json                   # SamModel: ViT-B encoder, prompt encoder, mask decoder
  preprocessor_config.json      # longest edge 1024, pad to 1024x1024, ImageNet mean/std
  model.safetensors             # git-ignored, 374,979,480 bytes
  README.md
```

## Adaptation contract

```python
from sam_vit_segmentation_pipeline import (
    SAMViTSegmentationPipeline, box_fill_baseline, build_sample_dataset, fetch_corpus, load_byod_dataset, read_corpus, split_dataset,
)

splits = build_sample_dataset(read_corpus(fetch_corpus()), seed=42)   # 296 ADE20K targets, 180 / 45 / 70
# or: splits = split_dataset(load_byod_dataset("my_masks.zip"), seed=42)  # your images + <stem>_mask.png each

pipe = SAMViTSegmentationPipeline.from_pretrained()                    # cuda:0 if available, else cpu
box_fill = box_fill_baseline(splits["test"])                           # iou, hit_rate, per_category
frozen_point = pipe.evaluate(splits["test"], prompt="point")
frozen_box = pipe.evaluate(splits["test"], prompt="box")
result = pipe.adapt(splits["train"], splits["validation"], epochs=6, lr=5e-5, prompts="mixed")
adapted_point = pipe.evaluate(splits["test"], prompt="point")
pipe.save_artifact("outputs/adapter")                                  # adapter.safetensors + manifest.json
again = SAMViTSegmentationPipeline.from_artifact("outputs/adapter")
```

- Records are `{id, image, mask, point, box, category?}`: `image` a PIL image within the ceilings, `mask` a boolean array of the same height and width (or a mask image, white = target), `point` one `[x, y]` click inside the mask, `box` the `[x0, y0, x1, y1]` box enclosing it; `validate_dataset` checks the shape (8..5,000 records, unique ids, click inside the mask, box enclosing it) before any model import, and `split_dataset` de-duplicates by decoded pixels; `check_split_disjoint` asserts no image is shared. `interior_point` (distance-transform maximum) and `mask_box` derive the two prompts from a mask, which is how the BYOD loader fills them.
- The default sample (`samples.py`) is the first three row groups of the ADE20K scene-parsing validation parquet shard (`zhoubolei/scene_parse_150` at the immutable parquet-conversion commit `e660d866a1351c70bcf07925d2600b60bd6e3bc3`; BSD-3-Clause), read with HTTPS range requests — the shard's declared size is checked and each row group's decoded SHA-256 and byte total are pinned in `ROW_GROUP_PINS` — and cached git-ignored under `weights/ade20k/`. `select_target` keeps the largest 4-connected component of one class covering 3..35 % of each image (296 targets from 300 images, about 55 classes, mostly *stuff*: wall, sky, floor, road, ceiling); `build_sample_dataset` draws a seeded 180 / 45 / 70 image-level split (`SAMPLE_DIGEST` pins the draw).
- `evaluate(records, *, prompt="point"|"box")` runs `predict_mask` on every record — `segment` with the record's click (label 1) or box, keeping the candidate the model scores highest — and returns `segmentation_metrics` (`metrics.py`): mean IoU and hit rate (IoU ≥ 0.5), overall and per category, plus `per_record`, `verdict` (`measured` / `measured-small-sample`) and `adapted`. `box_fill_baseline` (the box as the answer) and `centre_disk_baseline` (a disk around the click with the target's area) are the two prompt-only references the tutorial scores beside the model.
- `adapt(train, val=None, *, epochs=6, lr=5e-5, prompts="mixed", seed=0, progress=None)` trains only the mask decoder (4,058,340 of 93,735,472 parameters): every training image is embedded once by the frozen image encoder, each step encodes the record's click or box (`prompts`: `point`, `box`, or a seeded coin per record and epoch for `mixed`) and trains the single-mask logits against the target in the 256×256 low-resolution frame with binary cross-entropy plus a soft Dice term; AdamW without weight decay, gradient clipping at 1.0, seeded shuffling; epoch 0 records the frozen model and the epoch with the highest mean of the validation point- and box-prompt IoU is kept. The update is transactional: an exception restores the frozen weights.
- `save_artifact(dir)` writes the trained tensors as `adapter.safetensors` plus a `manifest.json` (format `org.valcorza.sam-vit-base.adapter.v1`: base id, revision and weight digest, tensor names, file size and SHA-256, training configuration, epoch history); `from_artifact(dir)` re-verifies the base snapshot, checks the manifest, the digest and the exact tensor set before deserialising, refuses any tensor outside the mask decoder, and overlays the tensors onto a freshly loaded base.

## Input ceilings

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16` points; one object per call; masks are binarised at logit `MASK_THRESHOLD = 0.0`. `iou_scores` is the model's own uncalibrated estimate and can exceed 1.0. See `MODEL_CARD.md` for the measured timings, the candidate-selection rule and the adaptation build record.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/sam-vit-segmentation-pipeline/blob/main/tutorials/sam_vit_segmentation_colab.ipynb)

`tutorials/sam_vit_segmentation_colab.ipynb` is declared `E2E` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the package's three modules (`pipeline.py`, `metrics.py`, `samples.py`), the model identity, the 4-file manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py` and `tools/validate_release_assets.py`; see `tutorials/README.md`). It stages and digest-verifies the snapshot, fetches the three digest-pinned ADE20K row groups and turns them into 296 prompt-to-mask targets under a stated rule (largest component of one class, 3..35 % of the image; interior click and tight box), splits them by image without leakage, segments a synthetic scene through the inference contract with an input manifest and a rejection probe, scores the frozen model's point- and box-prompt IoU on the 70 held-out targets beside the box-fill and centre-disk baselines (the frozen point prompt is below box-fill on these targets), runs a bounded fine-tuning of the mask decoder with the BCE + soft-Dice loss under mixed prompts and epoch selection on the mean validation point- and box-prompt IoU, re-scores the held-out split per class on both prompts, re-segments the scene and four targets as side-by-side panels, exports the adapter and reloads it with verified mask parity, and writes `sam_vit_segmentation_train.csv`, `sam_vit_segmentation_input_manifest.json`, `sam_vit_segmentation_mask_frozen.png`, `sam_vit_segmentation_mask_adapted.png`, `sam_vit_segmentation_evaluation_report.json`, `sam_vit_segmentation_examples/`, `sam_vit_segmentation_adapter/` and `sam_vit_segmentation_result.json` under `outputs/`.

The default path runs on CPU and uses CUDA automatically when present (about seven minutes on an RTX 5070 Ti after the downloads; the image encoder costs several seconds per image on CPU, so expect an hour or more on a 2-vCPU hosted runtime). The metrics it prints are one seeded split of one 296-target sample under one dataset's labelling convention — evidence that the adaptation contract works, not a segmentation benchmark or production-fitness evidence.

## Release status

**Release-grade** — the `E2E` notebook blob `0112e790` (committed at `4ef608f`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 912.1 s); the record is in `docs/release-verification.md` and `STATUS.md`. Static and unit checks — including the standalone generator parity checks — are necessary but were never the evidence; the hosted run is. A later change to the carried modules or the notebook returns the status to Candidate until re-verified.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, adaptation contract and build record, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
