# SAM ViT-B promptable segmentation pipeline

DIMER inference wrapper for **Segment Anything (SAM v1) ViT-B** (`facebook/sam-vit-base`), pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline segments one object per call from point clicks and/or a box and returns boolean masks at the input resolution with the model's predicted IoU per mask. Prompted mode only: automatic (grid-prompt) mask generation is not exposed. The SAM 2.1 sibling lives in `sam2-segmentation-pipeline` and shares this contract.

## Upstream alignment

- Model: `facebook/sam-vit-base`
- Revision: `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`
- Upstream weight license: Apache-2.0
- Upstream task: promptable image segmentation (points / boxes → masks)
- Repository adaptation: **none**; inference only

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

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed). On a fresh clone the manifest is committed but the weights are not: `SAMViTSegmentationPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/sam-vit-base/
  dimer-base-manifest.json      # modelId, revision, per-file bytes + SHA-256 (4 files, 374,993,239 bytes)
  config.json                   # SamModel: ViT-B encoder, prompt encoder, mask decoder
  preprocessor_config.json      # longest edge 1024, pad to 1024x1024, ImageNet mean/std
  model.safetensors             # git-ignored, 374,979,480 bytes
  README.md
```

## Input ceilings

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16` points; one object per call; masks are binarised at logit `MASK_THRESHOLD = 0.0`. `iou_scores` is the model's own uncalibrated estimate and can exceed 1.0. See `MODEL_CARD.md` for the measured CPU timings and the candidate-selection rule.

## Release status

**Candidate / source-complete.** The pipeline package, offline unit tests, a local CPU smoke run, and `MODEL_CARD.md` (MODEL_CARD_SPEC 1.1) exist. No tutorial notebook ships yet; nothing here is clean-runtime notebook evidence.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights are Apache-2.0; see `docs/WEIGHTS.md` and `MODEL_CARD.md`.
