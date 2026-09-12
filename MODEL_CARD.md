---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: mask-generation
base_model: facebook/sam-vit-base
---

# SAM ViT-B (DIMER package v0.1.0) — Promptable Image Segmentation (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fsam--vit--base-ffcc4d?style=flat)](https://huggingface.co/facebook/sam-vit-base)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-facebookresearch%2Fsegment--anything-181717?style=flat&logo=github&logoColor=white)](https://github.com/facebookresearch/segment-anything)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2304.02643-b31b1b.svg)](https://arxiv.org/abs/2304.02643)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Pipeline](https://img.shields.io/badge/Pipeline-sam--vit--segmentation--pipeline-2ea44f?style=flat&logo=github)](https://github.com/kurtvalcorza/sam-vit-segmentation-pipeline)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This release ships no tutorial notebook (`tutorials/` is absent). The package is exercised through its test suite (`tests/`) and the run instructions in the README; a `NOTEBOOK_SPEC` 1.0 `TASK-INFERENCE` notebook is a follow-up, not a claim this card makes.

---

###### Description

`facebook/sam-vit-base` is the Transformers-format release of the ViT-B checkpoint of the Segment Anything Model from Meta AI's *Segment Anything* (Kirillov et al., ICCV 2023, arXiv:2304.02643), pinned here to revision `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`. The snapshot `config.json` declares `SamModel` with three modules: a ViT-B image encoder (12 layers, `hidden_size` 768, 16-px patches over a 1024×1024 input, windowed attention with `window_size` 14 and global attention at layers 2, 5, 8 and 11, relative position embeddings, 256 output channels), a prompt encoder for points and boxes (`hidden_size` 256, 4 point embeddings, a 64×64 image-embedding grid), and a two-layer two-way Transformer mask decoder with an IoU-prediction head (`iou_head_depth` 3) and `num_multimask_outputs` 3. At inference the model embeds the image once, encodes the caller's point clicks and/or box, and decodes one or three candidate masks with a predicted IoU each; no adaptation happens. This is the original 2023 SAM; the sibling `sam2-segmentation-pipeline` packages SAM 2.1 Hiera-Small, whose Hiera backbone and memory modules also cover video, and the two are not benchmarked against each other here. This repository exposes the prompted image path through `SamModel` + `SamProcessor` and adds packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `SAMViTSegmentationPipeline.from_pretrained` (verified local loading, `trust_remote_code=False`), `validate_image`/`validate_prompts`/`segment` (image and prompt validation, boolean masks at input resolution), and `mask_iou`.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is promptable single-object image segmentation: input one RGB still (`PIL.Image.Image`, any mode, converted to RGB) plus, for one object, up to 16 point clicks labelled 1 (foreground) or 0 (background) and/or one xyxy box; output `masks`, a boolean array of shape `(K, H, W)` at the input resolution with `K = 3` candidate masks when `multimask=True` (the default, best for a single ambiguous click) or `K = 1` when `multimask=False` (best for a box or several clicks), and `iou_scores`, the model's own predicted IoU per mask. Envisioned applications are interactive annotation tools where a human clicks and the mask is proposed, cutting out an object located by a detector (boxes from `grounding-dino-detection-pipeline` are valid prompts here), measuring the pixel area of a prompted region, and generating training masks that a human then reviews. Within DIMER the pipeline is an inference component and the SAM v1 baseline beside the SAM 2.1 sibling, not an automatic "segment everything" service.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, and application developers integrating promptable segmentation into annotation tooling, research prototypes, internal enterprise systems, or the DIMER workbench. A user is expected to understand that the model segments whatever visual region the prompt points at, without knowing what the object is; that `iou_scores` is the model's estimate of its own mask quality, uncalibrated, unrelated to semantic correctness, and not even bounded by 1.0 (the smoke run returned 1.012 and 1.002); that a single click is ambiguous (part, object, or group) and the three candidates exist for that reason; that mask boundaries at the input resolution are up-sampled from a 256×256 logit grid; and that mean IoU can only be measured against labelled masks they supply. Users who need semantic labels, text prompts, video tracking, or automatic whole-image segmentation are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no automatic mask generation (the upstream `mask-generation` grid-prompt pipeline is not wrapped), no class labels, no text prompts (use `grounding-dino-detection-pipeline` to obtain boxes from text), no mask-input prompts, no video (SAM v1 is image-only; `sam2-segmentation-pipeline` also stops at images), one object per call, no batching.
2. **Input boundary:** `segment` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a call with neither points nor box, more than `MAX_PROMPTS = 16` points, points without labels or with labels other than 0/1, points or boxes outside the image, empty or inverted boxes, and a non-bool `multimask` (`ValueError`/`TypeError`). Every image is resized so its longest edge is 1024 px and padded to 1024×1024, so thin structures below roughly 1/1024 of the longer side are lost.
3. **Input boundary:** non-photographic imagery (medical volumes, microscopy, radar, line art) falls outside the SA-1B distribution; masks on it are undefined and the pipeline does not detect it.
4. **Decision boundary:** not for autonomous measurement or triage with medical, legal, safety, or financial consequence (lesion area, property boundaries, defect acceptance) without a human reviewing each mask and a locally measured IoU on labelled data.

#### Factors

###### Groups

This pipeline is not human-centric by design: it delineates the region a prompt points at and neither classifies nor identifies people. It will, however, segment a person, face, or body part as readily as any object when prompted there. The snapshot README says the model was trained on a dataset of 11 million images and 1.1 billion masks and quotes the paper's description of them as "licensed and privacy respecting"; it discloses no geographic or demographic composition, and the SAM paper's own responsible-AI analysis (which this repository has not reproduced) is the only group-level evidence upstream offers. This repository has not audited mask quality across skin tone, age, gender, disability, or dress, so any difference across such groups is unknown, not known to be absent. A downstream operator who segments images of people is responsible for a fairness audit on their own data: stratify a labelled sample by the relevant groups and compare `mask_iou` per stratum before relying on the output.

###### Instrumentation

The upstream training masks were produced by SAM's model-in-the-loop "data engine" over licensed photographs of unspecified camera provenance (paper abstract, quoted in the snapshot README); the masks are therefore themselves partly model-generated and human-corrected rather than sensor ground truth. Inference images arrive from whatever camera the operator uses; sensor resolution, lens distortion, compression, motion blur, and low light change the edges the model sees, and the longest-edge-1024 resize with bilinear resampling and ImageNet mean/std (`preprocessor_config.json`: `resample` 2, `pad_size` 1024×1024) plus the 256×256 mask grid bound the boundary precision regardless of source resolution. Prompts are a second instrument: a click a few pixels off the object, or a box that clips it, changes the mask; the pipeline validates that prompts lie inside the image but cannot tell whether they lie on the intended object.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; `from_pretrained` picks `cuda:0` when a GPU is visible, but the CUDA path was not exercised for this card. Measured 2026-09-12 in the Windows venv with `CUDA_VISIBLE_DEVICES=-1` and `device="cpu"`: `verify_snapshot` 0.21 s over 4 files (375 MB), load 3.77 s, a 320×240 synthetic scene 2.93 s for one point prompt and 2.53 s for one box prompt, a 4096×4096 noise image with a box 3.11 s; process wall 14.16 s. Cost is dominated by the ViT-B encoder at the fixed 1024×1024 working resolution — roughly three times the per-prompt time the SAM 2.1 Hiera-Small sibling recorded on the same machine and scene — and the caller's resolution mostly sets the size of the up-sampled boolean masks (a 4096×4096 three-mask result is 48 MiB). Data environment: the model assumes an ordinary photograph in which the prompted object has a visible boundary; low contrast, transparency, thin structures, and heavy occlusion produce masks that bleed or fragment, and the predicted IoU may stay high while they do.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. Each mask carries an entry in `iou_scores`, the mask decoder's own prediction of that mask's IoU with the intended object — a self-estimate used to rank the candidates, not a measurement against ground truth. The repository ships `mask_iou(a, b)`, the intersection-over-union of two boolean masks, because it is the primitive every segmentation metric is built from; mean IoU itself is not implemented, since it needs labelled masks and a convention for which candidate is scored (best-of-3 or argmax-of-`iou_scores`) that the caller must choose. To evaluate, the caller supplies ground-truth masks and computes `mask_iou` per object, then averages. The SAM paper's zero-shot benchmark numbers are upstream-reported and not reproduced or claimed here; the smoke run's `mask_iou` of 1.000 (three decimals; 12216 versus 12221 px) against a synthetic drawn rectangle is a consistency check, not a benchmark.

###### Decision thresholds

One threshold is applied: the up-sampled mask logits are binarised at `MASK_THRESHOLD = 0.0` (`SamProcessor.post_process_masks` default), so a pixel is foreground when its logit is positive; this is the implicit decision rule behind every `True` in `masks` and is not tuned per domain. No threshold is applied to `iou_scores`: all `K` candidates are returned with their scores and the caller picks (the smoke run used `argmax`, itself a threshold-free rule that the card names here). The choice between `multimask=True` and `False` is a second decision the caller owns. A deployment that wants an acceptance rule — for example rejecting masks whose predicted IoU is below some value, or shifting the logit threshold to favour tighter or looser masks — must set it against its own labelled data, weighing the cost of a mask that bleeds into the background against one that misses part of the object, and owns revisiting it when the image source changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings, mask areas, and one IoU on a synthetic scene, not accuracy. Run-to-run variability comes from floating-point kernel selection across CPU builds and accelerators and from bilinear up-sampling of the 256×256 logits to the input size; there is no sampling and no seed to set, so a fixed input and prompt on fixed hardware is repeatable but not guaranteed bitwise-identical across machines. `iou_scores` is the model's own estimate and is not calibrated: it is an unclipped regression output, so a 1.012 (returned on the smoke scene) is not a probability of anything, and on that scene the three candidates scored 0.955, 1.012 and 0.979 for masks of 17129, 12216 and 11887 px. A caller who needs calibrated confidence must fit a map from `iou_scores` to measured `mask_iou` on their own labelled data; a caller who needs an uncertainty estimate for a metric must compute it over many labelled images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint beyond the analysis in the SAM paper; nothing below should be read as implying one.

###### Data

The snapshot README states that the model "has been trained on a dataset of 11 million images and 1.1 billion masks" (SA-1B) and quotes the paper's description of the images as "licensed and privacy respecting"; the disclosure ends there — no per-item consent status, source list, or demographic breakdown is given in the snapshot, and the privacy processing (face and licence-plate blurring, described in the SAM paper) is an upstream claim this repository cannot verify, so personal data in the corpus is not ruled out. This repository distributes code, tests, and documentation; it does not distribute the 374,979,480-byte `model.safetensors`, which is staged locally under `weights/sam-vit-base/` and git-ignored, and it ships no sample images. The operator must audit the images and prompts they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will delineate whatever it is pointed at.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Its only validation is the offline unit suite and the CPU smoke run recorded in this repository. Foreseeable but unintended sensitive uses — outlining lesions or organs in medical images, isolating people in surveillance stills, measuring damage for insurance claims, delineating land parcels — would be admissible only with human review of every mask before action, a locally measured IoU against expert-drawn masks on the deployment's own data, a documented threshold and candidate-selection policy, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 4 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True` and always passes `trust_remote_code=False`. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused.
- **Input integrity:** `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; `validate_prompts` rejects a call with no prompt, more than 16 points, missing or non-0/1 labels, out-of-image points, and empty, inverted, or out-of-image boxes; `segment` rejects a non-bool `multimask` and raises `RuntimeError` on a backend result whose shape, dtype, or score count is wrong.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; float32 forced at load (`dtype=torch.float32`, matching the snapshot's `torch_dtype`); every result carries `model_id`, `model_revision`, and the exact validated `points`, `point_labels`, and `box` sent to the model.
- **Refusals:** automatic mask generation, mask-input prompts, batching, and download without the explicit flag are not exposed; the padded 1024×1024 working image is stripped back to the caller's resolution by passing `reshaped_input_sizes` to post-processing, so a caller never receives a mask with padding in it.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Wrong extent:** a click on a part yields the part, the object, or a group; the wrong candidate is chosen by `argmax` and the harm falls on whoever acts on the area or crop; likely for ambiguous single clicks (on the smoke scene the three candidates differed by 44 % in area).
- **Boundary bleed:** low-contrast or transparent edges produce masks that include background or exclude the object; the operator bears the harm when the mask drives a measurement.
- **Overconfident self-score:** `iou_scores` can be high — even above 1.0 — for a mask that is semantically wrong; automation bias follows when reviewers trust the number.
- **Prompt-driven targeting:** the model will isolate any person or body part it is pointed at; the data subject bears the harm when such masks drive tracking, redaction failures, or manipulation.
- **Privacy exposure:** images of people or private spaces are processed without any content check.
- **Bias amplification:** any imbalance in SA-1B is reproduced as uneven mask quality across appearance groups, undetected because no per-group evaluation exists here.
- **Resource use:** about 2.5–3 s per prompt on the reference CPU, a 375 MB checkpoint, and up to 48 MiB of boolean masks per call at the 4096-px ceiling; a request stream can saturate a shared host.

###### Use cases

Prohibited even where the model would work: covert surveillance or tracking of individuals, biometric or demographic profiling, social scoring, and any use that discriminates unlawfully in employment, housing, credit, insurance, education, or healthcare access. Also prohibited are deceptive or non-consensual image manipulation — isolating a person to composite, undress, or misrepresent them, or fabricating "measured" evidence from a mask — and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by an unreviewed mask are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `facebook/sam-vit-base`
- Revision: `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`
- Snapshot manifest: `weights/sam-vit-base/dimer-base-manifest.json`, 4 files, `totalBytes` 374993239
- `model.safetensors` SHA-256: `892c410e496344e527255ccdcb2cb7244a609acb5389c7c4fdba1288f861c579` (374,979,480 bytes)
- `config.json` SHA-256: `5ebd0d8643b486f3a716bf17c2a15531eb818b2b96ff0c6c5dcc88fa015161af` (6,566 bytes)
- Weight format: SafeTensors; loader `SamModel.from_pretrained(<dir>, revision=MODEL_REVISION, dtype=torch.float32, local_files_only=True, trust_remote_code=False)` with `SamProcessor` from the same directory (`preprocessor_config.json`, 466 bytes, also in the manifest).

## Input/output contract

- `SAMViTSegmentationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`.
- `segment(image, *, points=None, point_labels=None, box=None, multimask=True) -> dict` — one object per call; `points` is a sequence of `[x, y]` pixel pairs with `point_labels` of 0/1, `box` is `[x0, y0, x1, y1]` in pixels. Returns `masks` (bool `(K, H, W)`, `K` = 3 if `multimask` else 1), `iou_scores` (list of `K` floats, model-predicted, uncalibrated, not bounded by 1.0), `multimask`, the validated `points`/`point_labels`/`box`, `width`, `height`, `model_id`, `model_revision`.
- Ceilings: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`; `NUM_MULTIMASK_OUTPUTS = 3`; `MASK_THRESHOLD = 0.0`.
- `mask_iou(a, b) -> float` on boolean arrays; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `huggingface-hub==0.36.2`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Precision: float32; preprocessing resize longest edge to 1024 (bilinear, `resample` 2), pad to 1024×1024, ImageNet mean/std (`SamImageProcessor` from the snapshot); masks decoded at 256×256, padding removed via `reshaped_input_sizes`, up-sampled to the input size, binarised at logit 0.
- Measured 2026-09-12 in the Windows venv (`torch 2.14.0+cu130`, `torch.cuda.is_available()` False under `CUDA_VISIBLE_DEVICES=-1`), device `cpu`, source local snapshot: `verify_snapshot` 0.21 s (4 files, 375 MB); load 3.77 s; `segment` on a synthetic 320×240 scene (grey background, dark rectangle at [40, 60, 140, 180], red disc centred at (240, 120) with radius 40) with one foreground click at (90, 120) → `masks (3, 240, 320)` bool, `iou_scores` [0.955, 1.012, 0.979], areas [17129, 12216, 11887] px, `mask_iou` of the argmax candidate (index 1) against the drawn 12221-px rectangle 1.000 at three decimals, 2.93 s; box `[200, 80, 281, 161]` with `multimask=False` → one mask of 5132 px (drawn disc 5145 px), `iou_scores` [1.002], `mask_iou` 0.997, 2.53 s; 4096×4096 uniform-noise image with a box → `(1, 4096, 4096)` in 3.11 s. Process wall 14.16 s; nothing written to stderr.
- Tests: `pytest -q -o addopts= tests` — 12 passed, offline, no weights required; `ruff check src tests` clean.
- Not executed: CUDA path, automatic mask generation, mask-input prompts, any IoU measurement against real labelled masks, the Hub download path (`allow_download=True` is covered only by the injected-downloader unit test).

## References

- Kirillov, Mintun, Ravi, Mao, Rolland, Gustafson, Xiao, Whitehead, Berg, Lo, Dollár, Girshick. Segment Anything. ICCV 2023. https://arxiv.org/abs/2304.02643
- Dosovitskiy et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR 2021. https://arxiv.org/abs/2010.11929
- Ravi et al. SAM 2: Segment Anything in Images and Videos. arXiv:2408.00714, 2024 (the sibling `sam2-segmentation-pipeline`). https://arxiv.org/abs/2408.00714
- Upstream code: https://github.com/facebookresearch/segment-anything
- Upstream card: https://huggingface.co/facebook/sam-vit-base
- Transformers `SAM` documentation: https://huggingface.co/docs/transformers/model_doc/sam
