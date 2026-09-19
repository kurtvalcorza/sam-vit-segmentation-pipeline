"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E promptable-segmentation fine-tuning workflow: the pinned facebook/sam-vit-base
snapshot is digest-verified and loaded, three digest-pinned row groups of the ADE20K scene-parsing validation shard
(300 images, BSD-3-Clause) are fetched over HTTPS range requests and turned into one labelled target per image (the
largest connected component of one class covering 3..35 % of the image, with an interior click and the tight box), the
records are validated and split by image, a synthetic scene is segmented through the inference contract, the frozen
model's point- and box-prompt IoU over the held-out targets is measured beside two prompt-only baselines, a bounded
fine-tuning of the mask decoder runs with the BCE + soft-Dice loss under mixed prompts, the held-out split is scored
again per class, the scene and four held-out targets are re-segmented with the adapted decoder, and the adapter is
exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "sam_vit_segmentation_pipeline",
    "repo_name": "sam-vit-segmentation-pipeline",
    "stem": "sam_vit_segmentation",
    "notebook_name": "sam_vit_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "SAMViTSegmentationPipeline",
    "weights_key": "sam-vit-base",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers"],
    "title": "SAM ViT-B — DIMER E2E promptable-segmentation fine-tuning tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/sam-vit-segmentation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/sam-vit-segmentation-pipeline/blob/main/tutorials/sam_vit_segmentation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-facebook%2Fsam--vit--base-ffcc4d?style=flat",
            "https://huggingface.co/facebook/sam-vit-base",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-facebookresearch%2Fsegment--anything-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/facebookresearch/segment-anything",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2304.02643-b31b1b.svg", "https://arxiv.org/abs/2304.02643"),
    ],
    "capability": "promptable image segmentation (one point click or one box → one object's mask) and bounded supervised fine-tuning of the mask decoder on labelled point/box → mask records, using the pinned `facebook/sam-vit-base` weights (SAM v1, ViT-B)",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `facebook/sam-vit-base` snapshot (a 375 MB `model.safetensors`; no pickle is opened anywhere), fetches the first "
        "three row groups of the ADE20K validation parquet shard from the Hugging Face Hub at an immutable revision (about "
        "14 MB over HTTPS range requests; the shard's declared size is checked first and each row group's decoded content is "
        "refused on any SHA-256 or byte-total mismatch), turns the 300 images into 296 labelled targets with an interior click "
        "and a tight box each, splits them by image into 180 / 45 / 70, segments a synthetic scene through the inference "
        "contract with an input manifest and a rejection probe, measures the frozen model's point-prompt and box-prompt IoU "
        "over the 70 held-out targets beside two prompt-only baselines, runs a bounded fine-tuning of the mask decoder with "
        "the BCE + soft-Dice loss under mixed prompts and validation-IoU epoch selection, scores the held-out targets again "
        "per class, re-segments the scene and four held-out targets with the adapted decoder, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify mask parity. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On an RTX 5070 Ti the whole path took under ten minutes after the downloads; on CPU the "
        "image encoder alone costs several seconds per image at the fixed 1024×1024 working size, so expect an hour or more "
        "on a 2-vCPU hosted runtime — a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of images with a `<stem>_mask.png` beside each (white = the object; optionally a `labels.csv` of `id`, `file`, "
        "`category`) — at least eight images with sides between 16 and 4096 px. The click and the box are derived from each "
        "mask (its distance-transform maximum and its tight box), then the records pass through the same validation, "
        "image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the "
        "ADE20K sample. Uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "`facebook/sam-vit-base` is the Segment Anything Model of Kirillov et al. (2023) in its smallest published size: a "
        "ViT-B image encoder that embeds a 1024×1024 padded image once, a prompt encoder for clicks and boxes, and a "
        "lightweight two-way-transformer mask decoder that turns the two embeddings into up to three candidate masks at "
        "256×256 with a predicted IoU each (93,735,472 parameters, of which the decoder is 4,058,340; published under the "
        "**Apache-2.0** licence). The pipeline up-samples the chosen candidate to the input resolution, strips the padding "
        "and binarises at logit 0; the predicted IoU is a learned, uncalibrated ranking score, not a probability.\n\n"
        "What this notebook adds to inference is **adaptation with labelled targets**. SAM was trained to return *some* "
        "object at a click — a part, the whole, or the object with its surroundings — which is exactly the ambiguity a "
        "scene-parsing dataset resolves one way: ADE20K's targets are whole semantic regions, and in the validation images "
        "the qualifying ones are mostly amorphous *stuff* (walls, sky, floors, roads, ceilings) rather than crisp things. On "
        "such targets the frozen model's single click is weak — the build record measured a mean IoU of **0.564** for the "
        "point prompt on the 70 held-out targets, *below* the 0.614 of simply filling the box — while the box prompt reaches "
        "0.784. So the honest question is narrow: does a bounded fine-tuning of the mask decoder alone (the encoders stay "
        "frozen and every training image is embedded once) on 180 targets under **mixed** point and box prompts move the "
        "held-out **point-prompt IoU** past the two prompt-only baselines while keeping the box prompt where it was? Nothing "
        "here is a claim about your images: it is one seeded split of one sample of one dataset's labelling convention.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (a 4-file manifest); the upstream `pytorch_model.bin` "
        "and TensorFlow weights are not in the manifest and are never staged or loaded. Section 3 stages and digest-verifies "
        "those files before the processor or the model is constructed."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned slice of a real segmentation dataset, turn its annotations into "
        "prompt-to-mask targets under a stated selection rule, validate them and split them by image without leakage; "
        "segment a synthetic scene through the public API and read the output contract correctly (three candidates, an "
        "uncalibrated predicted IoU that can exceed 1.0, one object per call); measure the frozen model's point- and "
        "box-prompt IoU beside two prompt-only baselines and read the per-class breakdown; run a bounded fine-tuning of the "
        "mask decoder with a stated loss, explicit hyperparameters and validation-based epoch selection; evaluate on an "
        "image-disjoint test split; look at the adapted masks next to the frozen ones and the targets; and export a "
        "safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "automatic \"segment everything\" mask generation, text prompts (see the sibling Grounding DINO pipeline), "
        "mask-input prompts, several objects in one call, semantic class prediction (the class name is carried as a label for "
        "the breakdown, never predicted), video, fine-tuning of the image or prompt encoder, evaluation on the full ADE20K "
        "validation set or any benchmark proper (only one seeded 296-target sample from its first 300 images is scored here), "
        "and any claim that scene-parsing regions stand in for your objects. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU (float32) and uses CUDA automatically when available. The image encoder's cost is fixed by the 1024×1024 working size (about 3 s per image on the build workstation's CPU, 0.17 s on an RTX 5070 Ti), and the default path encodes every image several times (two frozen evaluations, one cached embedding pass for training, per-epoch validation, two adapted evaluations): the build record measured 30.6 s for the 180 training embeddings and 194 s for the six epochs with validation scoring on the RTX 5070 Ti, and the whole default path took about eight minutes there with the snapshot and row groups already cached. A CPU-only or 2-vCPU hosted runtime will take an hour or more. The pinned `torch==2.14.0` install and the 375 MB checkpoint are the large downloads of the run; the three row groups are about 14 MB.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a binary mask and intersection-over-union are; why a self-made reference is a plumbing check and a held-out split under one dataset's labelling convention is a measurement of that convention only.",
        "- **Data contract:** records are `{id, image, mask, point, box}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4096 px, `mask` a boolean array of the same height and width (or a mask image, white = target), `point` one `[x, y]` click inside the mask, `box` the `[x0, y0, x1, y1]` box enclosing it, an optional `category` of at most 32 characters. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images with a `<stem>_mask.*` each plus an optional `labels.csv`; the notebook derives the click and the box itself.",
        "- **Validation is structural, not semantic:** every image and mask is decoded, the click is checked to lie inside the mask and the box to enclose it, but nothing checks that the mask outlines what you meant — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads three row groups of `scene_parsing/validation/0000.parquet` from `https://huggingface.co/datasets/zhoubolei/scene_parse_150` at the immutable parquet-conversion revision `e660d866…` (about 14 MB over HTTPS range requests; the shard's declared size and every row group's decoded SHA-256 and byte total are pinned in the carried `samples.py` and refused on any mismatch). ADE20K scene parsing is published under the BSD-3-Clause licence (MIT CSAIL, Zhou et al. 2017); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. ADE20K targets, prompts and split\n\n"
                "`fetch_corpus` returns the three pinned row groups from the cache under `weights/ade20k/` (one parquet file "
                "per row group, re-hashed on every read) or the Hub — the shard's footer and the wanted row groups are read "
                "over HTTPS range requests, and each row group's decoded image and annotation bytes are refused unless their "
                "SHA-256 and byte total match `ROW_GROUP_PINS`. `read_corpus` turns each image into at most one record under "
                "a stated rule: `select_target` keeps the largest 4-connected component of any class covering "
                "`TARGET_MIN_FRACTION`..`TARGET_MAX_FRACTION` of the image (3..35 %), the **point** prompt is the mask's "
                "distance-transform maximum (`interior_point`, a click well inside the region) and the **box** its tight "
                "bounds (`mask_box`); the ADE20K class name is the record's `category`. `build_sample_dataset` draws a seeded "
                "image-level split (180 / 45 / 70); `validate_dataset` checks every record against the contract and "
                "`check_split_disjoint` asserts no image (by decoded-pixel digest) is shared; the training split's summary "
                "table is written to `outputs/{stem}_train.csv`.\n\n"
                "Look for: 300 rows → 296 targets over about 55 classes, image sides within 240..975 px, mask fractions "
                "within 3..35 %, three digests, and four refusal probes — a duplicate id, a click outside its mask, a box that "
                "does not enclose the mask, and a dataset too small to use — each rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/ade20k')\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}}: {{CORPUS_REPO}}@{{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'rows': sum(len(v) for v in corpus_groups.values()), 'bytes': sum(len(r['image']) + len(r['annotation']) for v in corpus_groups.values() for r in v), 'targets': len(corpus), 'classes': len({{r['category'] for r in corpus}}), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint, 'target_rule': {{'component': 'largest 4-connected component of one class', 'area_fraction': [TARGET_MIN_FRACTION, TARGET_MAX_FRACTION], 'point': 'distance-transform maximum', 'box': 'tight bounds'}}}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    top = dict(sorted(manifest['category_counts'].items(), key=lambda kv: -kv[1])[:6])\n"
                "    print({{name: {{'n': manifest['n_records'], 'classes': len(manifest['category_counts']), 'top_classes': top, 'image_side': manifest['image_side'], 'mask_fraction': manifest['mask_fraction'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{'id': example['id'], 'category': example.get('category'), 'image': list(example['image'].size), 'mask_px': int(example['mask'].sum()), 'point': example['point'], 'box': example['box']}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'click outside its mask': [{{**train_records[0], 'point': [0.0, 0.0]}}, *train_records[1:8]],\n"
                "    'box does not enclose the mask': [{{**train_records[0], 'box': [train_records[0]['point'][0], train_records[0]['point'][1], train_records[0]['point'][0] + 1, train_records[0]['point'][1] + 1]}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Segment a synthetic scene through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a deterministic 320×240 RGB "
                "scene drawn in code — grey background, a dark filled rectangle at `[40, 60, 140, 180]` and a red filled disc "
                "at `[200, 80, 280, 160]` — with one foreground click at (90, 120) inside the rectangle and the rectangle kept "
                "as a reference mask; a different image family from the photographs, and a scene the adapted decoder will "
                "segment again in Section 9. `validate_inputs` applies exactly the checks `segment` applies (image sides "
                "`MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, 1..`MAX_PROMPTS` clicks inside the image with 0/1 labels, one xyxy box) "
                "and returns an input manifest; a click outside the image is validated too and its rejection recorded as a "
                "finding. `segment` returns `masks` of shape `(3, H, W)` with `multimask=True`, one **model-predicted** IoU "
                "per candidate — a learned, uncalibrated ranking score that **can exceed 1.0** — and the cleaned prompts; "
                "the conventional rule keeps the highest-scored candidate, and `predict_mask` applies exactly that rule for "
                "every corpus record. The per-image `evaluation_report` against the self-drawn rectangle is `sample-sanity` "
                "— plumbing evidence, not a measurement; whether the model is *good at real targets* is what Section 6 "
                "measures on 70 ADE20K regions. The inference-only card recorded scores `[0.955, 1.012, 0.979]` and "
                "`mask_iou` 1.000 on this scene."
            ),
            "code": (
                "scene = Image.new('RGB', (320, 240), (128, 128, 128))\n"
                "draw = ImageDraw.Draw(scene)\n"
                "rectangle_box = [40, 60, 140, 180]\n"
                "draw.rectangle(rectangle_box, fill=(30, 30, 30))\n"
                "draw.ellipse([200, 80, 280, 160], fill=(220, 30, 30))\n"
                "scene_reference = np.zeros((240, 320), dtype=bool)\n"
                "scene_reference[60:181, 40:141] = True\n"
                "scene_points, scene_labels = [[90, 120]], [1]\n"
                "scene_name = 'synthetic_scene_320x240.png'\n"
                "scene_sha256 = hashlib.sha256(np.asarray(scene).tobytes()).hexdigest()\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_PROMPTS': MAX_PROMPTS, 'NUM_MULTIMASK_OUTPUTS': NUM_MULTIMASK_OUTPUTS, 'MASK_THRESHOLD': MASK_THRESHOLD, 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(scene, points=scene_points, point_labels=scene_labels, multimask=True, names=[scene_name])\n"
                "try:\n"
                "    validate_inputs(scene, points=[[scene.width, scene.height]], point_labels=[1])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'click-outside-image-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': scene_name, 'sha256': scene_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def segment_scene(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.segment(scene, points=scene_points, point_labels=scene_labels, multimask=True)\n"
                "    elapsed = time.perf_counter() - started\n"
                "    masks = result['masks']\n"
                "    best = int(np.argmax(result['iou_scores']))\n"
                "    checks = {{\n"
                "        'shape': masks.shape == (NUM_MULTIMASK_OUTPUTS, scene.height, scene.width),\n"
                "        'dtype_bool': masks.dtype == np.bool_,\n"
                "        'one_score_per_mask': len(result['iou_scores']) == masks.shape[0],\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'segment output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, scene_reference, sample_kind='synthetic (authored in this notebook)')\n"
                "    Image.fromarray(masks[best]).save(f'outputs/{stem}_mask_{{label}}.png')\n"
                "    print({{label: {{'seconds': round(elapsed, 3), 'checks': checks, 'iou_scores_model_predicted': [round(v, 4) for v in result['iou_scores']], 'scores_above_one': [i for i, v in enumerate(result['iou_scores']) if v > 1.0], 'best_candidate': best, 'mask_iou_vs_reference': {{m['candidate']: round(m['value'], 3) for m in report['metrics']}} if report['metrics'] else None, 'verdict': report['verdict']}}}})\n"
                "    return result, report\n\n\n"
                "frozen_scene_result, frozen_scene = segment_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test targets\n\n"
                "Two prompt-only baselines frame the adaptation, each scored by `segmentation_metrics` (carried in "
                "`metrics.py`): mean **IoU** against the target mask and the **hit rate** — the fraction of targets reaching "
                "IoU ≥ 0.5 — overall and per class. The **box-fill** baseline answers with every pixel inside the box: no "
                "model, and for a rectangular region the perfect answer. The **centre-disk** baseline draws a disk around the "
                "click with the target's own area — the size given away, the shape not. The **frozen model** is scored twice "
                "by `pipe.evaluate`: with the **point** prompt (one click, the model's own best-of-three candidate) and with "
                "the **box** prompt. Expect the frozen point prompt **below box-fill** — the build record measured 0.564 (hit "
                "rate 0.60) against 0.614 — because a single click on a wall or a floor is ambiguous to a model trained to "
                "return *an* object, while the box prompt, which states the extent, reaches 0.784. Read the per-class rows to "
                "see where the click fails: large stuff regions, not small things."
            ),
            "code": (
                "METRICS = ('iou', 'hit_rate')\n"
                "baseline_box = box_fill_baseline(test_records)\n"
                "baseline_disk = centre_disk_baseline(test_records)\n"
                "print({{'box_fill_baseline': {{k: round(baseline_box[k], 3) for k in METRICS}}, 'n': baseline_box['n'], 'note': baseline_box['baseline']}})\n"
                "print({{'centre_disk_baseline': {{k: round(baseline_disk[k], 3) for k in METRICS}}, 'note': baseline_disk['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_point = pipe.evaluate(test_records, prompt='point')\n"
                "frozen_box = pipe.evaluate(test_records, prompt='box')\n"
                "print({{'frozen_point_test': {{k: round(frozen_point[k], 3) for k in METRICS}}, 'frozen_box_test': {{k: round(frozen_box[k], 3) for k in METRICS}}, 'n': frozen_point['n'], 'verdict': frozen_point['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_point['definitions'], 'hit_threshold': HIT_THRESHOLD}})\n"
                "frozen_fields = {{c: {{'n': v['n'], 'point_iou': round(v['iou'], 3), 'box_iou': round(frozen_box['per_category'][c]['iou'], 3)}} for c, v in frozen_point['per_category'].items()}}\n"
                "print({{'by_class_frozen': dict(sorted(frozen_fields.items(), key=lambda kv: -kv[1]['n']))}})\n"
                "assert frozen_box['iou'] > baseline_disk['iou']"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the mask decoder\n\n"
                "`pipe.adapt` trains only the **mask decoder** — the two-way transformer, the up-scaling head and the "
                "IoU-prediction head, 4,058,340 of 93,735,472 parameters — while the image encoder and the prompt encoder stay "
                "frozen. Because the image encoder is frozen, every training image is embedded **once** (cached under "
                "`torch.no_grad`, about 30 s on the build GPU) and each step runs only the prompt encoder and the decoder: "
                "one record per step, the prompt drawn by a seeded coin — the record's click or its box (`PROMPTS = 'mixed'`) "
                "— and the decoder's single-mask logits compared with the target in the 256×256 low-resolution frame under "
                "**binary cross-entropy plus a soft Dice term**. AdamW without weight decay at a fixed learning rate, gradient "
                "clipping at 1.0, seeded shuffling, no scheduler. Epoch 0 records the frozen model's validation metrics; every "
                "epoch is scored on the 45 validation targets with the point prompt, and the epoch with the highest validation "
                "point IoU is kept.\n\n"
                "Watch the training loss fall from about 0.5 while the validation point IoU climbs by roughly 0.15 over six "
                "epochs: the decoder is learning *which* of its candidate granularities this dataset means by a click, which "
                "180 targets are enough to teach. The build record's counter-example — the same recipe at twice the rate for "
                "four epochs — gained less on the point prompt and cost the box prompt a tenth; the default is the "
                "configuration that lifted the click past both baselines while holding the box."
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "PROMPTS = 'mixed'  # @param [\"mixed\", \"point\", \"box\"]\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_point_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, prompts=PROMPTS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'embedding_seconds': adapt_result['embedding_seconds'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test targets were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted decoder is scored exactly as the frozen one was in Section 6 — point prompt and box prompt — the "
                "four systems are put side by side, and the per-class breakdown is repeated. Read it in this order: the "
                "**point-prompt IoU** first (the measure the epoch was selected on — the build record measured 0.564 → 0.709, "
                "past box-fill's 0.614 and the disk's 0.442; hit rate 0.60 → 0.79), then the **box-prompt IoU** (0.784 → "
                "0.772: held within a hundredth, the cost of teaching the click), then the per-class rows, where the large "
                "stuff classes gain the most. The cell asserts the adapted point IoU is above the frozen one and reports "
                "whether it is above box-fill. Seventy targets from one seeded split of one dataset give **no dispersion "
                "estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a "
                "gain on ADE20K's whole-region convention says nothing about a click on *your* objects until you measure it."
            ),
            "code": (
                "adapted_point = pipe.evaluate(test_records, prompt='point')\n"
                "adapted_box = pipe.evaluate(test_records, prompt='box')\n"
                "adapted_val = pipe.evaluate(val_records, prompt='point')\n"
                "adapted_fields = {{c: {{'n': v['n'], 'point_iou': round(v['iou'], 3), 'box_iou': round(adapted_box['per_category'][c]['iou'], 3)}} for c, v in adapted_point['per_category'].items()}}\n"
                "comparison = {{metric: {{'centre_disk': round(baseline_disk[metric], 3), 'box_fill': round(baseline_box[metric], 3), 'frozen_point': round(frozen_point[metric], 3), 'adapted_point': round(adapted_point[metric], 3), 'frozen_box': round(frozen_box[metric], 3), 'adapted_box': round(adapted_box[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_point_vs_frozen'] = {{metric: round(adapted_point[metric] - frozen_point[metric], 3) for metric in METRICS}}\n"
                "comparison['delta_point_vs_box_fill'] = {{metric: round(adapted_point[metric] - baseline_box[metric], 3) for metric in METRICS}}\n"
                "comparison['delta_box_vs_frozen'] = {{metric: round(adapted_box[metric] - frozen_box[metric], 3) for metric in METRICS}}\n"
                "comparison['by_class'] = {{c: {{'n': frozen_fields[c]['n'], 'frozen_point': frozen_fields[c]['point_iou'], 'adapted_point': adapted_fields[c]['point_iou'], 'frozen_box': frozen_fields[c]['box_iou'], 'adapted_box': adapted_fields[c]['box_iou']}} for c in sorted(frozen_fields, key=lambda c: -frozen_fields[c]['n'])}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'target_rule': {{'area_fraction': [TARGET_MIN_FRACTION, TARGET_MAX_FRACTION], 'point': 'distance-transform maximum', 'box': 'tight bounds'}},\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'box_fill': {{k: v for k, v in baseline_box.items() if k != 'per_record'}}, 'centre_disk': {{k: v for k, v in baseline_disk.items() if k != 'per_record'}}}},\n"
                "    'frozen_test': {{'point': frozen_point, 'box': frozen_box}},\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': {{'point': adapted_point, 'box': adapted_box}},\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_point['iou'] > frozen_point['iou']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_point_beats_box_fill': adapted_point['iou'] > baseline_box['iou']}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the masks, export the adapter and reload it\n\n"
                "The synthetic scene from Section 5 is segmented again by the adapted decoder — an image family the "
                "adaptation never saw, so this is a small look at what it did *outside* its corpus: the rectangle is a crisp "
                "thing, not ADE20K stuff, and the click should still return it (the build record kept `mask_iou` above 0.99) "
                "— and four held-out targets are written as side-by-side panels (`outputs/{stem}_examples/`: image with the "
                "click and the box drawn, frozen point mask, adapted point mask, target) so the numbers can be checked by eye: "
                "the adapted panels should cover the whole region where the frozen ones stopped at a part.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the mask decoder, about 16 MB — as `adapter.safetensors`, "
                "with a `manifest.json` recording the artifact format, the base model id and revision, the digest of the base "
                "`model.safetensors`, the tensor names, the file size and SHA-256, the training configuration and the epoch "
                "history (OUT8). `SAMViTSegmentationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact "
                "manifest, its digest and its exact tensor set **before** deserialising, refuses any tensor outside the mask "
                "decoder, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical point-prompt masks on eight test targets (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "adapted_scene_result, adapted_scene = segment_scene(pipe, 'adapted')\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "frozen_base = SAMViTSegmentationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR, device=pipe.device)\n\n\n"
                "def tint(image, mask, colour):\n"
                "    array = np.asarray(image.convert('RGB')).copy()\n"
                "    array[mask] = (0.45 * array[mask] + 0.55 * np.array(colour)).astype(np.uint8)\n"
                "    return Image.fromarray(array, mode='RGB')\n\n\n"
                "for record in test_records[:4]:\n"
                "    image = record['image']\n"
                "    prompted = image.copy()\n"
                "    marker = ImageDraw.Draw(prompted)\n"
                "    marker.rectangle(record['box'], outline=(255, 220, 0), width=3)\n"
                "    x, y = record['point']\n"
                "    marker.ellipse([x - 6, y - 6, x + 6, y + 6], fill=(255, 0, 0))\n"
                "    panels = [prompted, tint(image, frozen_base.predict_mask(record), (0, 120, 255)), tint(image, pipe.predict_mask(record), (0, 200, 80)), tint(image, record['mask'], (255, 255, 255))]\n"
                "    sheet = Image.new('RGB', (image.width * 4 + 30, image.height), (255, 255, 255))\n"
                "    for i, panel in enumerate(panels):\n"
                "        sheet.paste(panel, (i * (image.width + 10), 0))\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}_{{record.get('category', 'target').replace(' ', '-')}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panel_order': ['image with click and box', 'frozen point mask', 'adapted point mask', 'target']}})\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = SAMViTSegmentationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [pipe.predict_mask(r) for r in test_records[:8]]\n"
                "after = [reloaded.predict_mask(r) for r in test_records[:8]]\n"
                "parity = {{'identical_masks': sum(np.array_equal(a, b) for a, b in zip(before, after, strict=True)), 'of': len(before), 'max_pixels_differing': int(max((a != b).sum() for a, b in zip(before, after, strict=True)))}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_masks'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHT_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'shard_bytes': CORPUS_BYTES, 'row_groups': {{str(k): {{'sha256': v[0], 'bytes': v[1]}} for k, v in ROW_GROUP_PINS.items()}}, 'target_rule': {{'area_fraction': [TARGET_MIN_FRACTION, TARGET_MAX_FRACTION]}}}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256}}, 'frozen_report': frozen_scene, 'adapted_report': adapted_scene, 'output_files': ['outputs/{stem}_mask_frozen.png', 'outputs/{stem}_mask_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A single click on an ADE20K region is ambiguous to the frozen SAM decoder — it returns a part, or the region with its "
        "neighbours, often enough that the mean IoU (0.564 in the build record) sits below simply filling the box (0.614) — and "
        "a bounded fine-tuning of the 4-million-parameter mask decoder on 180 targets under mixed prompts resolves the "
        "ambiguity the way this dataset does (point IoU 0.709, hit rate 0.60 → 0.79) while holding the box prompt within a "
        "hundredth (0.784 → 0.772), with a 16 MB adapter that reloads mask-for-mask. That is the claim: the adaptation "
        "contract works end to end on a real labelled set, and the numbers it produces are read on two prompts, per class, "
        "against two prompt-only baselines and the frozen model rather than in isolation.\n\n"
        "The test split is 70 targets from one seeded draw of the first 300 validation images, the validation split that "
        "picks the epoch is 45, and the targets were chosen by a rule (largest component, 3..35 % of the image) that favours "
        "large stuff regions — walls, sky, floors, roads — over the small things a click is usually for. So a gain here says "
        "the decoder learned ADE20K's *whole-region* reading of a click, not that it will read your click the way you mean "
        "it, that it handles thin or occluded objects, or that its predicted IoU is now calibrated (it is not; scores above "
        "1.0 still occur). The adapted decoder is also a different model outside its corpus: the synthetic rectangle in "
        "Section 9 is one image of evidence that crisp things survive, not a measurement.\n\n"
        "Three things to carry to real data. **Baselines first:** fill the box and draw the disk on *your* targets before "
        "reading any model number, per class. **Labelling convention:** the masks the decoder learns from define what a "
        "click means; label your targets at the granularity you want returned, and keep the box prompt in the evaluation "
        "so a click-only gain that costs the box is visible. **Leakage:** keep every image in one split (the contract "
        "de-duplicates by decoded pixels) and split by scene or session when your images come from few sources.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, can "
        "acquire and digest-verify the pinned model snapshot, fetch and digest-verify a slice of a real segmentation dataset "
        "and turn it into prompt-to-mask targets under a stated rule, validate the demonstrated dataset contract without "
        "leakage, execute the inference contract and a bounded fine-tuning, evaluate against two prompt-only baselines and "
        "the frozen model on an image-disjoint split, and emit the shown machine-readable artifacts — without the repository "
        "being reachable. It does **not** establish benchmark superiority, segmentation quality on any other labelling "
        "convention, calibration of the predicted IoU, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `PROMPTS = 'point'` and read whether the "
        "click gains more while the box loses more; set `PROMPTS = 'box'` and watch the point prompt barely move; raise "
        "`EPOCHS` and watch the validation IoU pick the epoch while the loss keeps falling; change `LEARNING_RATE` to "
        "`1e-4` and read the faster, box-costing climb the build record measured; or bring your own masks through BYOD and "
        "read the two baselines before the adapted number.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/sam-vit-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/sam-vit-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/sam-vit-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/facebookresearch/segment-anything\n"
        "- Segment Anything (Kirillov et al., ICCV 2023): https://arxiv.org/abs/2304.02643\n"
        "- ADE20K scene parsing (Zhou et al., CVPR 2017; IJCV 2019; BSD-3-Clause): https://github.com/CSAILVision/sceneparsing — parquet conversion https://huggingface.co/datasets/zhoubolei/scene_parse_150\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
