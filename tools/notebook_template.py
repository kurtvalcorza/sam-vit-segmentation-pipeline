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
    "isolated_runtime": True,
    "infrastructure_labels": True,
    # The fleet's uv isolated-environment mechanism (generator /2.2): managed CPython, a
    # size- and SHA-256-verified uv wheel, and a lock compiled from the pyproject pins with
    # `uv pip compile pyproject.toml --python-version 3.12 --python-platform x86_64-manylinux_2_28 --generate-hashes
    # --only-binary :all: -o tutorials/requirements-colab.lock.txt`.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
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
        "Selecting **Run all** in a fresh supported runtime builds an isolated, hash-locked environment with the pinned dependencies "
        "(nothing is installed into the notebook kernel, so no restart is needed), stages and digest-verifies the "
        "pinned `facebook/sam-vit-base` snapshot (a 375 MB `model.safetensors`; no pickle is opened anywhere), fetches the first "
        "three row groups of the ADE20K validation parquet shard from the Hugging Face Hub at an immutable revision (about "
        "14 MB over HTTPS range requests; the shard's declared size is checked first and each row group's decoded content is "
        "refused on any SHA-256 or byte-total mismatch), turns the 300 images into 296 labelled targets with an interior click "
        "and a tight box each, splits them by image into 180 / 45 / 70, segments a synthetic scene through the inference "
        "contract with an input manifest and a rejection probe, measures the frozen model's point-prompt and box-prompt IoU "
        "over the 70 held-out targets beside two prompt-only baselines, runs a bounded fine-tuning of the mask decoder with "
        "the BCE + soft-Dice loss under mixed prompts and epoch selection on the mean validation point- and box-prompt IoU, scores the held-out targets again "
        "per class, re-segments the scene and four held-out targets with the adapted decoder, exports the adapter as "
        "safetensors with a manifest, and reloads that artifact into a fresh pipeline to verify mask parity. The default path "
        "needs no repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On an RTX 5070 Ti the whole path took about seven minutes after the downloads; on CPU the "
        "image encoder alone costs several seconds per image at the fixed 1024×1024 working size, so expect an hour or more "
        "on a 2-vCPU hosted runtime — a CUDA runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 (and `BYOD_PATH` to the zip or directory on Kaggle or "
        "Jupyter; on Colab an empty path opens the upload dialog) and re-run from that cell to supply one zip "
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
    "guided": {"opening": [(
        "**Who this notebook is for.** A learner who knows basic Python and NumPy, has used Colab or Jupyter and has met binary masks and intersection-over-union, and wants to see what a click or a box means to a promptable segmenter — and what a bounded fine-tuning of its mask decoder changes on a real labelled set. The audience is students and practitioners deciding whether SAM can be adapted to their own labelling convention; no prior experience with SAM, transformers or fine-tuning is assumed — each term is explained where it first matters and again in the **Glossary**. A GPU runtime is strongly recommended (the image encoder is slow on CPU).\n\n**Input → Model → Output.**\n\n| | |\n|---|---|\n| Input | `{{id, image, mask, point, box}}` records; the default is 296 ADE20K targets (three digest-pinned row groups of the validation shard, 300 images) split 180 / 45 / 70 by image, plus a 320 × 240 synthetic scene drawn in code |\n| Model | `facebook/sam-vit-base` (SAM v1, ViT-B, 93.7 M parameters): a frozen image encoder, a frozen prompt encoder and a 4.06 M-parameter mask decoder — the only part that is trained |\n| Output | mean IoU and hit rate of the frozen and the adapted model under the point and the box prompt against two prompt-only baselines, per class; a mask of the synthetic scene before and after; four side-by-side panels; a 16 MB safetensors adapter that reloads with verified mask parity; `result.json` |\n\n**How to use this notebook.** Choose a **GPU** runtime (T4 or better) and then **Runtime → Run all**. Run all completes in one pass: Section 1 installs nothing into the notebook's own Python, so no restart is needed (the recorded hosted run of the previous version needed one; this version removes it). Sections 1–3 are **infrastructure** — the isolated environment, the carried modules and the verified snapshot — and their cells are collapsed; you may run them without studying them. The learning path starts in Section 4. Form fields (`# @param`) are the only values meant to be edited, and the defaults reproduce the recorded run. Before each principal result the notebook asks you to **Predict**; after it comes a collapsible **Check your reasoning** with a worked answer from the recorded Kaggle T4 run of 20 September 2026. **Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Budget about fifteen minutes on a T4 (912 s in the recorded run, downloads included); an hour or more on CPU.\n\n**Roadmap:** 1–3 infrastructure → 4 ADE20K targets under a stated selection rule, the split and four refusals *(core concept: the data contract and leakage)* → 5 the inference contract on a synthetic scene *(core concept: three candidates and an uncalibrated predicted IoU)* → 6 two prompt-only baselines and the frozen model under both prompts *(evaluation practice: baselines before model numbers)* → 7 bounded fine-tuning of the mask decoder with epoch selection on both prompts *(core concept: what 4 M of 94 M parameters can learn)* → 8 held-out evaluation and a recorded verdict *(evaluation practice)* → 9 masks by eye, export and reload parity *(engineering)* → conclude."
    )]},
    "prerequisites": [
        "- **Learner:** basic Python, NumPy and Colab or Jupyter familiarity; no prior experience with SAM or fine-tuning. Prompts, the predicted IoU, the two baselines, the mask decoder, the loss, epoch selection and reload parity are explained where they are first used and again in the Glossary.",
        "- **Runtime:** a fresh supported **Linux x86_64** runtime (Google Colab, Kaggle or a Linux Jupyter server). Section 1 builds its own Python 3.12.12 environment from a hash-locked list of manylinux wheels, so the Python version of the kernel itself does not matter and nothing is installed into it; a Windows or macOS kernel is not supported. The default path runs on CPU (float32) and uses CUDA automatically when available. The image encoder's cost is fixed by the 1024×1024 working size (about 3 s per image on the build workstation's CPU, 0.17 s on an RTX 5070 Ti), and the default path encodes every image several times (two frozen evaluations, one cached embedding pass for training, per-epoch validation, two adapted evaluations): the build record measured 30.0 s for the 180 training embeddings and 283 s for the six epochs with validation scoring on both prompts on the RTX 5070 Ti, and the whole default path took 415 s there with the snapshot and row groups already cached. A CPU-only or 2-vCPU hosted runtime will take an hour or more. The pinned `torch==2.14.0` install and the 375 MB checkpoint are the large downloads of the run; the three row groups are about 14 MB.",
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
                "does not enclose the mask, and a dataset too small to use — each rejected before the model does anything.\n\n"
                "**Predict:** 300 images go in. Will every image yield a target under the 3..35 % rule, and which kinds of region — small things or large stuff — will the rule favour? Will any of the four probes be accepted?"
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                'def byod_source(path, kind):\n'
                '    """BYOD path first (a zip or a directory; works on Colab, Kaggle and Jupyter); on Colab an empty path opens the upload dialog."""\n'
                '    if str(path).strip():\n'
                '        source = Path(str(path).strip()).expanduser()\n'
                '        if not (source.is_file() or source.is_dir()):\n'
                "            raise FileNotFoundError(f'BYOD path {{str(source)!r}} does not exist (relative paths start at {{os.getcwd()}}); give the path of one {{kind}}.')\n"
                '    else:\n'
                '        try:\n'
                '            from google.colab import files\n'
                '        except ImportError:\n'
                "            raise RuntimeError(f'BYOD is on but BYOD_PATH is empty, and the upload dialog exists only in Google Colab: copy the {{kind}} into this runtime (or attach it as a Kaggle dataset) and set BYOD_PATH.') from None\n"
                '        uploaded = files.upload()\n'
                '        if len(uploaded) != 1:\n'
                "            raise ValueError(f'Upload exactly one {{kind}} (received {{len(uploaded)}} files; a cancelled dialog sends none). Run this cell again.')\n"
                '        name, payload = next(iter(uploaded.items()))\n'
                "        source = Path('work') / Path(name).name\n"
                '        source.parent.mkdir(parents=True, exist_ok=True)\n'
                '        source.write_bytes(payload)\n'
                "    if source.is_file() and not source.name.lower().endswith('.zip'):\n"
                "        raise ValueError(f'{{source.name}}: expected a zip of images with a <stem>_mask.png each, or a directory.')\n"
                '    return source\n'
                '\n'
                'if USE_BYOD:\n'
                "    byod_path = byod_source(BYOD_PATH, 'zip (or directory) of images with a <stem>_mask.png each')\n"
                '    records = load_byod_dataset(byod_path)\n'
                '    splits = split_dataset(records, seed=SPLIT_SEED)\n'
                "    data_source = 'BYOD (' + byod_path.name + ')'\n"
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
                "<details><summary>Check your reasoning</summary>No: the recorded run turned 300 rows into 296 targets (four images had no class component within 3..35 % of the image) over about 55 classes, and the rule favours large *stuff* — the hosted run's by-class table is led by wall (13 test targets), sky (10), floor (8), building (7) and road (7) — because the largest component of a class that covers a third of the image is a wall or a sky, not a cup. All four probes are refused before any model work, each naming its rule; a dataset with fewer than eight records is refused as too small.</details>"
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
                "`mask_iou` 1.000 on this scene.\n\n"
                "**Predict:** one click inside a crisp dark rectangle on grey. Will the best of the three candidates match the drawn rectangle almost exactly, and can a predicted IoU come out above 1.0?"
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
                '<details><summary>Check your reasoning</summary>Yes to both. The inference-only card recorded predicted IoUs of `[0.955, 1.012, 0.979]` — the middle candidate above 1.0, which is why the score is a ranking, not a probability — and `mask_iou` 1.000 for the selected candidate against the drawn rectangle; in the E2E pre-flight the selected candidate again scored 1.000. A crisp thing on a flat background is the easy case; Section 6 measures the hard one.</details>'
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
                "see where the click fails: large stuff regions, not small things. The cell records whether the frozen box prompt beats the "
                "centre-disk baseline as a verdict instead of stopping, so a BYOD set that behaves differently still reaches the export.\n\n"
                "**Predict:** rank the four systems — centre disk, box-fill, the frozen point prompt and the frozen box prompt — by mean IoU on the 70 held-out targets before the cell prints them. Does a model with one click beat a baseline with no model?"
            ),
            "code": (
                '# Sections 6 and 7 describe the frozen decoder and what one adaptation adds to it, so both start from the pretrained\n'
                '# mask decoder. The snapshot is taken once, before any training; a re-run restores it instead of scoring or training\n'
                '# whatever an earlier Section 7 left in the model.\n'
                'def restore_frozen_decoder():\n'
                '    """Put the mask-decoder tensors back to the frozen snapshot and forget any adapter, so the pipeline is the pretrained model again."""\n'
                '    model, _ = pipe._require_model()\n'
                '    state = model.state_dict()\n'
                '    model.load_state_dict({{name: value.to(state[name].device, state[name].dtype) for name, value in frozen_decoder_state.items()}}, strict=False)\n'
                '    model.eval()\n'
                '    pipe.adapter = None\n'
                '\n'
                "if 'frozen_decoder_state' not in globals():\n"
                '    _model, _ = pipe._require_model()\n'
                '    _decoder_names = set(_trainable_names(_model))\n'
                '    frozen_decoder_state = {{name: value.detach().cpu().clone() for name, value in _model.state_dict().items() if name in _decoder_names}}\n'
                "    print({{'frozen_decoder_snapshot': 'taken', 'tensors': len(frozen_decoder_state), 'adapted': pipe.adapter is not None}})\n"
                'else:\n'
                '    restore_frozen_decoder()\n'
                "    print({{'frozen_decoder_snapshot': 'restored', 'tensors': len(frozen_decoder_state), 'adapted': pipe.adapter is not None}})\n"
                '\n'
                "METRICS = ('iou', 'hit_rate')\n"
                'baseline_box = box_fill_baseline(test_records)\n'
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
                "# A recorded verdict, not an assert: a BYOD set where the box prompt does not beat the disk still reaches the export.\n"
                "frozen_verdict = 'above the centre-disk baseline' if frozen_box['iou'] > baseline_disk['iou'] else 'not above the centre-disk baseline'\n"
                "print({{'frozen_box_vs_centre_disk': frozen_verdict, 'frozen_box_iou': round(frozen_box['iou'], 3), 'centre_disk_iou': round(baseline_disk['iou'], 3)}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>In the recorded Kaggle T4 run: centre disk 0.442, frozen point prompt 0.564, box-fill 0.614, frozen box prompt 0.784 (hit rates 0.286, 0.60, 0.657, 0.857). A model with one click lost to a baseline with no model: on ADE20K's whole-region targets a single click is ambiguous, and SAM returns a part often enough to fall below filling the box. The box prompt, which states the extent, is far above both baselines — the verdict printed is `above the centre-disk baseline`.</details>"
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
                "epoch is scored on the 45 validation targets with both prompts, and the epoch with the highest **mean of the "
                "validation point- and box-prompt IoU** is kept — the box term keeps an epoch that taught the click at the "
                "box's expense from being chosen.\n\n"
                "Watch the training loss fall from about 0.39 to 0.29 while the validation point IoU climbs from 0.63 towards 0.72 over six "
                "epochs: the decoder is learning *which* of its candidate granularities this dataset means by a click, which "
                "180 targets are enough to teach. The build record's counter-examples — the same recipe at twice the rate for "
                "four epochs, which gained less on the click and cost the box a tenth, and selection on the point prompt alone, "
                "which once picked an epoch that cost the box 0.06 — are why the default selects on both prompts. Training is "
                "one record per step at a small learning rate, so the validation curve is noisy and the same recipe lands "
                "within a few hundredths of the numbers below from one GPU run to the next. The cell first restores the frozen decoder "
                "snapshot from Section 6, so a second run (after changing a setting) trains the pretrained decoder again, not the previous adaptation.\n\n"
                "**Predict:** six epochs, one record per step. Will the validation point IoU rise at every epoch, and will the kept epoch be the last one?"
            ),
            "code": (
                "EPOCHS = 6  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 5e-5  # @param {{type:\"number\"}}\n"
                "PROMPTS = 'mixed'  # @param [\"mixed\", \"point\", \"box\"]\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_point_iou': round(entry['val']['iou'], 3), 'val_box_iou': round(entry['val']['box_iou'], 3), 'val_score': round(entry['val']['score'], 3)}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "restore_frozen_decoder()  # every run of this cell trains the frozen decoder, never an earlier adaptation\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, prompts=PROMPTS, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'embedding_seconds': adapt_result['embedding_seconds'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                '<details><summary>Check your reasoning</summary>Not at every epoch, and not necessarily the last. The workstation pre-flight that fixed the recipe logged validation point IoU 0.629 → 0.645 / 0.694 / 0.654 / 0.639 / 0.691 / 0.715 and kept epoch 6 on the mean of both prompts, while the earlier point-only rule once kept epoch 3 of 6; the curve wobbles because each step sees one record. Read the `val_score` column, not the loss, to see which epoch is kept, and expect your own kept epoch to differ.</details>'
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test targets were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted decoder is scored exactly as the frozen one was in Section 6 — point prompt and box prompt — the "
                "four systems are put side by side, and the per-class breakdown is repeated. Read it in this order: the "
                "**point-prompt IoU** first (the build record measured 0.564 → 0.691, past box-fill's 0.614 and the disk's "
                "0.442; hit rate 0.60 → 0.79), then the **box-prompt IoU** (0.784 → 0.788: held, because the selection rule "
                "weighs it), then the per-class rows, where the classes a click reads worst on — wall, building, cabinet, "
                "seat — gain the most. The cell records verdicts instead of asserting — whether the adapted point IoU is above the frozen one "
                "(`improved` / `no gain` / `worse`), above box-fill, and whether the box prompt held — so a BYOD run that does not gain still "
                "exports and reloads; the verdicts go into the evaluation report and `result.json`. Seventy targets from one seeded split of one dataset give **no dispersion "
                "estimate**; the deltas are sample-sanity evidence that the adaptation contract works, not a benchmark, and a "
                "gain on ADE20K's whole-region convention says nothing about a click on *your* objects until you measure it.\n\n"
                "**Predict:** write down a direction and a size for the point-prompt delta and for the box-prompt delta. Will the click pass box-fill, and will the box prompt hold exactly, gain, or slip a little?"
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
                "# Recorded verdicts, not asserts: a BYOD run whose click does not gain still exports, reloads and writes result.json.\n"
                "delta_point = adapted_point['iou'] - frozen_point['iou']\n"
                "adaptation_verdict = 'improved' if delta_point > 0 else ('no gain' if delta_point == 0 else 'worse')\n"
                "comparison['verdicts'] = {{'frozen_box_vs_centre_disk': frozen_verdict, 'adapted_point_vs_frozen_point_iou': adaptation_verdict, 'adapted_point_vs_box_fill': 'above box-fill' if adapted_point['iou'] > baseline_box['iou'] else 'not above box-fill', 'adapted_box_vs_frozen_box_iou': 'held or improved' if adapted_box['iou'] >= frozen_box['iou'] else 'worse'}}\n"
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
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'verdicts': comparison['verdicts']}})"
            ),
        },
        {
            "md": (
                "<details><summary>Check your reasoning</summary>In the recorded Kaggle T4 run the point prompt went 0.564 → 0.663 (+0.099; hit rate 0.60 → 0.70) and passed box-fill's 0.614 by 0.049, while the box prompt slipped 0.784 → 0.778 (−0.006; hit rate 0.857 → 0.929). So the verdicts for that run are `improved`, `above box-fill` and `worse` for the box by six thousandths — within the few hundredths the recipe moves from one GPU run to the next (the workstation pre-flight gave 0.691 and 0.788). Wall, building, cabinet and seat gained most on the click.</details>"
            ),
        },
        {
            "md": (
                "## 9. Look at the masks, export the adapter and reload it\n\n"
                "The synthetic scene from Section 5 is segmented again by the adapted decoder — an image family the "
                "adaptation never saw, so this is a small look at what it did *outside* its corpus: the rectangle is a crisp "
                "thing, not ADE20K stuff, and the click should still return it (the build record's selected candidate kept `mask_iou` above 0.98) "
                "— and four held-out targets are written as side-by-side panels (`outputs/{stem}_examples/`: image with the "
                "click and the box drawn, frozen point mask, adapted point mask, target) so the numbers can be checked by eye: "
                "the adapted panels should cover the whole region where the frozen ones stopped at a part.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the mask decoder, about 16 MB — as `adapter.safetensors`, "
                "with a `manifest.json` recording the artifact format, the base model id and revision, the digest of the base "
                "`model.safetensors`, the tensor names, the file size and SHA-256, the training configuration and the epoch "
                "history (OUT8). `SAMViTSegmentationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact "
                "manifest, its digest and its exact tensor set **before** deserialising, refuses any tensor outside the mask "
                "decoder, and overlays the tensors onto a freshly loaded base — a new object from files, not the in-memory "
                "model (VER2). The cell asserts identical point-prompt masks on eight test targets (VER4).\n\n"
                "**Predict:** the reloaded pipeline is a new object built from files. Will all eight point-prompt masks be identical pixel for pixel, and will the adapted decoder still return the synthetic rectangle at the click?"
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
        {
            "md": (
                '<details><summary>Check your reasoning</summary>All eight masks identical — the recorded run reported `identical_masks: 8 of 8` with 0 pixels differing, because the reloaded pipeline runs the same tensors on the same device. The adapted decoder still returned the synthetic rectangle at the click (selected-candidate `mask_iou` above 0.98 in the build record): crisp things survive an adaptation to stuff, which is one image of evidence, not a measurement.</details>'
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A single click on an ADE20K region is ambiguous to the frozen SAM decoder — it returns a part, or the region with its "
        "neighbours, often enough that the mean IoU (0.564 in the build record) sits below simply filling the box (0.614) — and "
        "a bounded fine-tuning of the 4-million-parameter mask decoder on 180 targets under mixed prompts resolves the "
        "ambiguity the way this dataset does (point IoU 0.691, hit rate 0.60 → 0.79) while holding the box prompt (0.784 → "
        "0.788, hit rate 0.86 → 0.91), with a 16 MB adapter that reloads mask-for-mask. That is the claim: the adaptation "
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
        '## Troubleshooting\n'
        '\n'
        '- **Section 1 stops with "This notebook needs a Linux x86_64 runtime"** — use Google Colab, Kaggle or a Linux x86_64 Jupyter server.\n'
        '- **The uv wheel fails its size/SHA-256 check, or a download in Section 1 times out** — run Section 1 again; a complete environment is reused and an incomplete one is finished. If it repeats, `files.pythonhosted.org` or `pypi.org` is blocked or altered.\n'
        '- **You re-ran Section 1 on its own** — nothing is lost: it keeps the running worker and every variable. After a session restart, run from the top.\n'
        '- **"The isolated environment\'s Python process exited"** — usually out of memory; restart the session and choose **Run all**.\n'
        '- **Section 3 reports a size or SHA-256 mismatch, or cannot reach the Hub** — the message names the file. Delete the folder Section 3 prints as `weights_dir` and run Section 3 again (375 MB).\n'
        '- **Section 4 refuses a row group** — its decoded SHA-256 or byte total does not match `ROW_GROUP_PINS`: the cached parquet under `weights/ade20k/` is damaged (delete it) or the Hub served something else; nothing is fetched beyond the three pinned row groups.\n'
        '- **Sections 6–8 are very slow** — the runtime has no GPU: the image encoder costs about 3 s per image on CPU and the default path encodes every image several times. Switch to a GPU runtime and Run all.\n'
        '- **The frozen or adapted numbers differ from the recorded ones in the second decimal** — expected: one record per step at a small learning rate gives a noisy validation curve, and the recorded T4 and workstation runs themselves differ by a few hundredths. A verdict of `not above the centre-disk baseline` or `no gain` / `worse` is a finding to read, not an error.\n'
        '- **You re-ran Section 6 or 7 after training** — both restore the frozen decoder snapshot first, so the frozen numbers stay frozen and a second fine-tune starts from the pretrained decoder, never from the previous adaptation.\n'
        "- **Section 9's parity check fails** — the export or reload is broken; run Sections 7–9 again. Do not use the artifact.\n"
        '- **BYOD: "BYOD path … does not exist" / "the upload dialog exists only in Google Colab" / "Upload exactly one"** — set `BYOD_PATH` to a zip or a directory in the runtime (it works on Kaggle and Jupyter); on Colab an empty path opens the dialog, and a cancelled dialog stops with that message.\n'
        '- **A `ValueError` from `load_byod_dataset` or `validate_dataset`** — it names the file and the rule: a missing `<stem>_mask` beside an image, an image side outside 16..4096 px, a mask of another size, fewer than eight records, or a duplicate id.\n'
        '\n'
        '## Change one thing (next experiments)\n'
        '\n'
        "Each of these changes one default and keeps the rest of the path; Sections 6 and 7 restore the frozen decoder before they run, so the frozen numbers are the fixed reference. Set `PROMPTS = 'point'` and read whether the click gains more while the box loses more; set `PROMPTS = 'box'` and watch the point prompt barely move; raise `EPOCHS` and watch the validation score pick the epoch while the loss keeps falling; change `LEARNING_RATE` to `1e-4` and read the faster, box-costing climb the build record measured; change `SPLIT_SEED` for another draw of the 296 targets; or bring your own masks through BYOD and read the two baselines before the adapted number.\n"
        '\n'
        '## Glossary\n'
        '\n'
        "- **Promptable segmentation** — one click (a point with a foreground label) or one box in, one object's mask out; SAM returns three candidate masks at different granularities with a predicted IoU each.\n"
        "- **Predicted IoU** — the decoder's own learned ranking score for each candidate; uncalibrated, it can exceed 1.0, and the conventional rule keeps the highest.\n"
        '- **IoU / hit rate** — intersection over union of a mask with the target (1 = identical); the hit rate is the fraction of targets with IoU ≥ 0.5 (`HIT_THRESHOLD`).\n'
        "- **Box-fill / centre-disk baselines** — every pixel inside the box (no model; perfect for a rectangle) and a disk around the click with the target's own area (size given away, shape not).\n"
        "- **Target selection rule** — the largest 4-connected component of one ADE20K class covering 3..35 % of the image; the click is the mask's distance-transform maximum, the box its tight bounds.\n"
        '- **Mask decoder** — the two-way transformer, up-scaling head and IoU-prediction head (4,058,340 parameters) that the fine-tune trains; the image and prompt encoders stay frozen.\n'
        '- **Frozen decoder snapshot** — the pretrained decoder tensors Section 6 keeps once; Sections 6 and 7 restore them before scoring or training, so a re-run never builds on an earlier adaptation.\n'
        '- **BCE + soft Dice** — the training loss: binary cross-entropy on the 256 × 256 logits plus one minus the soft Dice overlap.\n'
        "- **Mixed prompts / epoch selection** — each training step uses the record's click or its box by a seeded coin; the kept epoch has the highest mean of the validation point- and box-prompt IoU.\n"
        '- **Image-disjoint split** — no image (by decoded-pixel digest) appears in two of the 180 / 45 / 70 splits.\n'
        '- **Reload parity** — the adapter written to disk, loaded onto a fresh base, reproduces the in-memory point-prompt masks pixel for pixel on eight test targets.\n'
        '- **Isolated environment** — the separate Python 3.12.12 environment Section 1 builds from the hash lock; every later cell runs there.\n'
        '- **BYOD** — bring your own data: a zip or directory of images with a `<stem>_mask.png` each, read from `BYOD_PATH` or the Colab upload dialog.\n'
        '\n'
        '## Conclusion (your notes)\n'
        '\n'
        "Before you leave, write three lines in this cell: (1) the frozen point-prompt IoU beside box-fill, and why a click is ambiguous on a wall or a floor; (2) the adapted point and box IoU and the verdicts Section 8 recorded — did the click gain, and did the box hold? (3) which class gained most and what that says about ADE20K's whole-region convention versus the objects you would click on in your own images.\n"
        '\n'
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
