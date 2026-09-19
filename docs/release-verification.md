# Release verification

`tutorials/sam_vit_segmentation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 4-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the ADE20K
  parquet-conversion commit `e660d866a1351c70bcf07925d2600b60bd6e3bc3` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `SAMViTSegmentationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus`, `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the click-outside-image refusal probe, `segment` with the sanity checks and
  the per-image `evaluation_report` on the synthetic scene, `box_fill_baseline`, `centre_disk_baseline`,
  `pipe.evaluate` on the frozen model with both prompts and the floor assertion, `pipe.adapt` with its explicit
  hyperparameters, `pipe.evaluate` on the validation and test splits after adaptation with the point-IoU assertion,
  `segment` + `evaluation_report` on the scene after adaptation, the example panels against a freshly loaded frozen
  base, `pipe.save_artifact`, `SAMViTSegmentationPipeline.from_artifact` and the mask-parity assertion, and the
  result fields `weight_file` / `weight_format` / `weight_sha256` and the `corpus` block), the eight expected
  `outputs/` paths, the learner-facing statements (Apache-2.0 weights, the uncalibrated predicted IoU that can
  exceed 1.0, adaptation with labelled targets, the frozen point prompt at 0.564 below box-fill, mixed prompts, the
  point-prompt IoU, the two prompt-only baselines, the BCE + soft-Dice loss, no dispersion estimate, the labelling
  convention and leakage guidance, the excluded tasks, the snapshot note) and the gated-off BYOD default; forbidden
  patterns (credential-in-URL, any `git clone` / `github.com/kurtvalcorza` / repository import on the primary path,
  a mutable `revision='main'`, direct `from transformers import` / `SamModel` / `SamProcessor` / `from torchvision
  import` / `from huggingface_hub import` / `urllib.request` / `pyarrow` / `scipy` / `safetensors` imports /
  `torch.optim` / `.backward(` / `requires_grad` / `pipe._model` / `extractall(` use **outside the carried module
  cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `safetensors`, `huggingface-hub`, `numpy`,
`pillow`, `scipy` and `pyarrow`, the package with `--no-deps`, runs `ruff check src tests tools`,
`tools/build_notebook.py --check`, and the unit suite (`tests/`, including `test_adaptation.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/sam-vit-base/` or the row-group cache `weights/ade20k/` (the standalone path writes the
   manifest itself, stages the missing files from the Hub and reads the pinned row groups from the Hub shard, so
   neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 6`, `LEARNING_RATE = 5e-5`, `PROMPTS = 'mixed'`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`,
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `scipy==1.18.1`, `pyarrow==25.0.1` (an interpreter restart after the
   install is expected where the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `SAMViTSegmentationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `mask_iou`, `segmentation_metrics`,
     `box_fill_baseline`, `centre_disk_baseline`, `fetch_corpus`, `read_corpus`, `select_target`, `interior_point`,
     `mask_box`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 4 manifest entries fetched from `facebook/sam-vit-base` at the immutable
     revision on a clean runtime, `verify_snapshot` returning its dict (4 files, the 375 MB `model.safetensors`
     re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory (the upstream
     `pytorch_model.bin` and TensorFlow weights are not in the manifest and must not be staged);
   - Section 4: `fetch_corpus` reading the three pinned row groups over range requests with the shard's declared
     size, every row group's SHA-256 and byte total matching; 296 targets from 300 rows over about 55 classes; the
     seeded split into 180 / 45 / 70 with `check_split_disjoint` reporting no shared image and the three dataset
     digests printed; `outputs/…_train.csv` written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 16, `MAX_IMAGE_SIDE` 4096, `MAX_PROMPTS` 16, `NUM_MULTIMASK_OUTPUTS`
     3, `MASK_THRESHOLD` 0.0, `MIN_RECORDS` 8, `MAX_RECORDS` 5000) surfaced; the synthetic scene drawn;
     `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding
     from the click-outside-image probe); `segment` on the 320×240 scene with every sanity check `True`,
     `outputs/…_mask_frozen.png` written and the per-image `evaluation_report` verdict `sample-sanity` (the
     inference-only card recorded scores `[0.955, 1.012, 0.979]` and `mask_iou` 1.000 — the scores are observations,
     not assertions);
   - Section 6: the centre-disk baseline (≈ 0.44 IoU), the box-fill baseline (≈ 0.61) and the frozen model's test
     scores (≈ 0.56 point / 0.78 box in the RTX 5070 Ti build record — the point prompt below box-fill on these
     targets) with the per-class breakdown, and the cell's assertion that the frozen box prompt is above the disk;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 4,058,340 trainable of 93,735,472 parameters, the
     embedding time, and a six-epoch history with the validation point and box IoU and their mean (build record: point 0.629 → 0.715, `best_epoch` 6);
   - Section 8: `pipe.evaluate` on the validation and test splits with the six-way comparison on both measures, the
     per-class breakdown and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test point IoU
     exceeds the frozen one — 0.691 versus 0.564 in the build record, hit rate 0.60 → 0.79, the box prompt 0.784 →
     0.788; the adapted point prompt also clears box-fill, reported, not asserted);
   - Section 9: the scene re-segmented by the adapted decoder with the `sample-sanity` report,
     `outputs/…_mask_adapted.png` and four example panels under `outputs/…_examples/` written; `pipe.save_artifact`
     writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (120 tensors, 16,248,448 bytes) and
     `SAMViTSegmentationPipeline.from_artifact` reloading it with 8/8 identical point-prompt masks on eight test
     targets (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity and
     licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block with the
     row-group pins and the target rule, the inference-contract reports, the comparison, the artifact digest, the
     reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `sam_vit_segmentation_colab.ipynb` (`E2E`) | `4ef608f` / `0112e790` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 13 files, 389 MB staged from the Hub into a clean cache; comparison {iou: {centre_disk: 0.442, box_fill: 0.614, frozen_point: 0.564, adapted_point: 0.663, frozen_box: 0.784, adapted_box: 0.778}, hit_rate: {centre_disk: 0.286, box_fill: 0.657, frozen_point: 0.6, adapted_point: 0.7, frozen_box: 0.857, adapted_box: 0.929}, delta_point_vs_frozen: {iou: 0.099, hit_rate: 0.1}, delta_point_vs_box_fill: {iou: 0.049, hit_rate: 0.043}, delta_box_vs_frozen: {iou: -0.006, hit_rate: 0.071}, by_class: {wall: {n: 13, frozen_point: 0.402, adapted_point: 0.419, frozen_box: 0.612, adapted_box: 0.669}, sky: {n: 10, frozen_point: 0.899, adapted_point: 0.93, frozen_box: 0.919, adapted_box: 0.934}, floor: {n: 8, frozen_point: 0.693, adapted_point: 0.798, frozen_box: 0.795, adapted_box: 0.782}, building: {n: 7, frozen_point: 0.373, adapted_point: 0.564, frozen_box: 0.82, adapted_box: 0.809}, road: {n: 7, frozen_point: 0.803, adapted_point: 0.832, frozen_box: 0.873, adapted_box: 0.848}, cabinet: {n: 3, frozen_point: 0.136, adapted_point: 0.421, frozen_box: 0.664, adapted_box: 0.661}, ceiling: {n: 3, frozen_point: 0.495, adapted_point: 0.614, frozen_box: 0.854, adapted_box: 0.842}, seat: {n: 3, frozen_point: 0.026, adapted_point: 0.361, frozen_box: 0.844, adapted_box: 0.763}, curtain: {n: 2, frozen_point: 0.651, adapted_point: 0.662, frozen_box: 0.593, adapted_box: 0.659}, tree: {n: 2, frozen_point: 0.185, adapted_point: 0.486, frozen_box: 0.938, adapted_box: 0.822}, animal: {n: 1, frozen_point: 0.926, adapted_point: 0.945, frozen_box: 0.926, adapted_box: 0.954}, bookcase: {n: 1, frozen_point: 0.461, adapted_point: 0.518, frozen_box: 0.488, adapted_box: 0.485}, chair: {n: 1, frozen_point: 0.052, adapted_point: 0.631, frozen_box: 0.877, adapted_box: 0.807}, computer: {n: 1, frozen_point: 0.339, adapted_point: 0.329, frozen_box: 0.731, adapted_box: 0.739}, dirt track: {n: 1, frozen_point: 0.867, adapted_point: 0.908, frozen_box: 0.499, adapted_box: 0.504}, earth: {n: 1, frozen_point: 0.821, adapted_point: 0.831, frozen_box: 0.837, adapted_box: 0.759}, grass: {n: 1, frozen_point: 0.553, adapted_point: 0.922, frozen_box: 0.851, adapted_box: 0.915}, person: {n: 1, frozen_point: 0.213, adapted_point: 0.37, frozen_box: 0.774, adapted_box: 0.625}, pier: {n: 1, frozen_point: 0.922, adapted_point: 0.88, frozen_box: 0.926, adapted_box: 0.603}, sea: {n: 1, frozen_point: 0.947, adapted_point: 0.95, frozen_box: 0.945, adapted_box: 0.955}, sidewalk: {n: 1, frozen_point: 0.931, adapted_point: 0.89, frozen_box: 0.933, adapted_box: 0.885}, water: {n: 1, frozen_point: 0.835, adapted_point: 0.875, frozen_box: 0.622, adapted_box: 0.575}}}; reload parity {identical_masks: 8, of: 8, max_pixels_differing: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-sam-vit-segmentation/v2/evidence/` in the workspace |
| `sam_vit_segmentation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `e365f33` / `0a3163dd77cf` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v1) | PASS — 8/8 code cells, 235.3 s, 10 files, 375 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/sam_vit_segmentation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/sam_vit_segmentation_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | `4ef608f` / `0112e790` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 912.1 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 13 files, 389 MB staged from the Hub into a clean cache; comparison {iou: {centre_disk: 0.442, box_fill: 0.614, frozen_point: 0.564, adapted_point: 0.663, frozen_box: 0.784, adapted_box: 0.778}, hit_rate: {centre_disk: 0.286, box_fill: 0.657, frozen_point: 0.6, adapted_point: 0.7, frozen_box: 0.857, adapted_box: 0.929}, delta_point_vs_frozen: {iou: 0.099, hit_rate: 0.1}, delta_point_vs_box_fill: {iou: 0.049, hit_rate: 0.043}, delta_box_vs_frozen: {iou: -0.006, hit_rate: 0.071}, by_class: {wall: {n: 13, frozen_point: 0.402, adapted_point: 0.419, frozen_box: 0.612, adapted_box: 0.669}, sky: {n: 10, frozen_point: 0.899, adapted_point: 0.93, frozen_box: 0.919, adapted_box: 0.934}, floor: {n: 8, frozen_point: 0.693, adapted_point: 0.798, frozen_box: 0.795, adapted_box: 0.782}, building: {n: 7, frozen_point: 0.373, adapted_point: 0.564, frozen_box: 0.82, adapted_box: 0.809}, road: {n: 7, frozen_point: 0.803, adapted_point: 0.832, frozen_box: 0.873, adapted_box: 0.848}, cabinet: {n: 3, frozen_point: 0.136, adapted_point: 0.421, frozen_box: 0.664, adapted_box: 0.661}, ceiling: {n: 3, frozen_point: 0.495, adapted_point: 0.614, frozen_box: 0.854, adapted_box: 0.842}, seat: {n: 3, frozen_point: 0.026, adapted_point: 0.361, frozen_box: 0.844, adapted_box: 0.763}, curtain: {n: 2, frozen_point: 0.651, adapted_point: 0.662, frozen_box: 0.593, adapted_box: 0.659}, tree: {n: 2, frozen_point: 0.185, adapted_point: 0.486, frozen_box: 0.938, adapted_box: 0.822}, animal: {n: 1, frozen_point: 0.926, adapted_point: 0.945, frozen_box: 0.926, adapted_box: 0.954}, bookcase: {n: 1, frozen_point: 0.461, adapted_point: 0.518, frozen_box: 0.488, adapted_box: 0.485}, chair: {n: 1, frozen_point: 0.052, adapted_point: 0.631, frozen_box: 0.877, adapted_box: 0.807}, computer: {n: 1, frozen_point: 0.339, adapted_point: 0.329, frozen_box: 0.731, adapted_box: 0.739}, dirt track: {n: 1, frozen_point: 0.867, adapted_point: 0.908, frozen_box: 0.499, adapted_box: 0.504}, earth: {n: 1, frozen_point: 0.821, adapted_point: 0.831, frozen_box: 0.837, adapted_box: 0.759}, grass: {n: 1, frozen_point: 0.553, adapted_point: 0.922, frozen_box: 0.851, adapted_box: 0.915}, person: {n: 1, frozen_point: 0.213, adapted_point: 0.37, frozen_box: 0.774, adapted_box: 0.625}, pier: {n: 1, frozen_point: 0.922, adapted_point: 0.88, frozen_box: 0.926, adapted_box: 0.603}, sea: {n: 1, frozen_point: 0.947, adapted_point: 0.95, frozen_box: 0.945, adapted_box: 0.955}, sidewalk: {n: 1, frozen_point: 0.931, adapted_point: 0.89, frozen_box: 0.933, adapted_box: 0.885}, water: {n: 1, frozen_point: 0.835, adapted_point: 0.875, frozen_box: 0.622, adapted_box: 0.575}}}; reload parity {identical_masks: 8, of: 8, max_pixels_differing: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-sam-vit-segmentation/v2/evidence/` in the workspace |
| 2026-09-20 | generated at `1f0429f` (pre-selection-rule; the executed blob differs from the committed one in the selection rule's prose and the recorded revision only) | Local WSL harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and row groups pre-staged | Default sample path, all 11 code cells, point-only epoch selection: frozen point 0.564 / box 0.784, adapt 172.6 s, epoch 3 kept (validation point IoU 0.63 → 0.72 / 0.70 / 0.75 / 0.64 / 0.66 / 0.70), adapted point 0.649 / box 0.725, reload parity 8/8 | 265.9 s | PASS — pre-flight only; the box loss led to selection on both prompts |
| 2026-09-20 | committed template at the candidate revision (blob differs from the committed one in the recorded generating revision only) | Local WSL harness (same), RTX 5070 Ti (`cuda:0`), snapshot and row groups pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (4 files), three row groups re-hashed from the cache, 296 targets split 180 / 45 / 70, four refusal probes raised, the synthetic scene segmented (selected-candidate `mask_iou` 1.000), baselines 0.442 / 0.614, frozen point 0.564 (hit 0.60) in 13.5 s and box 0.784, 180 embeddings 30.0 s, six epochs 282.8 s (validation point IoU 0.629 → 0.645 / 0.694 / 0.654 / 0.639 / 0.691 / 0.715, epoch 6 kept on the mean of both prompts), adapted point 0.691 (hit 0.786) / box 0.788 (hit 0.914), the scene re-segmented (0.982), four panels written, adapter 16,248,448 B / 120 tensors, reload parity 8/8 with 0 pixels differing, 8 outputs written | 415.1 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-14 | `e365f33` / `0a3163dd77cf` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v1) | Default sample path | 235.3 s | PASS — 8/8 ok code cells, 10 files, 375 MB staged; not evidence for the `E2E` blob |

## Current status

**Release-grade.** The `E2E` notebook blob `0112e790` (committed at `4ef608f`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 912.1 s, 13 files, 389 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

Facts a reviewer should weigh: the sample's targets are chosen by area (largest component covering 3..35 % of the
image), which makes them mostly *stuff* regions — wall, sky, floor, road, ceiling — where a single click is
ambiguous to the frozen decoder (0.564 point IoU against 0.614 for filling the box in the build record); the
adaptation's gain (to 0.691, past both prompt-only baselines) is the decoder learning ADE20K's whole-region reading
of a click and says nothing about other labelling conventions; the box prompt held (0.784 → 0.788) only because the
epoch is selected on the mean of both prompts — an earlier pre-flight that selected on the click alone lost 0.06 on
the box; the training loop is not bit-reproducible on a GPU and three runs of the recipe landed between 0.65 and
0.71 on the point prompt, so a Kaggle number a few hundredths off the build record is expected, not a finding — and the T4 run landed
there (point 0.663 with hit rate 0.70, box 0.778 with hit rate 0.929, epoch 3 kept: a fourth run inside the same
spread, the box within a hundredth of frozen); the
45-target validation split selects the epoch and the best epoch was the last of six, so the recipe is bounded by
budget rather than convergence; and the predicted IoU that ranks the candidates remains uncalibrated after
adaptation.
