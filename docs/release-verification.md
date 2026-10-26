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
  `tools/build_notebook.py` output for its recorded revision; the isolated-environment bootstrap cell (generator
  /2.2: hash-locked `uv` environment, nothing installed into the kernel, no restart); `NOTEBOOK_SOURCE` recorded in exports;
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
   `pillow==11.3.0`, `huggingface-hub==0.36.2`, `scipy==1.18.1`, `pyarrow==25.0.1` (installed into the isolated
   hash-locked environment of Section 1, so the runtime's preinstalled torch or numpy do not matter and no restart is needed);
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
| `sam_vit_segmentation_colab.ipynb` (`E2E`, `GUIDED`, spec 2.5) | `fa91417` / `2b2e4b2b` | 2026-10-25 | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | **PASSED in one pass, no restart, 0 errors** — 12/12 code cells, 771.3 s wall; default settings only; adapted point 0.616 / box 0.768, epoch 2 kept; see Recorded executions; evidence in `docs/execution-evidence/2026-10-25-fa91417/` |
| `sam_vit_segmentation_colab.ipynb` (`E2E`) | `4ef608f` / `0112e790` | 2026-09-20 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 13 files, 389 MB staged from the Hub into a clean cache; comparison {iou: {centre_disk: 0.442, box_fill: 0.614, frozen_point: 0.564, adapted_point: 0.663, frozen_box: 0.784, adapted_box: 0.778}, hit_rate: {centre_disk: 0.286, box_fill: 0.657, frozen_point: 0.6, adapted_point: 0.7, frozen_box: 0.857, adapted_box: 0.929}, delta_point_vs_frozen: {iou: 0.099, hit_rate: 0.1}, delta_point_vs_box_fill: {iou: 0.049, hit_rate: 0.043}, delta_box_vs_frozen: {iou: -0.006, hit_rate: 0.071}, by_class: {wall: {n: 13, frozen_point: 0.402, adapted_point: 0.419, frozen_box: 0.612, adapted_box: 0.669}, sky: {n: 10, frozen_point: 0.899, adapted_point: 0.93, frozen_box: 0.919, adapted_box: 0.934}, floor: {n: 8, frozen_point: 0.693, adapted_point: 0.798, frozen_box: 0.795, adapted_box: 0.782}, building: {n: 7, frozen_point: 0.373, adapted_point: 0.564, frozen_box: 0.82, adapted_box: 0.809}, road: {n: 7, frozen_point: 0.803, adapted_point: 0.832, frozen_box: 0.873, adapted_box: 0.848}, cabinet: {n: 3, frozen_point: 0.136, adapted_point: 0.421, frozen_box: 0.664, adapted_box: 0.661}, ceiling: {n: 3, frozen_point: 0.495, adapted_point: 0.614, frozen_box: 0.854, adapted_box: 0.842}, seat: {n: 3, frozen_point: 0.026, adapted_point: 0.361, frozen_box: 0.844, adapted_box: 0.763}, curtain: {n: 2, frozen_point: 0.651, adapted_point: 0.662, frozen_box: 0.593, adapted_box: 0.659}, tree: {n: 2, frozen_point: 0.185, adapted_point: 0.486, frozen_box: 0.938, adapted_box: 0.822}, animal: {n: 1, frozen_point: 0.926, adapted_point: 0.945, frozen_box: 0.926, adapted_box: 0.954}, bookcase: {n: 1, frozen_point: 0.461, adapted_point: 0.518, frozen_box: 0.488, adapted_box: 0.485}, chair: {n: 1, frozen_point: 0.052, adapted_point: 0.631, frozen_box: 0.877, adapted_box: 0.807}, computer: {n: 1, frozen_point: 0.339, adapted_point: 0.329, frozen_box: 0.731, adapted_box: 0.739}, dirt track: {n: 1, frozen_point: 0.867, adapted_point: 0.908, frozen_box: 0.499, adapted_box: 0.504}, earth: {n: 1, frozen_point: 0.821, adapted_point: 0.831, frozen_box: 0.837, adapted_box: 0.759}, grass: {n: 1, frozen_point: 0.553, adapted_point: 0.922, frozen_box: 0.851, adapted_box: 0.915}, person: {n: 1, frozen_point: 0.213, adapted_point: 0.37, frozen_box: 0.774, adapted_box: 0.625}, pier: {n: 1, frozen_point: 0.922, adapted_point: 0.88, frozen_box: 0.926, adapted_box: 0.603}, sea: {n: 1, frozen_point: 0.947, adapted_point: 0.95, frozen_box: 0.945, adapted_box: 0.955}, sidewalk: {n: 1, frozen_point: 0.931, adapted_point: 0.89, frozen_box: 0.933, adapted_box: 0.885}, water: {n: 1, frozen_point: 0.835, adapted_point: 0.875, frozen_box: 0.622, adapted_box: 0.575}}}; reload parity {identical_masks: 8, of: 8, max_pixels_differing: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-sam-vit-segmentation/v2/evidence/` in the workspace |
| `sam_vit_segmentation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `e365f33` / `0a3163dd77cf` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v1) | PASS — 8/8 code cells, 235.3 s, 10 files, 375 MB staged; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/sam_vit_segmentation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/sam_vit_segmentation_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-26 | `b94f23e` / `b1554148` (PR #10 head; final prose pass, regenerated from `05ffb3c`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `b94f23e8293381157c355d783aa78bd49c835b04`, git blob SHA-1 `b15541485fac09260eb921c27830ead50dba900a` checked, input unchanged; captured executed-notebook SHA-256 `a16786ec9f778f913488925f9be2730c6eb1a905e4ac03b3d931c245d4972844` equals `run_summary.json` `executed_sha256` | 727.1 s | **PASSED in one pass, no restart, 0 errors** — 12/12 code cells (`exec.log`); isolated environment built in 56 s; test (70 targets): centre disk 0.442, box-fill 0.614, frozen point 0.564 / box 0.784 → adapted point 0.644 / box 0.757 (Δ point 0.08, Δ vs box-fill 0.03, Δ box −0.027); validation point IoU 0.629 → 0.666 / 0.578 / 0.701 / 0.645 / 0.682 / 0.703, epoch 6 kept, training loss 0.3925 → 0.2877; verdicts `improved`, `above box-fill`, `worse`; adapter 120 tensors, 16,248,448 bytes, reload parity 8/8 (0 pixels differing); per class the click gained on building, cabinet, seat, tree and wall (wall 0.402 → 0.504, seat 0.026 → 0.194) and the box lost on tree, seat and person. Drift against the prose's recorded values (recorded, not fixed): wall 727.1 s, training loss 0.3925 → 0.2877, adapted box 0.757 (see Current status). Evidence in `docs/execution-evidence/2026-10-26-b94f23e/` |
| 2026-10-25 | `4f286ce` / `0dcc483e` (PR #10 head; historical wording, regenerated from `0c6e78c`; confirming run) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default sample path, no overrides; notebook bytes fetched at the full commit SHA `4f286cefabcc45e3ac07a0094694665f98cde609`, git blob SHA-1 `0dcc483edfdb71bd57e42b077db66c67c3f8bae5` checked, input unchanged; captured executed-notebook SHA-256 `fa9f25aa1fd6c85572a5a0c0a38b3dd02cb8f4fc17c429769661d767f29d46be` equals `run_summary.json` `executed_sha256` | 737.6 s | **PASSED in one pass, no restart, 0 errors** — 12/12 code cells (`exec.log`); isolated environment built in 56 s; test (70 targets): centre disk 0.442, box-fill 0.614, frozen point 0.564 / box 0.784 → adapted point 0.682 / box 0.769 (Δ point +0.117, Δ vs box-fill +0.068, Δ box −0.015); validation point IoU 0.629 → 0.636 / 0.689 / 0.71 / 0.687 / 0.641 / 0.696, epoch 6 kept, training loss 0.4003 → 0.277; verdicts `improved`, `above box-fill`, `worse`; adapter 120 tensors, 16,248,448 bytes, reload parity 8/8; per class wall 0.402 → 0.505, seat 0.026 → 0.386, the box lost on tree, seat and person. Drift against the prose's recorded values (not fixed, per the 2026-10-26 ruling): training loss 0.4003 → 0.277, validation point IoU 0.696 at epoch six (see Current status). Evidence in `docs/execution-evidence/2026-10-25-4f286ce/` |
| 2026-10-25 | `1088a3e` / `9a507580` (PR #10 head; prose kept true across runs, regenerated from `eef4e67`; downloaded from GitHub at the PR head and blob-verified before the session) | Colab CLI sequential execution, fresh Colab Tesla T4, via the workspace `colab-cli-serial-test-suite`. Not a browser Run all; the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (1/12–12/12) | Default sample path, no overrides, input unchanged | 762.2 s session wall | **PASSED — one pass, no restart, 0 errors, 12/12 code cells.** Isolated uv CPython 3.12.12 environment, 49 locked packages, 53 s. Synthetic scene predicted IoUs `[0.957, 1.0116, 0.982]`. Test (70 targets): centre disk 0.442, box-fill 0.614, frozen point 0.564 / box 0.784 → adapted point 0.685 / box 0.793 (Δ point +0.121, Δ vs box-fill +0.071, Δ box +0.009); validation point IoU 0.629 → 0.681 / 0.709 / 0.667 / 0.625 / 0.671 / 0.712, epoch 6 kept, training loss 0.3806 → 0.2573; verdicts `improved`, `above box-fill`, `held or improved`; per class wall 0.402 → 0.515, seat 0.026 → 0.428, the box lost on tree, seat and person; adapter 120 tensors, 16,248,448 bytes, reload parity 8/8. Prose check (confirming run): the adapted point IoU 0.685, its gains (+0.121, +0.071), the final validation point IoU 0.712, epoch 6, the wall gain 0.515, the scene IoUs and the 762.2 s timing lie inside the stated values and ranges. **Prose mismatch** (notebook not edited): (1) the prose says the box prompt slipped in every recorded hosted run, 0.784 → 0.768 to 0.779, with the verdict `worse` for the box in every run; this run's box rose 0.784 → 0.793 (+0.009, hit rate 0.857 → 0.943) with the verdict `held or improved`; (2) the per-class seat range 0.026 → 0.067 to 0.361; this run printed seat 0.026 → 0.428. Not exercised: BYOD on a hosted runtime. Evidence `docs/execution-evidence/2026-10-25-1088a3e/sam_vit_segmentation_colab_1088a3e_colab-cli-t4_default.ipynb` (sha256 `fa6c35746b76ccaa26dc87e5dcf291944962e9eac53acdc25eb6a14d7d06e25b`; table under [Current status](#current-status)). Status stays **Candidate** |
| 2026-10-25 | `fa91417` / `2b2e4b2b` (PR #10 head) | Colab CLI sequential execution, fresh Colab Tesla T4 VM (`colab exec -f` on a fresh session; fresh VM, empty Hugging Face cache, no repository checkout) — not a browser `Run all`; no execution counts, order from `exec.log` | Default path, every form field at its default, no overrides; notebook bytes fetched at the full commit SHA `fa9141780642d9747ae31da61254061c59d01cb6`, git blob SHA-1 `2b2e4b2b15929dc168b7b109eb86424ec1092e7f` checked, input unchanged; captured executed-notebook SHA-256 `cc4084f9f92a7cd8a71a951df5d96ba0ce60806e2dd359d5468047a37a4093e2` equals `run_summary.json` `executed_sha256` | 771.3 s | **PASSED in one pass, no restart, 0 errors** — 12/12 code cells (`exec.log`); isolated environment built in 48 s (CPython 3.12.12 from the 49-package lock, kernel Python 3.13.16, `cuda:0`); splits 180 / 45 / 70 targets; two prompt refusals; synthetic scene predicted IoUs `[0.957, 1.0116, 0.982]` (best candidate 1); test baselines centre disk 0.442 (hit 0.286), box-fill 0.614 (hit 0.657); frozen point 0.564 (hit 0.6), frozen box 0.784 (hit 0.857); validation point IoU 0.629 → 0.640 / 0.717 / 0.694 / 0.622 / 0.654 / 0.701, train loss 0.3936 → 0.26, `best_epoch` 2 (highest mean of validation point- and box-prompt IoU; 4,058,340 of 93,735,472 parameters); adapted point 0.616 (hit 0.629), adapted box 0.768 (hit 0.886); `delta_point_vs_frozen` IoU 0.051, `delta_point_vs_box_fill` 0.002, `delta_box_vs_frozen` −0.016; verdicts `above the centre-disk baseline`, `improved`, `above box-fill`, `worse` (box); `reload_parity` `{'identical_masks': 8, 'of': 8, 'max_pixels_differing': 0}`, `reloaded_best_epoch` 2. Not exercised: BYOD (REL12) and a browser `Run all` (REL2). Evidence in `docs/execution-evidence/2026-10-25-fa91417/` |
| 2026-10-09 | `d7091e2` / `edfa7ad65d07` (`NOTEBOOK_SOURCE.repository_revision` `53d9637`, the source revision the notebook was generated from; `53d9637..d7091e2` is the review-fix commit; generator `build_notebook.py/2.2`, `notebook_spec` 2.2) | Colab CLI 0.7.4 sequential execution (`colab exec -f`, not a browser Run all; order from `exec.log`, no execution counts), fresh Colab Tesla T4 VM (session `suite-sam-d7091e2-3c7f`), committed blob fetched at the 40-char SHA and Git-blob verified, no repository checkout, clean model cache, BYOD gates off, sample-path defaults; kernel Python 3.13.15; isolated uv environment Python 3.12.12, 49 locked packages, built in 57 s; torch 2.14.0+cu130, transformers 4.57.6; device `cuda:0` | Default E2E path, all 12 code cells (the three carried-module cells print nothing): snapshot verified (4 files), 300 rows → 296 targets over 55 classes split 180 / 45 / 70, four refusal probes rejected, the synthetic scene segmented (scores [0.957, 1.0116, 0.982], selected `mask_iou` 1.0), baselines 0.442 / 0.614, frozen point 0.564 (hit 0.60) / box 0.784 (hit 0.857), six epochs with validation point IoU 0.629 → 0.700 / 0.723 / 0.708 / 0.623 / 0.664 / 0.744 and epoch 6 kept (462.6 s), adapted point 0.688 (hit 0.743) / box 0.779 (hit 0.929), verdicts improved / above box-fill / box worse by 0.005, four panels written, adapter 16,248,448 B / 120 tensors, reload parity 8/8 with 0 pixels differing, 8 outputs | 738.7 s | **PASS** — one pass, no restart, 0 errors; executed notebook, `run_summary.json` and `exec.log` in `docs/execution-evidence/2026-10-09-d7091e2/` (executed notebook SHA-256 `d7a3ba33f3ab2b0b2abefc843bc398f4f833229d8d72231d97f0bebeed9aa10f`). Optional journeys (BYOD, the change-one-thing activity) not exercised |
| 2026-09-20 | `4ef608f` / `0112e790` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 912.1 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 13 files, 389 MB staged from the Hub into a clean cache; comparison {iou: {centre_disk: 0.442, box_fill: 0.614, frozen_point: 0.564, adapted_point: 0.663, frozen_box: 0.784, adapted_box: 0.778}, hit_rate: {centre_disk: 0.286, box_fill: 0.657, frozen_point: 0.6, adapted_point: 0.7, frozen_box: 0.857, adapted_box: 0.929}, delta_point_vs_frozen: {iou: 0.099, hit_rate: 0.1}, delta_point_vs_box_fill: {iou: 0.049, hit_rate: 0.043}, delta_box_vs_frozen: {iou: -0.006, hit_rate: 0.071}, by_class: {wall: {n: 13, frozen_point: 0.402, adapted_point: 0.419, frozen_box: 0.612, adapted_box: 0.669}, sky: {n: 10, frozen_point: 0.899, adapted_point: 0.93, frozen_box: 0.919, adapted_box: 0.934}, floor: {n: 8, frozen_point: 0.693, adapted_point: 0.798, frozen_box: 0.795, adapted_box: 0.782}, building: {n: 7, frozen_point: 0.373, adapted_point: 0.564, frozen_box: 0.82, adapted_box: 0.809}, road: {n: 7, frozen_point: 0.803, adapted_point: 0.832, frozen_box: 0.873, adapted_box: 0.848}, cabinet: {n: 3, frozen_point: 0.136, adapted_point: 0.421, frozen_box: 0.664, adapted_box: 0.661}, ceiling: {n: 3, frozen_point: 0.495, adapted_point: 0.614, frozen_box: 0.854, adapted_box: 0.842}, seat: {n: 3, frozen_point: 0.026, adapted_point: 0.361, frozen_box: 0.844, adapted_box: 0.763}, curtain: {n: 2, frozen_point: 0.651, adapted_point: 0.662, frozen_box: 0.593, adapted_box: 0.659}, tree: {n: 2, frozen_point: 0.185, adapted_point: 0.486, frozen_box: 0.938, adapted_box: 0.822}, animal: {n: 1, frozen_point: 0.926, adapted_point: 0.945, frozen_box: 0.926, adapted_box: 0.954}, bookcase: {n: 1, frozen_point: 0.461, adapted_point: 0.518, frozen_box: 0.488, adapted_box: 0.485}, chair: {n: 1, frozen_point: 0.052, adapted_point: 0.631, frozen_box: 0.877, adapted_box: 0.807}, computer: {n: 1, frozen_point: 0.339, adapted_point: 0.329, frozen_box: 0.731, adapted_box: 0.739}, dirt track: {n: 1, frozen_point: 0.867, adapted_point: 0.908, frozen_box: 0.499, adapted_box: 0.504}, earth: {n: 1, frozen_point: 0.821, adapted_point: 0.831, frozen_box: 0.837, adapted_box: 0.759}, grass: {n: 1, frozen_point: 0.553, adapted_point: 0.922, frozen_box: 0.851, adapted_box: 0.915}, person: {n: 1, frozen_point: 0.213, adapted_point: 0.37, frozen_box: 0.774, adapted_box: 0.625}, pier: {n: 1, frozen_point: 0.922, adapted_point: 0.88, frozen_box: 0.926, adapted_box: 0.603}, sea: {n: 1, frozen_point: 0.947, adapted_point: 0.95, frozen_box: 0.945, adapted_box: 0.955}, sidewalk: {n: 1, frozen_point: 0.931, adapted_point: 0.89, frozen_box: 0.933, adapted_box: 0.885}, water: {n: 1, frozen_point: 0.835, adapted_point: 0.875, frozen_box: 0.622, adapted_box: 0.575}}}; reload parity {identical_masks: 8, of: 8, max_pixels_differing: 0}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-sam-vit-segmentation/v2/evidence/` in the workspace |
| 2026-09-20 | generated at `1f0429f` (pre-selection-rule; the executed blob differs from the committed one in the selection rule's prose and the recorded revision only) | Local WSL harness (`run_nb_local.py`: nbclient, fresh `python3` kernel, `CUDA_VISIBLE_DEVICES=0`, `HF_HUB_OFFLINE=1`, `DIMER_NOTEBOOK_CI_PREINSTALLED=1`), Python 3.12.3, torch 2.14.0+cu130, RTX 5070 Ti (`cuda:0`), snapshot and row groups pre-staged | Default sample path, all 11 code cells, point-only epoch selection: frozen point 0.564 / box 0.784, adapt 172.6 s, epoch 3 kept (validation point IoU 0.63 → 0.72 / 0.70 / 0.75 / 0.64 / 0.66 / 0.70), adapted point 0.649 / box 0.725, reload parity 8/8 | 265.9 s | PASS — pre-flight only; the box loss led to selection on both prompts |
| 2026-09-20 | committed template at the candidate revision (blob differs from the committed one in the recorded generating revision only) | Local WSL harness (same), RTX 5070 Ti (`cuda:0`), snapshot and row groups pre-staged | Default sample path, all 11 code cells: pinned install skipped (pre-installed), `stage_missing_files` reported nothing to fetch, `verify_snapshot` PASS (4 files), three row groups re-hashed from the cache, 296 targets split 180 / 45 / 70, four refusal probes raised, the synthetic scene segmented (selected-candidate `mask_iou` 1.000), baselines 0.442 / 0.614, frozen point 0.564 (hit 0.60) in 13.5 s and box 0.784, 180 embeddings 30.0 s, six epochs 282.8 s (validation point IoU 0.629 → 0.645 / 0.694 / 0.654 / 0.639 / 0.691 / 0.715, epoch 6 kept on the mean of both prompts), adapted point 0.691 (hit 0.786) / box 0.788 (hit 0.914), the scene re-segmented (0.982), four panels written, adapter 16,248,448 B / 120 tensors, reload parity 8/8 with 0 pixels differing, 8 outputs written | 415.1 s | PASS — pre-flight only; not promotion evidence |
| 2026-09-14 | `e365f33` / `0a3163dd77cf` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-sam-vit-segmentation` v1) | Default sample path | 235.3 s | PASS — 8/8 ok code cells, 10 files, 375 MB staged; not evidence for the `E2E` blob |

## Current status

**DAT20 refusal display (2026-10-26): Candidate.** The current blob `49bdcbfa` (code commit `340a1e5`) shows a refused BYOD input as one `Refused: <file>: <reason>` line with no traceback, removes the earlier run's results from the session and this notebook's exports from `outputs/`, and stops Run all at that cell, while genuine failures keep their traceback (NOTEBOOK_SPEC DAT19/DAT20); its CPU timings now name their source run; its hosted Colab T4 runs (the default path and a scripted refusal, `USE_BYOD=True;BYOD_PATH='/content/sample_data'`) are pending, and the hosted runs of earlier blobs, including the 2026-10-26 run of `b15541485fac`, are history for it (REL14). The status below is history.

**Candidate (2026-10-26, final prose pass).** The current notebook blob is `b15541485fac09260eb921c27830ead50dba900a` (commit `ed55816`), regenerated from `05ffb3c` under Kurt's 2026-10-26 ruling to fix only the prose-rule items flagged on the confirming run of `0dcc483e`: the Section 7 sentence that said the same recipe lands within a few hundredths of the numbers from one GPU run to the next is removed, and the Prerequisites sentence 'No CPU run of the whole path has been recorded' is removed (the labelled CPU estimate stays). Markdown only; no code or output changed, and the drift recorded for that run below is kept, not fixed. This exact blob (PR #10 head `b94f23e`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-26, 12/12 code cells in one pass, no restart, 0 errors, 727.1 s wall; adapted point 0.644 / box 0.757, epoch 6 kept, reload parity 8/8; evidence in `docs/execution-evidence/2026-10-26-b94f23e/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The confirming run of `0dcc483e` and the earlier records below stay as history.

**Drift recorded, not fixed (2026-10-26, final prose pass; Kurt's ruling: drift is recorded here, never fixed).** The notebook prose of blob `b1554148` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Header and How to use: the recorded runs of the default path 'took between 12.3 and 15.2 minutes (738.7 s to 912.1 s across the four recorded hosted runs)'; this run took 727.1 s.
- Section 7: training loss 'from about 0.38–0.39 to 0.26' in the three named hosted runs; this run printed 0.3925 at epoch 1 and 0.2877 at epoch 6.
- Section 8: box prompt '0.784 → 0.768 to 0.793' (−0.016 to +0.009), 'within about 0.02 of frozen in each' of the named runs; this run printed adapted box 0.757 (Δ −0.027).

Evidence files for the 2026-10-26 default run of blob `b1554148`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-26-b94f23e/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `sam_vit_segmentation_colab_b94f23e_colab-cli-t4_default.ipynb` | `a16786ec9f778f913488925f9be2730c6eb1a905e4ac03b3d931c245d4972844` | `a16786ec9f778f913488925f9be2730c6eb1a905e4ac03b3d931c245d4972844` (unchanged, no local path) |
| `exec.log` | `2f8b95af557e3e3a4165428853489556694fd0d5503ba4c42644e4672ce27f49` | `b2af102f5be93039dab7e328a736813db503563fad51398b413062784435b6e6` (2 local-path replacements) |
| `run_summary.json` | `b00bc5a00714b0b6f55b48ab293c6ada69eb28ee795ed6cd18e8545c59e2e96e` | `b00bc5a00714b0b6f55b48ab293c6ada69eb28ee795ed6cd18e8545c59e2e96e` (unchanged, no local path) |

**Previous candidate (2026-10-26, historical wording).** The previous notebook blob is `0dcc483edfdb71bd57e42b077db66c67c3f8bae5` (commit `76c8e1f`), regenerated from `0c6e78c` under Kurt's 2026-10-26 ruling: every run-derived number in the prose is a record of the named hosted runs that printed it (Colab T4 of `edfa7ad6` 2026-10-09, `2b2e4b2b` 2026-10-25 and `9a507580` 2026-10-25; Kaggle T4 of `0112e790` 2026-09-20), with no expectation ranges, no generalised directions and no timing promise; it resolves the two prose mismatches of the `9a507580` run below (box 0.784 → 0.768 to 0.793, fell in three runs and rose in one; seat 0.067 to 0.428). Markdown and one error-message string only. This exact blob (PR #10 head `4f286ce`) **passed** its confirming hosted run: Colab CLI sequential execution on a fresh Colab Tesla T4, 2026-10-25, 12/12 code cells in one pass, no restart, 0 errors, 737.6 s wall; adapted point 0.682 / box 0.769, epoch 6 kept, reload parity 8/8; evidence in `docs/execution-evidence/2026-10-25-4f286ce/`. This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser Run all). Runs of earlier blobs are history (REL14). The records below stay as history.

**Drift recorded, not fixed (2026-10-26, Kurt's 2026-10-26 "historical wording, then stop" ruling).** The notebook prose of blob `0dcc483e` labels every run-derived number with the named recorded runs that printed it, so these differences are not prose errors. This confirming run printed these values outside the recorded values and spans the prose names. The notebook is not edited:

- Section 7: training loss 'from about 0.38–0.39 to 0.26' in the three named hosted runs; this run printed 0.4003 at epoch 1 and 0.277 at epoch 6.
- Section 7: validation point IoU 'between 0.701 and 0.744 at epoch six' in the three named hosted runs; this run printed 0.696 at epoch six.

Evidence files for the 2026-10-25 default run of blob `0dcc483e`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-4f286ce/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `sam_vit_segmentation_colab_4f286ce_colab-cli-t4_default.ipynb` | `fa9f25aa1fd6c85572a5a0c0a38b3dd02cb8f4fc17c429769661d767f29d46be` | `fa9f25aa1fd6c85572a5a0c0a38b3dd02cb8f4fc17c429769661d767f29d46be` (unchanged, no local path) |
| `exec.log` | `d127d4b0a717503600126cdd84fe1a4bd2a7d6008ebdeaf5c283d7a4a9149023` | `42ec6dacb6805ec3dff0cf54f9b1b2dd4037442260b5abce68ed6bb90885b276` (2 local-path replacements) |
| `run_summary.json` | `eb322e40e2e068341eb4b562e357d417e4ad3e6a2d96e841a553f5121ab9e162` | `eb322e40e2e068341eb4b562e357d417e4ad3e6a2d96e841a553f5121ab9e162` (unchanged, no local path) |

**Previous candidate (2026-10-25, prose kept true across runs; hosted run recorded).** The previous notebook blob is `9a507580af359eaf011ccd081d98b70a45a348d4` (commit `181f1a3`, PR #10 head `1088a3e`), regenerated from `eef4e67`; markdown only. That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 12/12 code cells, 762.2 s wall; adapted point 0.685 / box 0.793 against frozen 0.564 / 0.784 and box-fill 0.614, epoch 6 kept; reload parity 8/8; first row under "Recorded executions"). Most of the revised prose holds for this run (point IoU, gains, kept epoch, timing), but two statements do not: the box prompt improved here (0.784 → 0.793, verdict `held or improved`) where the prose says it slipped in every recorded hosted run, and seat rose to 0.428, above the stated 0.067–0.361 (see the table row). This PR's runs cover the default path only. Remaining gates: REL12 (a hosted BYOD run with one accepted and one refused input) and REL2 (a browser run of the interactive journeys). The 2026-10-25 run of the previous blob `2b2e4b2b` and the runs of earlier blobs below are history (REL14).

Evidence files, redacted byte-wise (the local run directory became `<local-run-dir>`, the Colab CLI install path `<colab-cli-tool-dir>`; no JSON re-serialised):

| File (under `docs/execution-evidence/2026-10-25-1088a3e/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `sam_vit_segmentation_colab_1088a3e_colab-cli-t4_default.ipynb` | `fa6c35746b76ccaa26dc87e5dcf291944962e9eac53acdc25eb6a14d7d06e25b` | `fa6c35746b76ccaa26dc87e5dcf291944962e9eac53acdc25eb6a14d7d06e25b` (unchanged, no local path) |
| `exec.log` | `f44f6d0d4b1c71067cdc39e51443317b11cb016b37bb9c790b3e4eb3bc3c83ac` | `8535922b34cfd15b01a9db5e97e3366d01e220c59985611ff69e6ac90369e09f` (2 local-path replacements) |
| `run_summary.json` | `ab1d567cc40e57e7c480b53664b35c0f120b14e07c3a92f044aeb607ab132937` | `ab1d567cc40e57e7c480b53664b35c0f120b14e07c3a92f044aeb607ab132937` (unchanged, no local path) |

**Previous candidate (2026-10-25).** The previous notebook blob is `2b2e4b2b15929dc168b7b109eb86424ec1092e7f` (commit `fa91417`), generated at `00c0804` for the tier C review fixes (T78-1 worked answers quote the labelled hosted Colab T4 run of `edfa7ad6`; T78-2 DAT9 overlap statement; T78-3 dated pre-flight; T78-4 spec 2.5 items); it changes markdown, the use notice, `metadata.dimer` and the `NOTEBOOK_SOURCE` record only. That exact blob completed one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-25 (Colab CLI sequential execution, 12/12 code cells, 771.3 s wall; frozen point 0.564 / box 0.784 against box-fill 0.614 and centre disk 0.442; adapted point 0.616 / box 0.768, epoch 2 kept; reload parity 8/8 with 0 pixels differing; evidence in `docs/execution-evidence/2026-10-25-fa91417/`). The paths this blob's runs cover: the default path only. The remaining gates are REL12 (a hosted BYOD run) and REL2 (a browser `Run all`). The runs of earlier blobs below are history (REL14).

Evidence files for the 2026-10-25 default run of blob `2b2e4b2b`, copied from the run directory and redacted byte-wise (local run paths replaced by `<local-run-dir>`, the Colab CLI install path by `<colab-cli-tool-dir>`; no JSON re-serialised; the executed notebook's captured hash equals `run_summary.json` `executed_sha256`):

| File (under `docs/execution-evidence/2026-10-25-fa91417/`) | Captured SHA-256 | Committed SHA-256 |
|---|---|---|
| `sam_vit_segmentation_colab_fa91417_colab-cli-t4_default.ipynb` | `cc4084f9f92a7cd8a71a951df5d96ba0ce60806e2dd359d5468047a37a4093e2` | `cc4084f9f92a7cd8a71a951df5d96ba0ce60806e2dd359d5468047a37a4093e2` (unchanged, no local path) |
| `exec.log` | `f0da493f4b8210a6beb095eb2bb60c31b93be5a5dc9c2a3f7ff20f94fa41bd6e` | `c3710fa56683ce0f78d7bf8f5ce6c6c1e16f5f9743969b06ebc0ffaef5cd732e` (2 local-path replacements) |
| `run_summary.json` | `12791e7b32149a3813d04b9a081a8ff83a27cebd842c9bb7422658a0813f293d` | `12791e7b32149a3813d04b9a081a8ff83a27cebd842c9bb7422658a0813f293d` (unchanged, no local path) |

**Prose check (2026-10-25, not fixed in this evidence commit; fixed for blob `9a507580` at `eef4e67`).** The notebook prose was compared number by number with what this run printed. These quoted figures differ from this run's output (the notebook is unchanged here; a later generator change can re-quote them):

- Section 5: the inference-only card's predicted IoUs `[0.955, 1.012, 0.979]` (labelled as the inference-only card); this run printed `[0.957, 1.0116, 0.982]` (best candidate 1, one score above 1.0).
- Section 7 worked answers: validation point IoU 0.629 → 0.700 / 0.723 / 0.708 / 0.623 / 0.664 / 0.744 with epoch 6 kept and training loss 0.39 → 0.26 (labelled as the hosted run of `edfa7ad6`); this run printed 0.629 → 0.640 / 0.717 / 0.694 / 0.622 / 0.654 / 0.701, kept epoch 2 (highest mean of validation point- and box-prompt IoU) and training loss 0.3936 → 0.26.
- Section 8: the stated expectation 'roughly 0.65–0.71' for the adapted point-prompt IoU and a box prompt 'within about 0.01 of frozen'; this run printed adapted point 0.616 (hit rate 0.6 → 0.629), only 0.002 above box-fill's 0.614, and box 0.784 → 0.768 (−0.016; hit rate 0.857 → 0.886). The verdicts were the same (`improved`, `above box-fill`, `worse`).
- Section 8 worked answer and Interpretation: point 0.564 → 0.688 (+0.124), box 0.784 → 0.779 and the per-class gains (building 0.373 → 0.637, cabinet 0.136 → 0.458, seat 0.026 → 0.312, tree 0.185 → 0.679, wall 0.402 → 0.548) are labelled as the run of `edfa7ad6`; this run's per-class point IoU moved much less (for example wall 0.402 → 0.415).
- Budget line: '912 s in the recorded run'; this run took 771.3 s of wall time, downloads included.


**Candidate** — the previous notebook blob `edfa7ad65d07` (commit `d7091e2`) completed Run all in one pass with no restart and 0 errors on a fresh Colab Tesla T4 on 2026-10-09 (Colab CLI 0.7.4 sequential execution, 12/12 code cells, 738.7 s wall; frozen point 0.564 / box 0.784 against box-fill 0.614 and centre disk 0.442; adapted point 0.688 / box 0.779, epoch 6 kept; reload parity 8/8 with 0 pixels differing); it is recorded in `docs/release-verification.md` with the executed notebook under `docs/execution-evidence/2026-10-09-d7091e2/`. The notebook was regenerated for the review fixes (SAM-M1..M5, SAM-m1..m3): an isolated uv environment from a hash lock (no in-kernel install, no restart), a frozen-decoder restore on re-runs, a named BYOD split minimum (50 distinct images), sample-path-only expectation checks, and the guided layer. Hosted BYOD (REL12) evidence is pending, so the status stays Candidate (REL14). The previous blob `0112e790` (committed at `4ef608f`) passed a clean Kaggle Tesla T4 run on 2026-09-20 (11/11 ok after one restart, 912.1 s).

History: the `E2E` notebook blob `0112e790` (committed at `4ef608f`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-20 (11/11 ok (1 restart after install cell), 912.1 s, 13 files, 389 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.

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
