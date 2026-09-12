# Weight provenance and DIMER hosting

- Upstream: `facebook/sam-vit-base`
- Immutable revision: `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f`
- Weight format: SafeTensors (`model.safetensors`, 374,979,480 bytes)
- Manifest: `weights/sam-vit-base/dimer-base-manifest.json` (4 files, 374,993,239 bytes total, per-file SHA-256): `README.md`, `config.json`, `model.safetensors`, `preprocessor_config.json`. The upstream repository's `pytorch_model.bin` and TensorFlow weights are not part of the manifest and are not staged.
- Upstream weight license: Apache-2.0
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the license and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `SamModel` / `SamProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory, float32 (`dtype=torch.float32`, matching the snapshot's `torch_dtype`).
