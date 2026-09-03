# Weight-mirror upload record and resumable procedure

The public model repository contains the exact 123-file Git LFS subset of
`Mia-AiLab/GLM-5.3-Flash-EXL3-TR3-4bpw` at
`25a44fdbf16862a46b7cc9921142c6c81350af2f`: 120 safetensors shards plus
`model.safetensors.index.json`, `quantization_config.json`, and
`tokenizer.json`, totaling 175,715,659,341 bytes. The payloads on `refs/pr/1`
were remotely verified by path, size, and LFS SHA-256 against the pinned
manifest. `jspark3/MIRROR-COMPLETION.json` was then added to that same PR and
verified byte for byte before maintainer merge into public main.

The 21 non-LFS files in the upstream manifest are verification inputs, not
upload inputs. DFlash2 is a separate dependency and must never enter this
repository.

## Non-negotiable safety rules

- Use one durable directory on a volume with at least 200,000,000,000 bytes
  free. Keep that same directory for every retry.
- Preserve `<mirror-dir>/.cache/huggingface/`. Never delete or edit it while a
  transfer is running, and never start a second fetch or upload in parallel.
- Regular files are required outside the Hub cache metadata. The tool rejects
  symlinks, extra files, missing files, size drift, and SHA-256 drift.
- Upload only to an existing Hugging Face pull-request ref such as
  `refs/pr/7`. Never upload the weights to `main`.
- A process exiting successfully is not proof of completion. Only
  `remote-verify` proves that all 123 additions match the manifest and that the
  live metadata stayed unchanged.
- The tool does not create or merge a pull request. Merge is a separate,
  maintainer-controlled boundary after remote proof and completion metadata.

## 1. Preflight and pinned fetch

Set these once in the operator shell:

```bash
PKG=/absolute/path/to/jspark3-v1.0.0
MIRROR_DIR=/absolute/path/on-a-large-volume/jspark3-hf-mirror-25a44fdb
HF_REPO=jakejharris/jspark3
cd "$PKG"
```

The preflight is offline. It validates the 144-entry manifest contract, the
installed `hf` CLI syntax, the free-space floor, the directory inventory, and
the symlink boundary:

```bash
python3 tools/mirror_weights.py preflight --dir "$MIRROR_DIR"
```

Inspect the exact fetch command without making a network request:

```bash
python3 tools/mirror_weights.py fetch --dir "$MIRROR_DIR"
```

Start the pinned, resumable fetch only after the dry run is correct:

```bash
python3 tools/mirror_weights.py fetch --dir "$MIRROR_DIR" \
  --confirm FETCH-JSPARK3-MIRROR --no-dry-run
```

If interrupted, run that exact command again with the same directory. Do not
remove the cache or create a second mirror directory. When it exits, prove the
local result independently:

```bash
python3 tools/mirror_weights.py verify "$MIRROR_DIR"
```

Verification must report exactly 144 files and 175,715,854,754 bytes with
every size and SHA-256 matching.

## 2. Create one draft pull request

Create an empty model-repository pull request and record the number printed by
the CLI:

```bash
hf discussions create "$HF_REPO" --type model --pull-request \
  --title "Add the verified JSpark3 mirrored checkpoint" \
  --body "Pinned source and hashes are recorded in jspark3/WEIGHTS-MANIFEST.json. Do not merge until remote verification passes."
PR_REF=refs/pr/REPLACE_WITH_NUMBER
```

Confirm that `PR_REF` names the intended open pull request before continuing.
Never substitute `main`, a branch name, or a commit SHA.

## 3. Resumable, LFS-only upload

The default is a dry run. It re-hashes all 144 local files and prints an
explicit 123-path `hf upload-large-folder` allowlist:

```bash
python3 tools/mirror_weights.py upload "$MIRROR_DIR" --pr-ref "$PR_REF"
```

Start the transfer only after reviewing that command:

```bash
python3 tools/mirror_weights.py upload "$MIRROR_DIR" --pr-ref "$PR_REF" \
  --confirm UPLOAD-JSPARK3-MIRROR --no-dry-run
```

The command holds a local exclusive lock and uses the installed
`hf upload-large-folder` resume state. If interrupted, rerun the exact command
with the same directory, PR ref, and allowlist. Never run another uploader in
parallel and never delete `.cache/huggingface/`.

For a read-only progress check while the uploader is between commits:

```bash
python3 tools/mirror_weights.py remote-status --pr-ref "$PR_REF"
```

That command reports progress and rejects unexpected additions or changes to
the live metadata, but it deliberately does not declare an incomplete upload
complete.

## 4. Prove remote completion

After the uploader exits, prove the PR revision itself:

```bash
python3 tools/mirror_weights.py remote-verify --pr-ref "$PR_REF"
```

Success means the PR adds exactly the 123 allowlisted LFS paths, all remote
sizes and LFS SHA-256 values equal the pinned manifest, the 29-file metadata
parent is byte-identical, and no DFlash2 path appears.

Create a small completion receipt only after that proof succeeds, then add it
to the same PR:

```bash
RECEIPT=/absolute/path/outside-the-mirror/MIRROR-COMPLETION.json
python3 tools/mirror_weights.py completion-receipt \
  --pr-ref "$PR_REF" --out "$RECEIPT"
hf upload "$HF_REPO" "$RECEIPT" jspark3/MIRROR-COMPLETION.json \
  --repo-type model --revision "$PR_REF" \
  --commit-message "Record verified mirror completion"
python3 tools/mirror_weights.py remote-verify \
  --pr-ref "$PR_REF" --receipt "$RECEIPT"
```

The last command permits that one receipt in addition to the 123 payloads and
checks its remote bytes against the locally proven receipt.

## 5. Maintainer merge boundary

Inspect the pull-request diff one final time. Only then may the maintainer run:

```bash
PR_NUMBER=${PR_REF#refs/pr/}
hf discussions diff "$HF_REPO" "$PR_NUMBER" --type model
hf discussions merge "$HF_REPO" "$PR_NUMBER" --type model
```

Do not automate this step. After merge, record the resulting immutable Hub
revision in the release metadata and rebuild its direct SBOM and checksums.
