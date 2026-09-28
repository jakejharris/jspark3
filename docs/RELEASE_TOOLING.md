# Release packaging maintenance

The release assembler is maintained separately from this source export.
[`release_packaging.patch`](../tools/release_packaging.patch) records the changes
to its `tools/render.py`, `tools/package.py`, `tools/comparison_variants.py`,
`tools/hf_upload.py`, `tools/publish_checks.py`, `tools/publish_release.py`,
its headline/package regression tests and
`9AM-COMMANDS.sh`. Apply it to the assembler checkout, not to this source tree:

```sh
git -C "$PACKAGING_DIR" apply --check "$SOURCE_DIR/tools/release_packaging.patch"
git -C "$PACKAGING_DIR" apply "$SOURCE_DIR/tools/release_packaging.patch"
python3 -B "$SOURCE_DIR/tools/test_release_tooling.py" --packaging-root "$PACKAGING_DIR"
python3 -B "$SOURCE_DIR/tools/test_publish_retry.py" --packaging-root "$PACKAGING_DIR"
```

The renderer and validator require the model card's ShapleyMcg v1.0 metadata
(`license: other`, `license_name: shapleymcg-license-1.0`, `license_link: LICENSE`).
The recipe's component licenses remain separate. The uploader defaults to
`HF_HUB_DISABLE_XET=1` before importing the HF SDK; the publication shell sets
the same default.

The publisher first validates and dry-runs the ref push, then uploads the
allowlisted HF payload, then pushes the release branch and annotated tag
atomically. It creates the GitHub release only after those steps succeed.
An HF failure cannot publish the tag. Retries compare remote branch, annotated
tag object and peeled commit against the approved plan. Matching refs are reused;
conflicting refs refuse. The HF uploader records its parent and untouched-file
inventory before committing, then verifies immutable committed bytes. A lost
acknowledgement is recovered by observing exact bytes, without a second commit.

GitHub publication creates a draft, fills only missing approved assets, verifies
every asset hash and the approved notes, then publishes. It finds drafts through
the authenticated release listing. Only empty `starter` placeholders in the
matching draft may be deleted; uploaded mismatches and extra assets refuse.
A completed rerun leaves assets unchanged and skips an already-closed PR. All
SDK entry points set the Xet default before import. Existing approval and
payload/toolchain checks still apply.

This is a cumulative patch from the assembler's pre-round-1 files. If round 1
is already applied, use a fresh assembler baseline or apply only the reviewed
delta; do not force a failed patch. The assembler's historical frozen inputs
still describe v1.8.0. Do not relabel those measurements as v1.8.3 results.

Regenerate and revalidate staged packages after changing these tools: the
assembler binds its tool and publication-plan hashes. Do not rehash old
packages to bypass that refusal. Historical freezes and numbers stay unchanged.
The regression tests use local command stubs and perform no publication.
Range headlines bind the c4/c8 code/prose and prefill tokens to frozen data;
changed endpoints, missing ranges and missing two-sweep scope refuse.

Before publishing a new source release, review the actual staged source archive,
release body, landing README, HF card and any GitHub-hosted HF-card copy together.
They must name/link the release being published, while historical tables retain
their v1.8.0 identity. Link the current INSTALL, preserve model license metadata
and attribution, disclose coop off beside coop-on numbers, and use the pinned
weight-download revision. Updating these live pages is a publication step;
source edits alone do not change them. Do not re-run the historical frozen
assembler as though it were a new v1.8.3 package.
