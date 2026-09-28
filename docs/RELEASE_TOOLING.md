# Release packaging maintenance

The release assembler is maintained separately from this source export.
[`release_packaging.patch`](../tools/release_packaging.patch) records the changes
to its `tools/render.py`, `tools/package.py`, `tools/comparison_variants.py`,
`tools/hf_upload.py` and
`9AM-COMMANDS.sh`. Apply it to the assembler checkout, not to this source tree:

```sh
git -C "$PACKAGING_DIR" apply --check "$SOURCE_DIR/tools/release_packaging.patch"
git -C "$PACKAGING_DIR" apply "$SOURCE_DIR/tools/release_packaging.patch"
python3 -B "$SOURCE_DIR/tools/test_release_tooling.py" --packaging-root "$PACKAGING_DIR"
```

The renderer and validator require the model card's ShapleyMcg v1.0 metadata
(`license: other`, `license_name: shapleymcg-license-1.0`, `license_link: LICENSE`).
The recipe's component licenses remain separate. The uploader defaults to
`HF_HUB_DISABLE_XET=1` before importing the HF SDK; the publication shell sets
the same default.

The publisher first validates and dry-runs the ref push, then uploads the
allowlisted HF payload, then pushes the release branch and annotated tag
atomically. It creates the GitHub release only after those steps succeed.
An HF failure cannot publish the tag. This is not an atomic transaction across
hosts: a failure after the HF commit still requires reconciliation before a
retry. Existing approval, compare-and-swap and payload checks remain in place.

Regenerate and revalidate staged packages after changing these tools: the
assembler binds its tool and publication-plan hashes. Do not rehash old
packages to bypass that refusal. Historical freezes and numbers stay unchanged.
The regression tests use local command stubs and perform no publication.
