# Manifests

Every list here is `sha256sum` format (`<sha256>  <path>`), paths relative to the folder it describes, sorted
bytewise. `cd <folder> && sha256sum -c --strict <list>` checks a folder without the recipe's scripts.

| List | Describes | Checked by |
|---|---|---|
| `inputs/base-weights.sha256` | `$DATA/base/weights`: every file of the base weights at `BASE_REV` | `scripts/fetch-weights.sh` |
| `inputs/drafter.sha256` | `$DATA/drafter`: every file of the DFlash2 draft model at `DRAFTER_REV` | `scripts/fetch-weights.sh` |
| `inputs/ablit-source.sha256` | `$DATA/ablit/source`: the refusal-removed source at `ABLIT_SOURCE_REV` | `scripts/fetch-weights.sh --weights ablit` |
| `inputs/ablit-weights.sha256` | `$DATA/ablit/weights`: the converted ablit weights | `scripts/convert-ablit.sh` |
| `base/rank0.sha256`, `rank1`, `rank2` | `$DATA/base/rank<R>`: one box's third of the base weights, as the measured boxes served it | `scripts/split.sh` |
| `ablit/rank0.sha256`, `rank1`, `rank2` | `$DATA/ablit/rank<R>`: the same for the ablit weights | `scripts/convert-ablit.sh`, `scripts/split.sh --weights ablit` |

`ablit/rank<R>.sha256` describe the ablit thirds as `convert-ablit.sh` writes them. They match the thirds the published ablit numbers were measured on in every weight file and header; only `.complete` and `MANIFEST.sha256` differ, because they name the conversion. `scripts/ablit/vontra.sha256`, the conversion's own list of its base input, is identical to `inputs/base-weights.sha256`.

A per-rank list includes `.complete`, the marker `split.sh` writes when the split finishes, and the chat template the
split copies from the weights; at serve time `template/chat-template.jinja` is mounted over that copy. A base third's
marker is empty. An ablit third's marker is `<ABLIT_CONVERSION_ID> rank=<R> world=3`, and the third also holds
`MANIFEST.sha256`, the sorted list of every other file in it, as on the thirds the ablit numbers were measured on.
`convert-ablit.sh` writes all three thirds this way, and `split.sh` does the same when it splits the converted
weights again (`scripts/verify-split.py seal`). A split
that differs from its list in any file is not marked verified, and `serve.sh` refuses it.
`python3 scripts/verify-split.py check <folder> <list>` names every missing, extra or different file.
