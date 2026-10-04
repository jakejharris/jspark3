# v2.0.1 install-documentation sweep (2026-10-04)

Scope: documentation pass 3 on top of `599c77e` (installation corrections) and
`619b2f8` (known issues and reversible hotfix instructions). **Prepared locally;
not published.** Forge owns the subsequent release-body and HF-card updates.

## Review dispositions

- **P1 accepted, fixed in source.** The HF card selected tagged INSTALL, while
  tagged README and the release Quick start selected the uncorrected guides.
  The card now selects maintained INSTALL, UPGRADING and startup known issues.
  [Release errata](release-errata-v2.0.1.md) is the exact block for Forge to prepend
  to the v2.0.1 release body, above Quick start. Publication remains pending.
- **P2 accepted, fixed.** Tagged OPERATIONS says warm starts are quick because
  kernels are cached. The tagged launch command installs the wheel in every fresh
  container ([serve.sh](https://github.com/jakejharris/jspark3/blob/v2.0.1/scripts/serve.sh#L125-L126));
  the maintained [rebuild entry](TROUBLESHOOTING.md#kernels-rebuild-on-every-start)
  explains the timestamp invalidation and pending hotfix. Explicit errata now sit
  beside the reboot link in TROUBLESHOOTING and the operations link in INSTALL.

## Current guidance and tagged references

Each disposition describes this branch, not the live platform state.

| Surface | Fixed or left | Reason / resulting route |
|---|---|---|
| HF card: opening install notice | **Fixed** | Added direct maintained INSTALL and startup-known-issues links beside the release link, so readers need not reach the lower Install section to find the errata. |
| [HF card: Install](../huggingface/README.md#install) | **Fixed** | Tagged INSTALL replaced with maintained `main/INSTALL.md`; added `main/UPGRADING.md` and the maintained known-issues section. Explicitly keeps the runnable recipe at `v2.0.1`. |
| [HF card: Draft model and commercial use](../huggingface/README.md#draft-model-and-commercial-use) | **Fixed** | The launch-only summary now links to the complete no-drafter install/preflight procedure. |
| [HF card: session-tier pointer](../huggingface/README.md#install) | **Fixed** | Bare OPERATIONS reference replaced with maintained INSTALL's session-tier section. |
| [HF card: Known issues](../huggingface/README.md#known-issues) | **Fixed** | Added a maintained startup/hotfix pointer; preserved all 17 API issue numbers and text. Hotfix was later validated on hardware (see the hotfix document for scope). |
| [HF card: rollback](../huggingface/README.md#upgrade-and-rollback) | **Fixed** | Added a maintained UPGRADING pointer after the existing pinned mirror correction. The EXL3 footer now also selects maintained rollback instructions. Kept the v1.8.4 tag and mirror revision. |
| HF card: release badge, RigMark and measurement links | **Left** | Identify the original release and its measured results, not installation instructions. The opening notice and Install section now supply direct maintained-guide links. |
| [README: entry points](../README.md) | **Left** | Passes 1–2 already point to maintained INSTALL, UPGRADING and startup known issues. Release/results links still identify the original release and measurements. |
| [README: draft-model summary](../README.md#draft-model-license) | **Fixed** | Added the maintained no-drafter install/preflight link beside the launch-only summary. |
| [README: Releases](../README.md#releases) | **Fixed** | Added the required download-source correction beside the v1.8.4 install link. Clarified that the frozen v1.1.0 files are the runnable recipe, while v2.0.1 guides are maintained here. |
| [INSTALL: operations link](../INSTALL.md#stopping-restarting-upgrading) | **Fixed** | Retained the tagged operations reference with an explicit rebuild erratum and maintained startup/recovery link. |
| [TROUBLESHOOTING: reboot recovery](TROUBLESHOOTING.md#a-rank-never-becomes-ready) | **Fixed** | Explicitly rejects the tagged warm-start claim beside the link; directs stalled compiles to the guarded lock-recovery entry. |
| [TROUBLESHOOTING: unavailable sources](TROUBLESHOOTING.md#dead-or-unavailable-upstream-weight-source) | **Left** | The tagged UPGRADING lines are evidence for a separate v2 installation, not rollback instructions. The adjacent maintained UPGRADING link supplies the corrected mirror procedure. |
| TROUBLESHOOTING: other tagged OPERATIONS links | **Left** | Links target settings profiles, weight terms/conversion cost, disk-full behavior, other GPU work, capacity, images, API access and log redaction. Those sections do not make the corrected warm-start claim; reboot guidance now has its explicit erratum. |
| [UPGRADING: v1.8.4 install link](../UPGRADING.md#going-back-to-v184) | **Left** | It intentionally identifies the historical guide whose download command is replaced immediately below by the pinned mirror instructions. |
| [Kernel rebuild hotfix](hotfixes/v2.0.1-kernel-rebuild.md) | **Left** | Links already select maintained install/lock instructions and immutable source evidence. This pass did not validate the hotfix; it was validated separately on hardware (see the hotfix document for scope). |
| Script-message quotations in TROUBLESHOOTING | **Left** | Literal `INSTALL.md` mentions reproduce v2.0.1 diagnostics; rewriting them would stop matching what users see. Maintained guidance surrounds the quotations. |
| v2.0.1 tag: README, INSTALL, UPGRADING, TROUBLESHOOTING and OPERATIONS | **Left** | Immutable release history. The release errata block routes readers to maintained corrections without moving the tag or replacing assets. |
| v2.0.1 release body / Quick start | **Fixed in draft; live left** | Exact prepend block is in [release-errata-v2.0.1.md](release-errata-v2.0.1.md). Forge applies it after the maintained guides reach main; no release API was called. |

## Historical v1 surfaces

These files belong to the frozen v1.1.0 export, not the v2.0.1 install path.
Their relative `INSTALL.md` links or `main/docs/INSTALL.md` links still resolve to
that historical guide. They are retained individually for the reasons below;
the maintained v2 guide is root `INSTALL.md`, not `docs/INSTALL.md`.

| Surface | Fixed or left | Reason |
|---|---|---|
| [docs/INSTALL.md](INSTALL.md) | **Left** | Explicitly installs v1.1.0. Its old source alternatives are historical; redirecting its commands to v2 would mix incompatible recipes. |
| [docs/OPERATIONS.md](OPERATIONS.md) | **Left** | Local v1 operations and its three v1 install anchors remain paired with the v1 recipe; this is not the tagged v2 operations file reviewed above. |
| [docs/ARCHITECTURE.md](ARCHITECTURE.md) | **Left** | Dated download clarification points to the historical v1 setup it describes. |
| [docs/TECHNICAL-REPORT.md](TECHNICAL-REPORT.md) | **Left** | v1 technical report's install and source references describe that measured recipe. |
| [docs/REPRODUCIBILITY.md](REPRODUCIBILITY.md) | **Left** | Reproduces the v1 recipe and its inputs; retains the matching install guide. |
| [RELEASE-GATE.md](../RELEASE-GATE.md) | **Left** | Historical v1 release state and dated download clarification. |
| [FINAL-RELEASE-INDEX.md](../FINAL-RELEASE-INDEX.md) | **Left** | Historical v1 release inventory and dated download clarification. |
| [docker/README.md](../docker/README.md) | **Left** | v1 Docker payload documentation; setup link selects its own recipe generation. |
| [recipe/README.md](../recipe/README.md) | **Left** | Frozen v1 recipe's prose reference to its install guide. |
| [SECURITY.md](../SECURITY.md) | **Left** | Refers to local v1 operations for the frozen recipe's privileges. |
| [CHANGELOG.md](../CHANGELOG.md) | **Left** | Records an actual historical correction to v1 INSTALL; not a current installation entry point. |
| [release/RELEASE-NOTES.md](../release/RELEASE-NOTES.md) | **Left** | v1 release notes, not the mutable v2.0.1 GitHub release body; keep its historical setup link. |
| [release/ANNOUNCEMENT-BLOG.md](../release/ANNOUNCEMENT-BLOG.md) | **Left** | Historical v1 announcement copy and setup references. |
| [release/ANNOUNCEMENT-SOCIAL.md](../release/ANNOUNCEMENT-SOCIAL.md) | **Left** | Historical v1 announcement copy and setup link. |
| [huggingface/jspark3/UPLOAD.md](../huggingface/jspark3/UPLOAD.md) | **Left** | Historical v1 upload instructions and dated download clarification; not the current card source. |
| [huggingface/jspark3/PROVENANCE.md](../huggingface/jspark3/PROVENANCE.md) | **Left** | Historical v1 mirror provenance and matching setup link. |
| [huggingface/jspark3/V1.1.0-RELEASE.md](../huggingface/jspark3/V1.1.0-RELEASE.md) | **Left** | Explicit v1.1.0 release record, including its tagged installation link. |

## Discovery and handoff

- Searched repository text (including hidden tracked configuration) for
  `INSTALL.md`, `UPGRADING.md`, `TROUBLESHOOTING.md`, `OPERATIONS.md`, tagged
  v2.0.1 guide URLs, `main/docs/INSTALL`, warm-start/cache claims, no-drafter
  summaries and Mia-AiLab source references. Inspected the v2.0.1 tag's README,
  operations reboot section and launch command directly through local Git.
- No card generator or template is tracked in this repository. The maintained
  card source is `huggingface/README.md`; `tools/build_release_assets.sh` packages
  files rather than rendering that card. Preserve these corrections if external
  release tooling regenerates it. `tools/validate_release.py` references the
  historical install/operations files for validation, not user navigation, and
  remains unchanged. Root `SHA256SUMS` is refreshed for this documentation pass;
  historical manifests and receipt inventories remain unchanged.
- The frozen `.github/workflows/release.yml` still uses `release/RELEASE-NOTES.md`
  as its body source. **Left:** it is the v1 export workflow, not the current v2.0.1
  body; the exact v2 errata is handed to Forge separately. No workflow or generator
  code changed.
- Merge the maintained guides to main before publishing the HF card or pasting
  the errata: the new external links deliberately target main, not this branch.
  Forge should prepend the exact errata file above the release's existing Quick
  start, preserving its other text and existing known-issue numbering. The HF
  publication source is this branch's card, not the entire `huggingface/` tree.
- Live HF/release pages, websites, blogs and external generators were not changed
  or verified in this repository-only pass. No push, PR, upload or release edit
  was performed. Tag `v2.0.1` remains object
  `8bf422fd58c72a40ea9c1da2db41c7cc4678ce16`, pointing to commit
  `0be336670bb1ce8827ff7dbf1f33d1d61ea33cd7`.
