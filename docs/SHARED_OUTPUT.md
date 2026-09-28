# Shared output policy

Shared receipts, gate JSON, logs and console output contain fixed labels,
validated identities, numbers, booleans and hashes. They never copy arbitrary
remote log lines, API text or exception messages. Compiler detection publishes
only the matched fixed marker. Gate decisions still inspect complete responses.
API token strings and request identifiers are hashed where equality matters;
metric names and token-usage fields come from fixed lists. This is projection,
not credential-pattern redaction.

`recipe/scripts/_diagnostics.py` owns the public/private boundary. CLI failure
handlers and the uncaught-exception hook share recognized traceback structure
and retain the complete exception privately. Child commands capture stdout and
stderr privately. A bounded tail is available separately for inspection. Private
files have mode 0600 and the suffix `.may-contain-secrets-do-not-share.log`;
standalone operations use a private temporary directory. Diagnostic storage
failure is reported without echoing the secondary exception. Full capture uses
space proportional to output, so operators must manage retention.

Qualification names its shared evidence explicitly. Admission refuses private
suffixes in evidence inventories. The private files are optional diagnostic
pointers, never files that admission needs to open. Share the qualification
receipt directory **excluding all private sidecars**. Build stage directories,
container `/evidence` and runtime module logs remain private; do not bundle them.
Experiment JSON and ordinary `.log` summaries can be shared, but compiled
artifacts, toolchain reports and raw module evidence cannot. Source releases
contain none of these generated files.

[The audit inventory](../manifests/shared-output-audit.json) enumerates every
Python, shell, native-source and patch file, its output/capture sites and its
disposition. It includes runtime channels and synthetic tests to make their
boundary explicit. Embedded worker programs use private RPC streams; controller
callers project their responses before writing shared output. Atomic serialization
helpers accept already-reviewed payloads; they do not sanitize arbitrary objects.
In-memory parsing helpers are not a public serialization API.

`validate_release.py` refuses any new executable file or changed executable
bytes until that inventory is reviewed and updated. Whole-file hashes cover
changes upstream of the writer, too. This is a review coverage guard, not a
static taint proof. Do not regenerate the inventory merely to make a check pass:
review each changed producer, its consumers, error handling and bundle inventory.
Add a regression containing opaque unexpected text, exercise the actual writer,
and prove both absence in shared output and presence in private diagnostics.

Discovery includes executable permission bits and shebangs as well as known
source suffixes, including extensionless and hidden shipped files. Git
administration and private task evidence are excluded from this executable
inventory; release validation separately refuses private evidence in exports.

## Component qualification projections

The component runner uses the same diagnostic boundary. Remove private sidecars
before sharing campaign evidence. Seals copy only manifest-listed bundle assets;
compiler logs, intermediates, unrelated files and private sidecars are excluded.
The historical and reproducible candidate toolchain reports are explicitly
approved by digest. A different report requires review even if the native output
is identical. Native artifacts still remain outside source releases.

| Producer / field | Shared representation | Consumer and preserved decision |
| --- | --- | --- |
| `build_native.py` physical inventory | Fixed architecture and GB10 identity kind; machine and UUID hashes | Two-build equality and distinct physical builder checks; no raw UUID or driver reply |
| `qualify_coop.py` plan and execution | Fixed matrix arguments, image hash, scope limits; host mount sources hashed | Gate validator checks exact matrix tail, GPU/network/image scope and plan/execution equality |
| `coop_environment.py` linkage | Fixed resolved-cudart fact, after checking the complete `ldd` output | Environment validator still requires shared cudart and no unresolved library |
| Environment tool versions / driver | Hashes of nonempty complete replies; exact GB10 name and capability | Presence and identity retained; versions had no version-order consumer; complete reports private |
| H1 command output | Fixed control schema, source/bundle hashes, comparison counts, finite sensitivity numbers | Payload self-hash, all 156 comparisons, expected perturbation rejection and threshold checks |
| Profile output | Closed rank/geometry/row/pattern labels; comparison outcomes and finite timing statistics | All 468 cases, nine completions, numerical tolerances, worst-case ratios and policy selection |
| Sanitizer output | Counted instrumented candidate launches and fixed clean summary, checked against complete raw output first | Missing/filtered launches, contradictory/error summaries and incomplete instrumentation still refuse; application completion, coverage and replay counts retained |
| Policy output | Fixed completion and numeric geometry / stock choices | Exact choices must match the policy selected from measured profiles |
| `coop_evidence.py` campaign index | Validated identities and named gate/profile evidence hashes | Seal validates projected artifacts; private files are never opened or required |
| Runtime qualification / admission | Component/native/policy/image hashes, fixed stock settings, hashed boot timestamp | Same component and boot throughout; admission checks the release seal; component exceptions retained privately |

Checkpoint, helper, sanitizer package and bundle identities retain their exact
authenticated contracts. The runner's code inventory also binds the projection
helper and diagnostic boundary. Raw compiled sources and the deterministic
native builder are unchanged. Container setup/import output stays in the private
capture; only the projected environment report is shared. Selection has no
free-text output consumer: its authenticated policy and bundle are the evidence.
Container setup assigns temporary diagnostic directories to the retained
`/work/private-protocol` bind mount, so removing a container preserves nested
child failures. That directory is private and is never copied into a seal.
Container-created diagnostics keep the container's ownership; inspect them with
the same local Docker access used for the campaign.

Regression coverage includes opaque text in real runner capture and admission
failure paths, full-output sanitizer negatives beyond the diagnostic tail,
profile selection and numerical failures, and a complete seal after deleting
private files. Source/receipt hash bindings intentionally require fresh receipts
after controller changes; this integration does not mint hardware evidence.

Native reproducibility refusals publish fixed artifact labels and validated
SHA-256 pairs. Failed native workspaces remain private (mode 0700) with both
builds and compiler intermediates intact; they are never successful build
receipts or shared evidence dependencies. Completed ELF bytes are not rewritten.
