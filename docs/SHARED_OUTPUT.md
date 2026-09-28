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

Native reproducibility refusals publish fixed artifact labels and validated
SHA-256 pairs. Failed native workspaces remain private (mode 0700) with both
builds and compiler intermediates intact; they are never successful build
receipts or shared evidence dependencies. Completed ELF bytes are not rewritten.
