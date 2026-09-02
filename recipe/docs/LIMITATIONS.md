# Limitations of the recipe

- Proven only on the pinned checkpoint, draft, image, and hardware. Any other
  revision or digest is refused by design, not adapted.
- Requires exactly three DGX Sparks with two RoCE-v2 legs each. No two-node
  or four-node variant is provided.
- The W8A16 overlay changes the runtime representation of the trunk weights
  (attention, shared and dense MLP, LM head) to INT8 Marlin at load. It
  excludes the KDA f/g modules and leaves routed experts in EXL3. Quality was
  checked by exact-output correctness gates in the measurement batteries, not
  by a public accuracy benchmark.
- The internal promotion gates that the measured build did not clear are
  disclosed in the repository's `docs/LIMITATIONS.md`. They concern pacing and
  a throughput floor, not correctness or stability.
- The API endpoint has no authentication. Front it with your own gateway.
- Containers run privileged (host network, host IPC, `/dev/infiniband`,
  `IPC_LOCK`) and pin 64 GiB of host memory per rank with swap disabled.
- The default DFlash2 draft path is non-commercial research and evaluation
  use; the target checkpoint is attribution-required. See the repository's
  `docs/LICENSING.md`.
