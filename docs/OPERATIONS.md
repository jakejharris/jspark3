# Operations and admission

Keep admission closed until every rank passes the identity, safety, memory,
ablation-receipt and correctness gates. `recipe/scripts/verify_stock.py` and
the lifecycle verifier retain their refusal paths. Kernel-cache seeding and
post-warmup compile monitoring precede admission. A syntax check or a rendered
command is not a serving receipt.

The export includes native verification and standalone prefill, prefix-cache
and admission gates under `tools/v16/`. The private campaign driver stays
private. Operators run the following qualification protocol with their own
orchestration and retain evidence outside the source export:

1. Bind the candidate's source/runtime inventories, image, model, selected
   toggles and adaptive epoch. Start all three ranks with admission closed.
2. Seed the required kernel caches, warm required shapes, and verify cache
   identity plus zero post-readiness compilation on every rank.
3. Run native `verify_stock.py` and lifecycle verification. Under
   `production-stock`, require mode 0, no donor keys, and three rank-correct
   native receipts with exactly `schema_version=1`, `ablit=0`, `rank`,
   `state=DISABLED`, and `applied_layers=[]`. The aggregate verifier must
   report `production_stock.status=PASS`.
4. Check effective no-swap resource limits, memory admission and host reserves;
   apply post-load page-cache hygiene before the measured prefill gate.
5. Run correctness, prefix-cache and prefill gates, including required repeats
   after initial prefill dips and noise checks for greedy near-ties. Use the
   gate tools' explicit receipt inputs; an older tool's defaults cannot waive
   the candidate's qualification requirements.
6. Admit requests only after every required receipt passes for this same
   configuration. Production mode-0 admission requires fresh hardware qualification
   of the exact shipped configuration; historical QA receipts do not supply it.

Apply the selected fadvise/page-cache hygiene after loading, retain its
receipt, and measure the prefill gate after hygiene. Record the first prefill pass, apply hygiene and use the repeat to decide the
prefill gate. A post-hygiene failure remains a failure. Preserve both receipts;
this rule does not waive correctness. First-pass canaries are recorded only. Quick is diagnostic. Post-hygiene canaries and the >=1100 tok/s prefill floor block admission.
No post-readiness kernel compilation is allowed by the final qualification.

For hosts using snapd and runc 1.2.5, the operating procedure holds automatic
snap refreshes with `sudo snap refresh --hold` during the serving window.
Record that host change and resume the operator's update policy afterward.
A daemon reload can reset container swap limits. Verify the effective
`memory.swap.max` after service-manager changes; `MemorySwapMax=0` is the
required no-swap setting. Docker memory-plus-swap configuration must resolve
to that effective cgroup value on every rank.
The exported `recipe/scripts/fleetctl.py` wraps the inherited controller with
the sealed resource-scope helper. It adds the systemd zero-swap annotation,
retains native verification and checks the effective scope after start.

Selected startup flags and runtime epochs are recorded separately in [the final binding](../manifests/final-binding.json). A resident startup flag does not authorize an active runtime epoch. TRIAR stays inactive; thirds and active TRIAR are deferred. Adaptive-policy changes require drained requests, coordinated rank receipts and the full selected-configuration checks.

Choose stock or edited behavior before launch. Changing modes currently
requires a service restart and recomputes conversation prefixes. No runtime
swap qualification is claimed by this base export.

## Operator admission and measurement controls

Native verification proves runtime identity and gate conditions; it does not
provide a complete traffic-admission service. Keep ordinary traffic blocked
through operator-controlled routing or equivalent access controls until the
qualification protocol passes. A running container or successful status
command does not establish that requests may be admitted.

Run prefill checks through `tools/v16/prefill_gate.py`; its explicit default
floor is 1100 tok/s. Bind the floor, workload, source/runtime identity and cache
state to the qualification record. The gate rejects post-warmup compilation.
"Pi-shaped" means long shared prefixes, tool definitions and multi-turn coding
sessions. A greedy frozen Pi replay fixes the recorded requests and uses greedy
decoding; it is distinct from interactive use and from a quick triage run.

Page-cache hygiene uses `posix_fadvise(DONTNEED)` over the selected model and
deployment files as the operator, with read access and per-rank file/byte/error
receipts. Review errors against the operator's recorded baseline. No portable
hygiene command is included here. Record time since startup and distinguish a
first pass from the post-hygiene repeat; the cause of observed slow first
passes is not established by this export.

Check `memory.swap.max` on every rank after host or service-manager changes.
It must remain zero. A changed value invalidates the resource qualification;
keep admission closed until the required configuration and checks are restored.
The scope annotation is `org.systemd.property.MemorySwapMax=0`. Verify snap
holds with `snap refresh --time`, and retain the operator's resumption policy.

Perform supported adaptive-epoch changes with admission closed and requests
drained, coordinate all ranks, and record/read back the resulting identity.
Do not infer that every option has a live toggle. Source, image, native build,
weight mode and memory-layout changes require a fresh start and the full gate
sequence. Future levers are excluded until their sealed implementation and
qualification enter a later source checkpoint.
