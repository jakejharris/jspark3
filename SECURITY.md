# Security policy

## Current release (v2.0.1)

The server listens on loopback (127.0.0.1) only, with no authentication and no CORS. Reach it through an SSH tunnel or a reverse proxy that adds authentication; don't expose the port.

Conversation state, including the prompt's token ids, is cached on each host's own disk (up to 64 GiB per host) so returning to a long conversation is fast. It never leaves your machines. OPERATIONS explains where it lives, how to clear it and how to turn it off (SESSION_TIER=off).

The installer pins every download by revision or digest and verifies it by
SHA-256 before use. The `ablit` variant downloads a gated source with your own
`HF_TOKEN`; keep that token out of shared shells, logs and screenshots.

## Reporting

Report vulnerabilities in the recipe, its tooling, the vendored engine
(`engine/` on the release branch) or the documentation privately to the
maintainer, https://github.com/jakejharris. Do not open a public
issue for an exploitable problem.

## What is out of scope

Upstream projects: the model weights, the draft model, the NVIDIA container
image, the upstream engine project and other upstream sources. Report those
upstream. This engine is a fork of TensorFold 0.3.6.2 (MIT).

## v1.x releases

JSpark3 v1 runs privileged containers (host networking, host IPC,
`/dev/infiniband`, `IPC_LOCK`) on hardware you own. Read `docs/OPERATIONS.md`
before exposing the endpoint.

In scope for v1.x: the lifecycle controller, preflight, entrypoint, transforms
and overlay under `recipe/`; the validators and build scripts under `tools/`; and
documentation that could lead to an unsafe configuration. Out of scope: the
upstream model checkpoints, the pinned container image, vLLM, ExLlamaV3 and
upstream source revisions.

- The recipe never fetches floating `latest` content. Every input is pinned by
  revision or digest and verified before launch.
- The API endpoint has no authentication of its own. Put it behind your own
  gateway; the smoke tools read `OPENAI_API_KEY` only to pass it through.
- Receipts written by the controller contain container identities and the
  paths you configured. Treat them as private operational records.
