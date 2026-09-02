# Security policy

JSpark3 v1 runs privileged containers (host networking, host IPC,
`/dev/infiniband`, `IPC_LOCK`) on hardware you own. Read `docs/OPERATIONS.md`
before exposing the endpoint.

## Reporting

Report vulnerabilities in the recipe, its tooling, or its documentation through
the repository's private vulnerability reporting once it is live, or to the
maintainer contact listed in `manifests/release.json` once it is filled in.
Do not open a public issue for an exploitable problem.

## What is in scope

- The lifecycle controller, preflight, entrypoint, transforms, and overlay
  under `recipe/`.
- The validators and builders under `tools/`.
- Documentation that could lead an operator to an unsafe configuration.

## What is out of scope

- The upstream model checkpoints, the pinned container image, vLLM,
  ExLlamaV3, and upstream source revisions. Report those upstream.

## Design notes

- The recipe never fetches floating `latest` content. Every input is pinned by
  revision or digest and verified before launch.
- The API endpoint has no authentication of its own. Put it behind your own
  gateway; the smoke tools read `OPENAI_API_KEY` only to pass it through.
- Receipts written by the controller contain container identities and the
  paths you configured. Treat them as private operational records.
