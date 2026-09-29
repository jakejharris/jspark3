# JSpark3 v1.8.4 technical report

This source release prepares cooperative MoE for GLM-5.3 Flash on three DGX
Sparks. The [qualification handoff](COOP_QUALIFICATION_HANDOFF.md) and
[release checklist](COOP_RELEASE_CHECKLIST.md) describe the evidence required
for the compiled component and the operator's serving boot. Component
qualification alone does not establish a serving performance result.

The [v1.8.4 measurement record](../release/MEASUREMENTS-v1.8.4.md) carries this
release's measurement status and scope. Historical results remain in their
[frozen record](../release/results.json). The earlier
[v1/v1.1 technical report](https://github.com/jakejharris/jspark3/blob/646b89930e001dddcbe734745f8c3245d391d3e9/docs/TECHNICAL-REPORT.md)
remains a separate historical publication; its figures do not describe v1.8.4.

The target model uses [ShapleyMcg](https://github.com/brandonmmusic-max/shapleymcg/tree/4dc85c999983bf46ebdae3821839a5079cc5de17), created by Brandon M. Music.

This work includes or was produced using ShapleyMcg, created by Brandon M. Music (https://github.com/brandonmmusic-max/shapleymcg). ShapleyMcg is licensed under the ShapleyMcg License v1.0, an attribution-required license that grants no rights to the person known as "0xSero." Use of ShapleyMcg without this attribution is unlicensed.

## ShapleyMcg citation

```bibtex
@misc{music2026shapleymcg,
  author = {Music, Brandon M.},
  title  = {ShapleyMCG: An Auditable Calibration-to-Encoding Pipeline for
            Low-Bit Mixture-of-Experts Models},
  year   = {2026},
  url    = {https://github.com/brandonmmusic-max/shapleymcg},
  note   = {Licensed under the ShapleyMcg License v1.0}
}
```

The notice and citation follow Schedule B of the [pinned ShapleyMcg License
v1.0](https://github.com/brandonmmusic-max/shapleymcg/blob/4dc85c999983bf46ebdae3821839a5079cc5de17/LICENSE).
The source license hash remains in [REQUIRED_ATTRIBUTION.md](../REQUIRED_ATTRIBUTION.md).
