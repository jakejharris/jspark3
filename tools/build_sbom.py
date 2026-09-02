#!/usr/bin/env python3
"""Emit a CycloneDX 1.5 SBOM for JSpark3 v1 from manifests/dependencies.json.

The SBOM lists the recipe itself and every pinned external input (container
image, checkpoints, upstream source revisions). It is deterministic: the serial
number is derived from the content so rebuilding an unchanged tree yields the
same document. Nothing is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import uuid


def component_image(image: dict) -> dict:
    name, _, digest = image["reference"].partition("@")
    return {
        "type": "container",
        "bom-ref": "pkg:oci/glm-5.3-flash-2x-dgx-sparks@" + digest.replace(":", "%3A"),
        "name": name.rsplit("/", 1)[-1],
        "group": name.rsplit("/", 1)[0],
        "version": digest,
        "purl": f"pkg:oci/glm-5.3-flash-2x-dgx-sparks@{digest.replace(':', '%3A')}?repository_url={name}&arch={image['architecture']}",
        "hashes": [{"alg": "SHA-256", "content": digest.split(":", 1)[1]}],
        "supplier": {"name": image["publisher"]},
        "description": "Pinned upstream serving image; observed server fingerprint " + image["observed_server_fingerprint"],
        "properties": [
            {"name": "jspark3:config_digest", "value": image["config_digest"]},
            {"name": "jspark3:redistributed_here", "value": "false"},
        ],
    }


def component_model(key: str, model: dict, role: str) -> dict:
    hashes = []
    for field in ("native_config_sha256", "runtime_config_sha256", "model_index_sha256", "model_sha256", "tokenizer_sha256", "tokenizer_config_sha256"):
        if field in model:
            hashes.append({"name": "jspark3:" + field, "value": model[field]})
    return {
        "type": "machine-learning-model",
        "bom-ref": f"pkg:huggingface/{model['repository']}@{model['revision']}",
        "name": model["repository"].split("/", 1)[1],
        "group": model["repository"].split("/", 1)[0],
        "version": model["revision"],
        "purl": f"pkg:huggingface/{model['repository']}@{model['revision']}",
        "description": role,
        "licenses": [{"license": {"name": model["license"]}}],
        "properties": hashes + mirror_properties(model) + [
            {"name": "jspark3:redistributed_here",
             "value": model.get("mirror", {}).get("status", "false")},
        ],
    }


def mirror_properties(model: dict) -> list[dict]:
    """Record the mirror state on the component it mirrors.

    The mirror has no digest of its own: it is the same repository at the same
    revision. Emitting a second component
    would mean either duplicating the upstream purl or inventing an identifier, so
    the mirror is recorded as properties of the component it copies.
    """
    mirror = model.get("mirror")
    if not mirror:
        return []
    properties = [
        {"name": "jspark3:mirror_status", "value": mirror["status"]},
        {"name": "jspark3:mirror_repository", "value": mirror["repository"]},
        {"name": "jspark3:mirror_revision", "value": mirror["revision"]},
        {"name": "jspark3:mirror_intended_destination", "value": mirror["intended_mirror"]},
        {"name": "jspark3:mirror_files", "value": str(mirror["files"])},
        {"name": "jspark3:mirror_bytes", "value": str(mirror["bytes"])},
        {"name": "jspark3:mirror_upload_files", "value": str(mirror["upload_files"])},
        {"name": "jspark3:mirror_upload_bytes", "value": str(mirror["upload_bytes"])},
        {"name": "jspark3:mirror_destination_policy", "value": mirror["destination_policy"]},
        {"name": "jspark3:mirror_transfer_client", "value": mirror["transfer_client"]},
        {"name": "jspark3:mirror_resume_cache", "value": mirror["resume_cache"]},
        {"name": "jspark3:mirror_completion_receipt", "value": mirror["completion_receipt"]},
        {"name": "jspark3:mirror_merge_policy", "value": mirror["merge_policy"]},
        {"name": "jspark3:mirror_manifest", "value": mirror["manifest"]},
    ]
    if mirror.get("gate"):
        properties.append({"name": "jspark3:mirror_gate", "value": mirror["gate"]})
    return properties


def component_source(key: str, source: dict) -> dict | None:
    if "commit" not in source:
        return None
    repo = source["repository"].removeprefix("https://github.com/")
    return {
        "type": "library",
        "bom-ref": f"pkg:github/{repo}@{source['commit']}",
        "name": repo.split("/", 1)[1],
        "group": repo.split("/", 1)[0],
        "version": source["commit"],
        "purl": f"pkg:github/{repo}@{source['commit']}",
        "description": source["role"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependencies", type=Path, default=Path("manifests/dependencies.json"))
    parser.add_argument("--release", type=Path, default=Path("manifests/release.json"))
    parser.add_argument("--output", type=Path, default=Path("manifests/sbom.cdx.json"))
    args = parser.parse_args()
    deps = json.loads(args.dependencies.read_text(encoding="utf-8"))
    release = json.loads(args.release.read_text(encoding="utf-8"))
    version = release["tag"].lstrip("v")
    root = {
        "type": "application",
        "bom-ref": f"pkg:github/jakejharris/jspark3@v{version}",
        "name": release["slug"],
        "version": version,
        "description": release["product_type"],
        "licenses": [{"license": {"id": "Apache-2.0"}}],
        "purl": f"pkg:github/jakejharris/jspark3@v{version}",
    }
    components = [
        component_image(deps["container_image"]),
        component_model("target", deps["target_checkpoint"], "target checkpoint served by the recipe (fetched by the operator, never bundled)"),
        component_model("draft", deps["draft_checkpoint"], "DFlash2 speculative draft checkpoint (fetched by the operator, never bundled)"),
    ]
    for key, source in deps["source_revisions"].items():
        component = component_source(key, source)
        if component:
            components.append(component)
    components.append({
        "type": "library",
        "bom-ref": "vllm@487ecf187",
        "name": "vllm",
        "version": "487ecf187",
        "description": deps["source_revisions"]["vllm"]["pinned_build"] + "; " + deps["source_revisions"]["vllm"]["role"],
    })
    body = {
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": root,
            "tools": [{"name": "jspark3-build-sbom", "version": version}],
            "properties": [
                {"name": "jspark3:policy", "value": deps["policy"]},
                {"name": "jspark3:release_status", "value": release["status"]},
                {"name": "jspark3:release_date", "value": release["date_released"]},
                {"name": "jspark3:release_url", "value": release["live_links"]["release_page"]},
                {"name": "jspark3:hf_metadata_revision", "value": release["weights_mirror"]["hf_revision"]},
                {"name": "jspark3:hf_weight_mirror_status", "value": release["weights_mirror"]["status"]},
                {"name": "jspark3:owned_image_status", "value": deps["owned_runtime_image"]["status"]},
                {"name": "jspark3:owned_image_publication_policy", "value": deps["owned_runtime_image"]["publication_policy"]},
                {"name": "jspark3:runtime_image_reference", "value": deps["owned_runtime_image"]["runtime_reference"]},
                {"name": "jspark3:owned_image_redistributed", "value": str(deps["owned_runtime_image"]["redistributed_here"]).lower()},
            ],
        },
        "components": components,
        "dependencies": [{"ref": root["bom-ref"], "dependsOn": [c["bom-ref"] for c in components]}],
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    body["serialNumber"] = "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, "jspark3-sbom-" + digest))
    args.output.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(components)} components)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
