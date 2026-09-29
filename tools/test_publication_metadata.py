#!/usr/bin/env python3
"""Offline checks for authorized release metadata, without remote publication."""
import contextlib
import copy
import io
import json
from pathlib import Path
import shutil
import tempfile

import validate_release as release

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path) -> bool:
    report = release.Report()
    with contextlib.redirect_stdout(io.StringIO()):
        release.check_release_manifest(root, report, landing=True)
    return report.failed == 0


def main() -> None:
    assert check(ROOT), 'authorized release content must pass'
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw) / 'source'
        shutil.copytree(ROOT, root, ignore=shutil.ignore_patterns('.git', '__pycache__', 'dist'))
        path = root / 'manifests/release.json'
        original = json.loads(path.read_text())
        changes = [
            (('publication_authorized',), False),
            (('tag',), 'v1.0.0'),
            (('status',), 'v1.1.0-staged-not-released'),
            (('date_released',), None),
            (('date_released',), '2026-09-06'),
            (('live_links', 'release_page'), None),
            (('publication_record', 'observed_publication'), {'status': 'PUBLISHED'}),
            (('publication_record', 'kind'), 'remote-receipt'),
            (('historical', 'v1.0.0', 'date_released'), '2026-09-07'),
        ]
        for keys, value in changes:
            changed = copy.deepcopy(original)
            parent = changed
            for key in keys[:-1]:
                parent = parent[key]
            parent[keys[-1]] = value
            path.write_text(json.dumps(changed))
            assert not check(root), f'accepted inconsistent publication metadata: {keys}'
        path.write_text(json.dumps(original))
        citation = root / 'CITATION.cff'
        citation.write_text(citation.read_text().replace('date-released: 2026-09-07\n', ''))
        assert not check(root), 'accepted undated citation for dated release'
    print(f'PASS publication metadata: authorized content accepted; {len(changes)} '
          'metadata mutations and missing citation date rejected; no remote writes')


if __name__ == '__main__':
    main()
