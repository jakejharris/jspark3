#!/usr/bin/env python3
"""The current landing page's credit appendix must not rewrite frozen v1 notices."""
import contextlib
import io
from pathlib import Path
import shutil
import tempfile

import validate_release as release

ROOT = Path(__file__).resolve().parents[1]


def check(root: Path, landing: bool = True) -> bool:
    report = release.Report()
    with contextlib.redirect_stdout(io.StringIO()):
        release.check_copies(root, report, landing=landing)
    return report.failed == 0


def main() -> None:
    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        for name in (
            'LICENSE', 'THIRD_PARTY_NOTICES.md', 'REQUIRED_ATTRIBUTION.md',
            'README.md', 'huggingface/README.md', 'docs/LICENSING.md',
            'recipe/LICENSE', 'recipe/THIRD_PARTY_NOTICES.md',
            'recipe/REQUIRED_ATTRIBUTION.md', 'huggingface/jspark3/RECIPE-LICENSE',
            'huggingface/jspark3/THIRD_PARTY_NOTICES.md',
            'huggingface/jspark3/REQUIRED_ATTRIBUTION.md',
            'huggingface/RESULTS.json', 'results/results.json',
        ):
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / name, target)
        notices = root / 'THIRD_PARTY_NOTICES.md'
        corrected = notices.read_bytes()
        historical = (root / 'recipe/THIRD_PARTY_NOTICES.md').read_bytes()
        assert corrected.startswith(historical)
        assert check(root), 'the landing-page correction must preserve frozen copies'
        assert not check(root, landing=False), 'release mode must still require identical notices'
        notices.write_bytes(historical)
        assert check(root, landing=False), 'the historical export must still pass'
        notices.write_bytes(corrected.replace(b'FlyCockpit', b'WrongAuthor', 1))
        assert not check(root), 'a change to historical attribution must fail'
        notices.write_bytes(corrected)
        mirror = root / 'huggingface/jspark3/THIRD_PARTY_NOTICES.md'
        mirror.write_bytes(corrected)
        assert not check(root), 'a changed frozen mirror must fail'
        mirror.write_bytes(historical)
        license_file = root / 'LICENSE'
        license_file.write_bytes(license_file.read_bytes() + b'changed\n')
        assert not check(root), 'the landing exception must not relax license checks'
    print('PASS landing notices: dated credit appendix accepted; historical, mirror and license changes rejected')


if __name__ == '__main__':
    main()
