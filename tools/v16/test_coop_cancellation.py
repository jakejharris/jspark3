#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Opt-in local Docker cancellation tests; only harmless sleep commands, no GPU.

python3 -B tools/v16/test_coop_cancellation.py --docker-image LOCAL_ARM64_IMAGE
Use --evidence-dir NEW_DIRECTORY to retain each test's bind mounts and logs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import uuid

sys.dont_write_bytecode = True
IMAGE = None
EVIDENCE = None
RUNNER = Path(__file__).resolve().parent

# Exercise main's real create/start/cleanup path, substituting only the expensive
# source/image checks and compiler workload. Docker and OS signals remain real.
CHILD = '''import json, sys, time
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0, sys.argv[1])
import experiment_coop_build as e
work, image, phase = Path(sys.argv[2]), sys.argv[3], sys.argv[4]
e.COMMAND = 'printf retained-evidence > /w/evidence.txt; echo ready; exec sleep 300'
if phase in ('cleanup', 'failure'):
    e.COMMAND = 'printf retained-evidence > /w/evidence.txt; echo ready; exit 7'
receipt = work / 'image.json'
receipt.write_text('{}')
record = {'config_digest': image,
          'source_recipe_sha256': e.native.sha(e.ROOT / 'recipe/SHA256SUMS')}
original = e.subprocess.run
def run(command, **kwargs):
    if command[:2] == ['docker', 'rm'] and phase in ('cleanup', 'repeat'):
        (work / 'cleanup-ready').touch()
        while not (work / 'release').exists(): time.sleep(0.01)
    result = original(command, **kwargs)
    if command[:2] == ['docker', 'create'] and phase == 'create':
        (work / 'create-ready').touch()
        while not (work / 'release').exists(): time.sleep(0.01)
    return result
sys.argv = ['experiment', '--image-receipt', str(receipt), '--output', str(work / 'builds')]
with patch.object(e.native, 'verify_local_image'), \\
     patch.object(e.native, 'read_operator_record', return_value=record), \\
     patch('validate_release.verify', return_value={'failed': 0}), \\
     patch.object(e.subprocess, 'run', side_effect=run):
    raise SystemExit(e.main())
'''


def docker(*args, check=True):
    return subprocess.run(['docker', *args], check=check, capture_output=True, text=True)


class CancellationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if IMAGE is None:
            raise unittest.SkipTest('requires --docker-image with a local ARM64 image')
        cls.image = docker('image', 'inspect', '--format', '{{.Id}}', IMAGE).stdout.strip()
        # Same name prefix as the experiment: cleanup must never use a prefix scan.
        cls.peer = docker('run', '-d', '--rm', '--name', 'jspark3-coop-' + uuid.uuid4().hex,
                          '--platform', 'linux/arm64', '--network', 'none', '--cpus', '0.25',
                          '--memory', '256m', '--memory-swap', '256m',
                          '-e', 'NVIDIA_VISIBLE_DEVICES=void', '--entrypoint', '/bin/sleep',
                          cls.image, '300').stdout.strip()
        cls.addClassCleanup(lambda: docker('rm', '--force', cls.peer, check=False))

    def wait_for(self, condition, child):
        deadline = time.monotonic() + 30
        while not condition():
            self.assertIsNone(child.poll(), 'runner exited before the test barrier')
            self.assertLess(time.monotonic(), deadline, 'runner did not reach the test barrier')
            time.sleep(0.02)

    def exercise(self, phase, signum=None, group=False):
        if EVIDENCE:
            work = Path(tempfile.mkdtemp(prefix=phase + '-', dir=EVIDENCE))
        else:
            directory = tempfile.TemporaryDirectory()
            self.addCleanup(directory.cleanup)
            work = Path(directory.name)
        child_script = work / 'child.py'
        child_script.write_text(CHILD)
        with (work / 'runner.log').open('w') as log:
            child = subprocess.Popen([sys.executable, '-B', str(child_script), str(RUNNER),
                                      str(work), self.image, phase], stdout=log, stderr=log,
                                     start_new_session=True)
            try:
                stage = work / 'builds/candidate-1'
                marker = (work / (phase + '-ready') if phase in ('create', 'cleanup')
                          else stage / 'evidence.txt')
                if phase != 'failure':
                    self.wait_for(marker.exists, child)
                running = None
                if phase not in ('cleanup', 'failure'):
                    cid = (stage / 'container.cid').read_text().strip()
                    before = json.loads(docker('inspect', cid).stdout)[0]
                    running = before['State']['Running']
                    self.assertEqual(running, phase != 'create')
                if phase == 'create':
                    self.assertFalse((stage / 'evidence.txt').exists())
                if signum:
                    if group:
                        os.killpg(child.pid, signum)
                    else:
                        child.send_signal(signum)
                if phase == 'repeat':
                    self.wait_for((work / 'cleanup-ready').exists, child)
                    child.send_signal(signal.SIGTERM)
                if phase in ('create', 'cleanup', 'repeat'):
                    (work / 'release').touch()
                code = child.wait(timeout=30)
                self.assertEqual(code, 128 + signum if signum else 9, (work / 'runner.log').read_text())
                ids = [p.read_text().strip() for p in (work / 'builds').glob('*/container.cid')]
                for owned in ids:
                    self.assertNotEqual(owned, self.peer)
                    inspected = docker('inspect', owned, check=False)
                    self.assertNotEqual(inspected.returncode, 0, 'experiment container survived')
                    self.assertIn('no such', inspected.stderr.lower())
                self.assertEqual(docker('inspect', '--format', '{{.State.Running}}', self.peer).stdout.strip(), 'true')
                self.assertTrue((stage / 'console.log').is_file())
                self.assertTrue((stage / 'cleanup.log').is_file())
                if phase != 'create':
                    self.assertEqual((stage / 'evidence.txt').read_text(), 'retained-evidence')
                else:
                    self.assertFalse((stage / 'evidence.txt').exists(), 'cancelled create started compiling')
                self.assertEqual(len(ids), 3 if phase == 'failure' else 1, 'cancellation started another run')
                proof = {'phase': phase, 'signal': signum, 'process_group': group, 'exit_code': code,
                         'running_before_signal': running,
                         'container_ids': ids, 'all_removed': True, 'peer_running': True,
                         'bind_evidence_retained': True,
                         'console_sha256': hashlib.sha256((stage / 'console.log').read_bytes()).hexdigest()}
                (work / 'proof.json').write_text(json.dumps(proof, indent=2) + '\n')
                print(json.dumps(proof), flush=True)
            finally:
                # Test cleanup after a failed assertion must not leave a sleeper.
                (work / 'release').touch()
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=10)
                for path in (work / 'builds').glob('*/container.cid'):
                    docker('rm', '--force', path.read_text().strip(), check=False)

    def test_terminal_interrupt_removes_running_container(self):
        self.exercise('running-int', signal.SIGINT, group=True)

    def test_terminate_runner_removes_running_container(self):
        self.exercise('running-term', signal.SIGTERM)

    def test_cancel_during_create_never_starts_container(self):
        self.exercise('create', signal.SIGTERM, group=True)

    def test_cancel_during_cleanup_finishes_removal(self):
        self.exercise('cleanup', signal.SIGTERM)

    def test_second_signal_does_not_interrupt_cleanup(self):
        self.exercise('repeat', signal.SIGINT, group=True)

    def test_compile_failure_removes_all_containers(self):
        self.exercise('failure')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docker-image', help='already-local ARM64 image with bash and sleep; never pulled')
    parser.add_argument('--evidence-dir', type=Path)
    args, remaining = parser.parse_known_args()
    IMAGE, EVIDENCE = args.docker_image, args.evidence_dir
    if EVIDENCE:
        EVIDENCE.mkdir(parents=True, exist_ok=False)
    unittest.main(argv=[sys.argv[0], *remaining])
