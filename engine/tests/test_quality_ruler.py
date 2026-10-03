"""Desk fixture controls for the strict bare-tool quality gate."""

import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "quality"))
import quality  # noqa: E402


@pytest.mark.parametrize("flags,expected", [([], True), (["--tool-prose"], False),
                                             (["--tool-stop"], False)])
def test_fake_tool_control_enforces_bare_call(flags, expected):
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    fake = Path(__file__).resolve().parents[1] / "tools" / "quality" / "fake_server.py"
    process = subprocess.Popen([sys.executable, str(fake), "--port", str(port), *flags])
    try:
        for _ in range(40):
            try:
                message, _, finish = quality.chat(f"http://127.0.0.1:{port}", "fake", "Add 17 and -4 using the add tool.",
                                                  128, 1, [{"type": "function", "function": {"name": "add"}}])
                break
            except OSError:
                if process.poll() is not None:
                    raise RuntimeError("fake quality server exited")
                time.sleep(0.05)
        else:
            raise TimeoutError("fake quality server did not start")
        assert quality.bare_tool_call(message, finish, "add", {"a": 17, "b": -4}) is expected
    finally:
        process.terminate()
        process.wait(timeout=3)
