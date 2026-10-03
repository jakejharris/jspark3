"""The vendored engine must never fetch or install a different TensorFold."""

import subprocess
import urllib.request

import pytest

from tensorfold import cli, update


@pytest.mark.parametrize("setting", [None, "0", "1"])
def test_startup_and_update_stay_offline(monkeypatch, capsys, setting):
    def forbidden(*args, **kwargs):
        pytest.fail("update attempted network or process execution")

    monkeypatch.setattr(urllib.request, "urlopen", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    if setting is None:
        monkeypatch.delenv("TENSORFOLD_NO_UPDATE_CHECK", raising=False)
    else:
        monkeypatch.setenv("TENSORFOLD_NO_UPDATE_CHECK", setting)
    assert update.check_in_background() is None
    assert update.first_run_notice() is None
    assert cli.main(["update", "--check"]) == 0
    assert cli.main(["update"]) == 1
    assert cli.main(["update", "--force"]) == 1
    assert update.INSTALL_URL in capsys.readouterr().out


def test_cli_still_accepts_no_update_check():
    assert cli.build_parser().parse_args(["serve", "owner/model", "--no-update-check"]).no_update_check
