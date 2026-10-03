"""Updates for this vendored engine are managed by the enclosing JSpark3 release."""

INSTALL_URL = "https://github.com/jakejharris/jspark3/blob/release/v2.0.1/INSTALL.md"


def check_in_background():
    """Keep startup offline; a recipe update must replace the complete release."""
    return None


def first_run_notice():
    """The enclosing recipe supplies release notes."""
    return None


def update(*, check_only=False, force=False):
    """Explain the release update procedure without fetching or installing anything."""
    print(f"[tensorfold] This engine is managed by JSpark3. Follow {INSTALL_URL} "
          "to update the complete release; no update was checked or installed.")
    return 0 if check_only else 1
