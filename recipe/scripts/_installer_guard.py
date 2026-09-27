"""Refuse to run a sealed installer that could reach the live tree.

Every sealed Mia/vLLM installer falls back to a live image path
(``/usr/local/lib/python3.12/dist-packages/...`` or ``/opt/glm53/...``) when
its override is unset. Inside the container that default is the real package
root, so one missing override would edit it outside the hash transaction.
``require_overrides`` reads the installer's source, finds every environment
variable or CLI option whose default names a live path, and refuses unless the
caller supplies all of them. A live path literal that no override can redirect
refuses outright.
"""

from __future__ import annotations

import ast
from pathlib import Path

LIVE_PREFIXES = ("/usr/local/lib/python3.12/dist-packages", "/opt/glm53")


def _strings(node: ast.AST, constants: dict[str, str]) -> list[str]:
    """String constants in ``node``, with module-level string names resolved."""
    found = []
    for child in ast.walk(node):
        if isinstance(child, ast.Constant) and isinstance(child.value, str):
            found.append(child.value)
        elif isinstance(child, ast.Name) and child.id in constants:
            found.append(constants[child.id])
    return found


def _live(node: ast.AST, constants: dict[str, str]) -> bool:
    return any(text.startswith(LIVE_PREFIXES) for text in _strings(node, constants))


def path_knobs(source: str) -> set[str]:
    """Env vars and CLI options through which ``source`` locates a live path."""
    tree = ast.parse(source)
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) \
                and isinstance(node.value.value, str):
            constants.update({t.id: node.value.value for t in node.targets if isinstance(t, ast.Name)})
    knobs: set[str] = set()
    covered: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        name = node.func.attr
        default = None
        if name in ("get", "getenv") and len(node.args) >= 2:
            default = node.args[1]
        elif name == "add_argument":
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
        if default is None or not node.args or not isinstance(node.args[0], ast.Constant) \
                or not _live(default, constants):
            continue
        knobs.add(node.args[0].value)
        covered.update(id(child) for child in ast.walk(default))
    docstrings = {id(node.value) for node in ast.walk(tree)
                  if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant)}
    named = {name for name, text in constants.items() if text.startswith(LIVE_PREFIXES)}
    for node in ast.walk(tree):
        loose = (isinstance(node, ast.Constant) and isinstance(node.value, str)
                 and node.value.startswith(LIVE_PREFIXES) and id(node) not in covered
                 and id(node) not in docstrings and not _is_named_constant(tree, node, named))
        loose = loose or (isinstance(node, ast.Name) and node.id in named and id(node) not in covered
                          and isinstance(node.ctx, ast.Load))
        if loose:
            raise ValueError(f"live path at line {node.lineno} has no override")
    return knobs


def _is_named_constant(tree: ast.Module, node: ast.AST, named: set[str]) -> bool:
    return any(isinstance(top, ast.Assign) and top.value is node
               and any(isinstance(t, ast.Name) and t.id in named for t in top.targets)
               for top in tree.body)


def require_overrides(installer: Path, supplied) -> None:
    """Raise ValueError unless ``supplied`` covers every live-path knob."""
    try:
        knobs = path_knobs(installer.read_text(encoding="utf-8"))
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"{installer.name}: {exc}") from None
    missing = sorted(knobs - set(supplied))
    if missing:
        raise ValueError(f"{installer.name}: live-path override not supplied: {', '.join(missing)}")
