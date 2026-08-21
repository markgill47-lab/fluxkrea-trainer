"""The guard test. The rule the whole architecture rests on.

``core/`` never imports a UI toolkit. Everything the application actually
*does* lives there as plain Python, so it can be wrapped in a daemon and
driven over SSH. v1's ``ImageProcessor`` constructs ``QProgressDialog``
inside its processing loops, which is why nothing in it can be unit
tested, scripted, or run headless - the single most expensive structural
mistake in v1 (doc 01).

This exists before there is much to guard, deliberately: it is far easier
to keep a rule than to reinstate one.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import fluxkrea

CORE = Path(fluxkrea.__file__).resolve().parent / "core"

#: Import a single one of these into ``core/`` and the fleet workflow dies.
UI_TOOLKITS = {
    "PyQt5",
    "PyQt6",
    "PySide2",
    "PySide6",
    "qtpy",
    "tkinter",
    "wx",
    "kivy",
    "gi",
    "pygame",
    "matplotlib",  # not a toolkit, but it drags one in and implies rendering
    "IPython",
}

#: Layering: core is the bottom. It may not reach up into its own clients.
FORBIDDEN_PACKAGES = {
    "fluxkrea.daemon",
    "fluxkrea.cli",
    "fastapi",
    "starlette",
    "uvicorn",
    "flask",
    "aiohttp",
    "requests",
    "httpx",
}

#: OpenCV is a legitimate core dependency for detection, and also ships a
#: complete GUI. Importing it is fine; opening a window is not.
CV2_GUI_CALLS = {
    "imshow",
    "namedWindow",
    "waitKey",
    "startWindowThread",
    "destroyAllWindows",
    "destroyWindow",
    "createTrackbar",
    "selectROI",
}


def core_modules() -> list[Path]:
    return sorted(p for p in CORE.rglob("*.py") if "trainer" not in p.relative_to(CORE).parts)


def test_core_package_exists() -> None:
    assert CORE.is_dir(), f"expected a core package at {CORE}"
    assert core_modules(), "guard test found no modules to guard - check the path"


def absolute_name(module: Path, node: ast.ImportFrom) -> str:
    """Resolve a relative import to its dotted name.

    Without this the guard sees ``from ...daemon.queue import RunSpec`` as a
    relative import and waves it through - which is exactly how core came to
    reference a daemon type once already.
    """
    package = module.parent if module.name == "__init__.py" else module.parent
    for _ in range(node.level - 1):
        package = package.parent
    parts = package.relative_to(CORE.parent.parent).parts
    return ".".join([*parts, node.module]) if node.module else ".".join(parts)


@pytest.mark.parametrize("module", core_modules(), ids=lambda p: p.name)
def test_no_ui_or_client_imports(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    banned = UI_TOOLKITS | FORBIDDEN_PACKAGES

    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.level == 0 and node.module else [absolute_name(module, node)]

        for name in names:
            root = name.split(".")[0]
            offenders = {n for n in banned if root == n.split(".")[0] and name.startswith(n)}
            if root in UI_TOOLKITS:
                offenders.add(root)
            assert not offenders, (
                f"{module.name}:{node.lineno} imports {name!r}. "
                f"core/ stays headless so it can run over SSH - move this to "
                f"daemon/ or a client."
            )


@pytest.mark.parametrize("module", core_modules(), ids=lambda p: p.name)
def test_no_cv2_window_calls(module: Path) -> None:
    tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in CV2_GUI_CALLS
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "cv2"
        ):
            pytest.fail(
                f"{module.name}:{node.lineno} calls cv2.{node.func.attr} - "
                "that opens a window, and the fleet nodes have no display."
            )


def test_the_guard_catches_a_relative_import_into_the_daemon(tmp_path: Path) -> None:
    """The guard has to see through relative imports, or it guards nothing."""
    offender = CORE / "backends" / "spec.py"
    tree = ast.parse("from ...daemon.queue import RunSpec", filename=str(offender))
    node = next(n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom))
    assert absolute_name(offender, node) == "fluxkrea.daemon.queue"


def test_core_imports_cleanly_without_ui_modules(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing core with every UI toolkit poisoned must still work.

    Catches a lazy import inside a function, which the AST walk above
    would see but a future refactor might hide behind an alias.
    """
    for name in UI_TOOLKITS:
        monkeypatch.setitem(sys.modules, name, None)

    for module in core_modules():
        rel = module.relative_to(CORE.parent.parent)
        dotted = ".".join(rel.with_suffix("").parts)
        if dotted.endswith(".__init__"):
            dotted = dotted[: -len(".__init__")]
        __import__(dotted)
