"""Open a folder in the node's file manager.

Deliberately in ``daemon/`` and not in ``core/``. Core is headless so it
can run over SSH on a machine with no display; opening a file manager is
the exact opposite of that, and the guard test would be right to reject
it. This is a node-local convenience, and it belongs on the side of the
line that already knows about the machine it is running on.

**It opens on the node, not on the machine looking at the browser.** Over
an SSH tunnel those are different computers, and a window opening on a
fleet node in another room helps nobody - so the API reports the path back
either way, and the client shows it.
"""

from __future__ import annotations

import os
import platform
import subprocess
from pathlib import Path

#: The file manager per platform, given a folder.
OPENERS = {
    "Windows": ("explorer",),
    "Darwin": ("open",),
    "Linux": ("xdg-open",),
}


def open_folder(folder: Path) -> tuple[bool, str]:
    """Show *folder* in the node's file manager. Returns ``(opened, why)``.

    Never raises: a node with no desktop is an ordinary state on this
    fleet, not an error, and the caller still wants the path.
    """
    if not folder.is_dir():
        return False, f"{folder.as_posix()} is not there"

    system = platform.system()
    opener = OPENERS.get(system)
    if not opener:
        return False, f"no file manager known for {system or 'this platform'}"

    if system == "Linux" and not os.environ.get("DISPLAY") and not os.environ.get(
        "WAYLAND_DISPLAY"
    ):
        # The usual case on a fleet node: headless, so there is nothing to
        # open onto. Worth saying rather than spawning a process that fails.
        return False, "this node has no display, so there is nothing to open it on"

    try:
        # No shell, and the path is passed as an argument rather than
        # interpolated - a folder name is operator data, and a run name can
        # come from a dataset folder somebody else created.
        subprocess.Popen(  # noqa: S603 - fixed opener, path as an argv element
            [*opener, str(folder)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as exc:
        return False, f"could not open a file manager: {exc}"

    return True, f"opened on {platform.node()}"


__all__ = ["open_folder"]
