"""Turning core events into terminal output.

This is the CLI's listener - the thing the daemon replaces with an SSE
fan-out. Core emits the same events either way (doc 02).
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from typing import Any, TextIO

from ..core.events import Emitter, Event, Finished, Log, LossPoint, Progress

LEVEL_MARK = {"debug": "  ", "info": "  ", "warning": "! ", "error": "x "}


class Console:
    """Progress on one rewritten line, logs above it. No dependencies."""

    def __init__(
        self,
        stream: TextIO | None = None,
        *,
        quiet: bool = False,
        verbose: bool = False,
        interactive: bool | None = None,
    ) -> None:
        self.stream = stream if stream is not None else sys.stderr
        self.quiet = quiet
        self.verbose = verbose
        self.interactive = self.stream.isatty() if interactive is None else interactive
        self._bar_width = 0
        self._last_paint = 0.0

    # -- emitter ----------------------------------------------------------

    def __call__(self, event: Event) -> None:
        if isinstance(event, Progress):
            self._progress(event)
        elif isinstance(event, Log):
            self._log(event)
        elif isinstance(event, Finished):
            self._clear()
            self.write(("done: " if event.ok else "failed: ") + event.detail)
        elif isinstance(event, LossPoint):
            if self.verbose:
                self.write(f"step {event.step}: loss {event.value:.5f}")

    def as_emitter(self) -> Emitter:
        return self

    def event(self, payload: dict[str, Any]) -> None:
        """Render a decoded SSE frame.

        The wire form of the same dataclasses core emits, so this is the
        remote half of ``__call__`` - one vocabulary from the training loop
        to the terminal (doc 06).
        """
        kind = payload.get("kind")
        if kind == "progress":
            self._progress(
                Progress(
                    step=int(payload.get("step", 0)),
                    total=int(payload.get("total", 0)),
                    message=str(payload.get("message", "")),
                )
            )
        elif kind == "log":
            self._log(Log(line=str(payload.get("line", "")), level=payload.get("level", "info")))
        elif kind == "loss":
            if self.verbose:
                self.write(f"step {payload.get('step')}: loss {payload.get('value')}")
        elif kind == "finished":
            self._clear()
            ok = bool(payload.get("ok"))
            detail = str(payload.get("detail", ""))
            self.write(("done: " if ok else "failed: ") + detail)

    # -- pieces -----------------------------------------------------------

    def _log(self, event: Log) -> None:
        if self.quiet and event.level not in ("warning", "error"):
            return
        if event.level == "debug" and not self.verbose:
            return
        self._clear()
        self.write(f"{LEVEL_MARK.get(event.level, '  ')}{event.line}")

    def _progress(self, event: Progress) -> None:
        if self.quiet or not self.interactive or event.total <= 0:
            return
        now = time.monotonic()
        final = event.step >= event.total
        if not final and now - self._last_paint < 0.08:
            return
        self._last_paint = now

        width = max(20, min(shutil.get_terminal_size((80, 24)).columns - 34, 40))
        filled = int(width * event.fraction)
        bar = "#" * filled + "-" * (width - filled)
        text = f"  [{bar}] {event.step}/{event.total} {event.message}".rstrip()
        self._bar_width = len(text)
        self.stream.write("\r" + text)
        self.stream.flush()
        if final:
            self._clear()

    def _clear(self) -> None:
        if self._bar_width and self.interactive:
            self.stream.write("\r" + " " * self._bar_width + "\r")
            self.stream.flush()
            self._bar_width = 0

    def write(self, line: str = "") -> None:
        self.stream.write(line + "\n")
        self.stream.flush()


def emit_json(payload: Any, stream: TextIO | None = None) -> None:
    """Machine-readable output, on stdout, so it can be piped."""
    target = stream if stream is not None else sys.stdout
    json.dump(payload, target, indent=2, default=str)
    target.write("\n")
    target.flush()


def table(rows: list[tuple[str, ...]], headers: tuple[str, ...] = ()) -> str:
    """A plain aligned table. No box drawing - this gets piped and grepped."""
    body = [headers, *rows] if headers else list(rows)
    if not body:
        return ""
    widths = [max(len(str(row[col])) for row in body) for col in range(len(body[0]))]
    lines = []
    for index, row in enumerate(body):
        lines.append("  ".join(str(cell).ljust(widths[col]) for col, cell in enumerate(row)).rstrip())
        if headers and index == 0:
            lines.append("  ".join("-" * width for width in widths))
    return "\n".join(lines)
