"""Running a training subprocess and turning its output into events.

Every backend that shells out needs the same four things: launch without a
shell, stream stdout line by line, translate lines into events, and stop
cleanly when cancelled. v1 does this once per backend, slightly
differently each time; here it is written down once.

Three details that are easy to get wrong and expensive to debug:

**No shell.** v1 builds a command string and passes ``shell=True``, so a
dataset path with a space or an ampersand in it becomes somebody else's
problem. An argument list has no such failure mode.

**Unbuffered child.** Without ``PYTHONUNBUFFERED``, a Python child writing
to a pipe buffers in 8KB blocks, and a training run appears frozen for
minutes at a time. ``bufsize=1`` on the parent side does nothing about it.

**Terminate, then kill.** A trainer asked to stop needs a moment to close
its checkpoint file. Killing immediately can leave a half-written
safetensors that looks valid until it is loaded.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from ..events import Emitter, Log, no_op, safe

#: How long a cancelled process gets to exit on its own before it is killed.
GRACE = 30.0

#: Callable that turns one output line into events. Returns True if it
#: recognised the line, which is only used to decide whether to also log it.
LineHandler = Callable[[str], bool]


class ProcessRunner:
    """One training subprocess, streamed."""

    def __init__(
        self,
        command: Sequence[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.command = [str(part) for part in command]
        self.cwd = Path(cwd) if cwd else None
        self.env = env
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()
        self.returncode: int | None = None

    # -- lifecycle --------------------------------------------------------

    def run(
        self,
        emit: Emitter = no_op,
        cancel: threading.Event | None = None,
        *,
        on_line: LineHandler | None = None,
    ) -> int:
        """Launch, stream until it exits, and return the exit code.

        Blocking by design - the queue calls this on a worker thread.
        """
        emit = safe(emit)
        environment = {**os.environ, **(self.env or {})}
        environment.setdefault("PYTHONUNBUFFERED", "1")

        emit(Log(line=" ".join(self.command), level="debug"))

        try:
            with self._lock:
                self._process = subprocess.Popen(  # noqa: S603 - argument list, no shell
                    self.command,
                    cwd=str(self.cwd) if self.cwd else None,
                    env=environment,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    **_new_process_group(),
                )
        except OSError as exc:
            emit(Log(line=f"could not start {self.command[0]}: {exc}", level="error"))
            self.returncode = -1
            return -1

        watcher = self._watch(cancel, emit) if cancel is not None else None

        assert self._process.stdout is not None
        try:
            for raw in self._process.stdout:
                line = raw.rstrip("\r\n")
                if not line.strip():
                    continue
                # A trainer's own progress bar rewrites one line with \r; keep
                # only the last segment or the log fills with partial bars.
                line = line.rsplit("\r", 1)[-1]
                if on_line is None or not on_line(line):
                    emit(Log(line=line))
        finally:
            self.returncode = self._process.wait()
            if watcher is not None:
                watcher.set()
            with self._lock:
                self._process = None

        return self.returncode

    def _watch(self, cancel: threading.Event, emit: Emitter) -> threading.Event:
        """Stop the child when the cancel token is set, without blocking reads."""
        finished = threading.Event()

        def wait() -> None:
            while not finished.is_set():
                if cancel.wait(0.25):
                    emit(Log(line="Stopping training", level="warning"))
                    self.stop()
                    return

        threading.Thread(target=wait, name="cancel-watch", daemon=True).start()
        return finished

    def stop(self) -> None:
        """Ask the process to stop, and insist if it will not."""
        with self._lock:
            process = self._process
        if process is None or process.poll() is not None:
            return

        try:
            _interrupt(process)
        except OSError:
            pass

        deadline = time.monotonic() + GRACE
        while time.monotonic() < deadline:
            if process.poll() is not None:
                return
            time.sleep(0.2)

        try:
            process.kill()
        except OSError:
            pass

    def is_running(self) -> bool:
        with self._lock:
            process = self._process
        return process is not None and process.poll() is None


def _new_process_group() -> dict[str, object]:
    """Put the child in its own group, so stopping it does not stop us.

    On Windows that is a creation flag; on POSIX it is ``start_new_session``.
    Without it, a Ctrl+C in the terminal reaches the trainer directly and the
    daemon never gets to record what happened.
    """
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _interrupt(process: subprocess.Popen[str]) -> None:
    """The gentlest stop each platform offers.

    A trainer that handles SIGINT gets to finish writing its checkpoint;
    ``terminate`` is the fallback for one that does not.
    """
    if sys.platform == "win32":
        try:
            process.send_signal(signal.CTRL_BREAK_EVENT)
            return
        except (OSError, ValueError):
            process.terminate()
            return
    process.send_signal(signal.SIGINT)


def python_executable(configured: str = "") -> str:
    """Which interpreter runs the trainer.

    A configured one wins - ai-toolkit usually lives in its own virtual
    environment with a torch build matched to the card, and the daemon's
    interpreter is not it.
    """
    return configured.strip() or sys.executable
