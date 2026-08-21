"""A daemon for the life of one command.

The CLI is an API client (doc 06), and the API is where all the logic
lives (doc 02). On a desktop with no daemon running, that would mean
``fk dataset scan ./poses`` fails until somebody starts a service - which
is a bad trade for a single-machine user, and the usual response is to add
a "local mode" that calls ``core`` directly. That local mode is exactly
the divergence doc 02 forbids.

So instead: start the real daemon, on an ephemeral loopback port, in a
background thread, for the duration of the command. Same app, same routes,
same socket, same serialisation - just a shorter life. Nothing about the
request path differs from talking to a lab box.

It binds ``127.0.0.1`` with port 0, so the OS picks an unused port and
nothing is reachable from off the machine.
"""

from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING

from ..core.config import Config

if TYPE_CHECKING:  # pragma: no cover
    import uvicorn

    from ..daemon.state import State

#: How long to wait for the server to bind before giving up. Binding a
#: loopback socket is immediate; this is a deadlock guard, not a timeout.
STARTUP_TIMEOUT = 20.0


class EmbeddedDaemon:
    """A uvicorn server on an ephemeral port, started and stopped in process."""

    def __init__(self, config: Config, state: State | None = None) -> None:
        self.config = config
        #: A prepared state, for callers that need to own the registry or the
        #: queue - two nodes in one process, which is what a fleet test is.
        self.state = state
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> str:
        import uvicorn

        from ..daemon.app import create_app

        settings = uvicorn.Config(
            create_app(self.config, self.state),
            host="127.0.0.1",
            port=0,
            log_level="warning",
            access_log=False,
            lifespan="on",
        )
        self._server = uvicorn.Server(settings)
        # Uvicorn installs signal handlers by default, which only works on the
        # main thread and would steal Ctrl+C from the CLI besides.
        self._server.install_signal_handlers = lambda: None  # type: ignore[method-assign]

        self._thread = threading.Thread(target=self._server.run, name="fk-embedded", daemon=True)
        self._thread.start()

        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._server.started and self._server.servers:
                sockets = self._server.servers[0].sockets
                if sockets:
                    self.port = sockets[0].getsockname()[1]
                    return self.base_url
            if not self._thread.is_alive():
                break
            time.sleep(0.01)

        self.stop()
        raise RuntimeError("the embedded daemon did not start")

    def stop(self, timeout: float = 10.0) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout)
        self._server = None
        self._thread = None
