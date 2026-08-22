"""Who may call, and which paths they may name.

The API launches processes and rewrites dataset folders. It is effectively
remote code execution scoped to the node, and doc 06 sets three rules:

* **Binds 127.0.0.1 by default.** Remote access is an SSH tunnel, which
  matches the existing workflow and needs no new auth story.
* **Binding beyond loopback requires a token.** The daemon refuses to
  start listening wider without one - no silent open port.
* **No secrets in config files.** The token comes from ``FLUXKREA_TOKEN``
  or the OS keyring, never from the committed config.

The fourth rule is not in the doc but follows from the first three: a
request may not name a path outside the configured dataset roots. Without
that, ``POST /datasets`` with ``{"path": "C:/"}`` is a file browser for
the whole machine, and the tunnel does not help.
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

from ..core import paths
from ..core.config import Config, secret

LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"})

#: Header the CLI sends. ``Authorization: Bearer <token>`` is accepted too,
#: because that is what every HTTP tool already knows how to do.
TOKEN_HEADER = "x-fluxkrea-token"


class Denied(Exception):
    """A request that will not be served. Carries the status to return."""

    def __init__(self, message: str, status: int = 403) -> None:
        super().__init__(message)
        self.status = status


def is_loopback(host: str) -> bool:
    return host.strip().lower() in LOOPBACK


def required_token(config: Config) -> str | None:
    """The token this daemon expects, or ``None`` if it wants none.

    A token set while bound to loopback is still honoured - some people
    want it, and refusing to check a token that exists would be strange.
    """
    return secret("token")


def lab_mode(config: Config) -> bool:
    """Is this node deliberately serving a room, with no token?

    The one sanctioned way past :func:`check_bind`. It is a config-file
    setting rather than an API one on purpose: turning it on is a decision
    made by somebody with a shell on the node, not by anything that arrives
    over the wire.
    """
    return bool(getattr(config.daemon, "lab_mode", False))


def check_bind(config: Config) -> None:
    """Refuse to start listening beyond localhost without a token.

    Unless ``daemon.lab_mode`` is set, which is a classroom of students on
    one LAN node reaching it by URL. That trades the token for the network
    being the boundary, so it also insists on ``dataset.roots``: without
    them the path check is a no-op and the API reaches the whole disk.
    """
    if is_loopback(config.daemon.host) or required_token(config):
        return

    if lab_mode(config):
        if not config.dataset.roots:
            raise Denied(
                f"refusing to bind {config.daemon.host}:{config.daemon.port} in lab mode "
                "with no dataset.roots set. Lab mode drops the token, so the roots are "
                "the only thing scoping what this API can read and write - and empty "
                "roots mean the whole machine. Set dataset.roots in config.toml.",
                status=500,
            )
        return

    raise Denied(
        f"refusing to bind {config.daemon.host}:{config.daemon.port} without a token. "
        "Set FLUXKREA_TOKEN, bind 127.0.0.1 and reach it through an SSH tunnel, or "
        "set daemon.lab_mode = true if this is a lab node on a trusted network.",
        status=500,
    )


def lan_addresses(port: int) -> list[str]:
    """URLs this node is reachable at from the rest of the network.

    Printed at startup for the lab case: "log in at this URL" is the whole
    student-facing instruction, and asking a tutor to find the node's IP
    themselves is how a class starts twenty minutes late.
    """
    import socket

    found: set[str] = set()
    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            address = info[4][0]
            if not address.startswith("127."):
                found.add(address)
    except OSError:
        pass

    if not found:
        # No usable answer from the resolver - a node with no DNS entry for
        # its own name, which is common. Ask the routing table instead by
        # opening an unconnected UDP socket; nothing is sent.
        try:
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                probe.connect(("192.0.2.1", 9))  # TEST-NET-1: routed nowhere
                found.add(probe.getsockname()[0])
            finally:
                probe.close()
        except OSError:
            pass

    return [f"http://{address}:{port}" for address in sorted(found)]


def check_token(config: Config, presented: str | None) -> None:
    """Constant-time token check. No-op when no token is configured."""
    expected = required_token(config)
    if not expected:
        return
    if not presented or not hmac.compare_digest(presented, expected):
        raise Denied("a valid token is required", status=401)


def extract_token(headers: dict[str, str] | object) -> str | None:
    """Pull a token from either accepted header shape."""
    get = getattr(headers, "get", None)
    if get is None:
        return None
    direct = get(TOKEN_HEADER)
    if direct:
        return str(direct).strip()
    auth = get("authorization")
    if auth and str(auth).lower().startswith("bearer "):
        return str(auth)[7:].strip()
    return None


# --------------------------------------------------------------------------
# path scoping
# --------------------------------------------------------------------------


def check_path(config: Config, candidate: str | os.PathLike[str]) -> Path:
    """Resolve a client-supplied path, refusing anything outside the roots.

    With no roots configured the check is skipped, which is only reasonable
    because the default bind is loopback. Configure ``dataset.roots`` on
    any node that listens wider - and the bind check above already forces a
    token in that case.
    """
    target = paths.expand(candidate)
    roots = config.dataset.roots
    if not roots:
        return target

    if any(paths.is_within(target, root) for root in roots):
        return target

    listed = ", ".join(root.as_posix() for root in roots)
    raise Denied(f"{target.as_posix()} is outside the configured dataset roots ({listed})")
