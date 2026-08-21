"""Shared route dependencies.

The state lives on the ASGI app, not in a module global, so two daemons in
one process - which is exactly what the test suite does - never share a
registry or a queue.
"""

from __future__ import annotations

from fastapi import Request

from ..state import State


def get_state(request: Request) -> State:
    return request.app.state.fluxkrea
