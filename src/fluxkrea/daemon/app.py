"""The HTTP app.

Nothing here but wiring: the state object holds what a daemon *is*, the
routers hold what it *serves*, and this puts them together with the two
cross-cutting concerns - the token check and turning ``Denied`` into a
status code.

Run it with ``fk serve``, or as a systemd user unit on a fleet node. It
binds 127.0.0.1 by default and refuses to bind wider without a token
(doc 06, "security").
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from .. import __version__
from ..core.config import Config, load
from . import web
from .routes import datasets, jobs, node, tasks
from .security import Denied, check_token, extract_token
from .state import State

API = "/api/v1"

#: Reachable without a token, so a monitoring probe or a tunnel check does
#: not need credentials to answer "is this thing up".
OPEN_PATHS = frozenset({f"{API}/health", "/docs", "/openapi.json", "/redoc"})


def create_app(config: Config | None = None, state: State | None = None) -> FastAPI:
    """Build an app. Pass *state* to share a registry or queue with a test."""
    resolved = state if state is not None else State(config=config or load())

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        resolved.shutdown()

    app = FastAPI(
        title="FluxKrea Trainer",
        version=__version__,
        summary="Dataset tooling and LoRA training for one GPU node",
        lifespan=lifespan,
    )
    app.state.fluxkrea = resolved

    @app.middleware("http")
    async def authenticate(request: Request, call_next):  # noqa: ANN202
        if request.url.path not in OPEN_PATHS:
            try:
                check_token(resolved.config, extract_token(request.headers))
            except Denied as denied:
                return JSONResponse({"error": str(denied)}, status_code=denied.status)
        return await call_next(request)

    @app.exception_handler(Denied)
    async def denied_handler(request: Request, exc: Denied) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.exception_handler(NotADirectoryError)
    async def missing_folder(request: Request, exc: NotADirectoryError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(ValueError)
    async def bad_value(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    for router in (node.router, datasets.router, tasks.router, jobs.router):
        app.include_router(router, prefix=API)

    @app.get("/api", include_in_schema=False)
    def identity() -> dict[str, Any]:
        return {
            "name": "fluxkrea",
            "version": __version__,
            "node": resolved.node_name,
            "api": API,
            "docs": "/docs",
        }

    # The client arrives with the daemon (doc 02), so it is served last -
    # after every API route, because its catch-all would otherwise swallow
    # them. With no build present this is a no-op and `/` stays JSON.
    served = web.mount(app, API)
    if served is None:

        @app.get("/", include_in_schema=False)
        def root() -> dict[str, Any]:
            return {
                "name": "fluxkrea",
                "version": __version__,
                "node": resolved.node_name,
                "api": API,
                "docs": "/docs",
                "client": "not built - run `npm run build` in web/",
            }

    app.state.web_root = served
    return app


def serve(config: Config | None = None) -> None:
    """Run the daemon. Refuses a non-loopback bind without a token."""
    import uvicorn

    from .security import check_bind

    resolved = config or load()
    check_bind(resolved)

    uvicorn.run(
        create_app(resolved),
        host=resolved.daemon.host,
        port=resolved.daemon.port,
        log_level=resolved.log_level,
        access_log=resolved.log_level == "debug",
    )
