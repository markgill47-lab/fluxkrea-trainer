"""Serving the browser client.

Doc 02's reason for choosing a web client at all: "nothing to install on
any machine you drive from — the client arrives with the daemon, reached
through the same SSH tunnel as the CLI." That only holds if the daemon
actually serves it, which is this module.

The client is a single-page app, so any path that is not an API route and
not a real file has to return ``index.html`` and let the client route it.
The API is mounted first, so it always wins.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.requests import Request

#: Where a built client is looked for, in order. The installed location
#: comes first so a wheel serves its own bundled copy rather than a stale
#: development build that happens to be on the same machine.
# From src/fluxkrea/daemon/: up to the package, then out of src/ to
# the repository root where a development build lives.
CANDIDATES = ("_web", "../../../web/dist")

#: Assets are content-hashed by the build, so they can be cached hard.
#: ``index.html`` must not be, or a deploy is invisible until a hard reload.
IMMUTABLE = "public, max-age=31536000, immutable"


def find_client(explicit: str | os.PathLike[str] | None = None) -> Path | None:
    """Locate a built client, or ``None`` if it has not been built."""
    if explicit:
        candidate = Path(explicit)
        return candidate if (candidate / "index.html").is_file() else None

    if (env := os.environ.get("FLUXKREA_WEB_DIR", "").strip()):
        candidate = Path(env)
        return candidate if (candidate / "index.html").is_file() else None

    here = Path(__file__).resolve().parent
    for relative in CANDIDATES:
        candidate = (here / relative).resolve()
        if (candidate / "index.html").is_file():
            return candidate
    return None


class ClientFiles(StaticFiles):
    """The build's asset directory, cached hard.

    Everything under it is content-hashed by the bundler, so the filename
    changes whenever the bytes do and the response can be immutable. The
    mount is scoped to that directory, so there is nothing here that is not
    an asset - the paths arriving are already relative to it.
    """

    async def get_response(self, path: str, scope):  # noqa: ANN001, ANN201
        response = await super().get_response(path, scope)
        response.headers["cache-control"] = IMMUTABLE
        return response


def mount(app: FastAPI, api_prefix: str, directory: Path | None = None) -> Path | None:
    """Serve the client at ``/``, leaving the API untouched.

    Returns the directory being served, or ``None`` when there is no build -
    in which case ``/`` keeps returning the JSON identity document, and the
    daemon is perfectly usable from ``fk`` and ``curl``.
    """
    root = find_client(directory)
    if root is None:
        return None

    app.mount("/assets", ClientFiles(directory=root / "assets"), name="assets")

    index = root / "index.html"

    @app.get("/{full_path:path}", include_in_schema=False)
    async def spa(request: Request, full_path: str):  # noqa: ANN202
        # The API is mounted before this and matches first, but a mistyped
        # API path would otherwise fall through and return the app shell
        # with a 200 - which looks to a client like a broken JSON parse
        # rather than a 404.
        if full_path.startswith(api_prefix.strip("/")):
            return JSONResponse({"error": f"no such endpoint: /{full_path}"}, status_code=404)

        direct = (root / full_path).resolve() if full_path else index
        if full_path and direct.is_file() and root.resolve() in direct.parents:
            return FileResponse(direct)

        return FileResponse(index, headers={"cache-control": "no-cache"})

    return root
