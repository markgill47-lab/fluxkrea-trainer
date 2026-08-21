"""Configuration and captioner endpoints - what the settings screen talks to.

Two rules shape this module, and both are refusals.

**Not every setting is editable over HTTP.** ``dataset.roots`` is the
allow-list every path check is measured against, and ``daemon.host`` is
what keeps this API on loopback. A client that can widen either has
widened the API's reach, not merely changed a preference - so those are
edited in ``config.toml`` on the node, by someone with a shell on it.
Everything else is fair game: the daemon already launches training
processes for an authenticated caller, and refusing to let that same
caller set a mask feather would be theatre.

**Secrets never arrive here.** Doc 05 makes a secret in ``config.toml`` a
hard error at load; accepting one over the API and writing it there would
route around that. An API key comes from the environment or the OS keyring
on the node it is used from, and ``GET /config/secrets`` reports only
whether each one was found.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Body, Depends

from ...core.captioners import (
    CaptionerError,
    available as captioners_available,
    from_config as build_captioner,
    labels as captioner_labels,
)
from ...core.config import SECRET_ENV, ConfigError, secret
from ..security import Denied
from ..state import State
from .deps import get_state

router = APIRouter(tags=["settings"])

#: Sections a client may write, and within them the keys it may not.
#: Absent from this map means the whole section is read-only.
EDITABLE: dict[str, frozenset[str]] = {
    "captioner": frozenset(),
    "mask": frozenset(),
    "backends": frozenset(),
    # roots scopes every path check in the daemon (see the module docstring).
    "dataset": frozenset({"roots"}),
}

#: Editable, but only read at startup - worth saying so in the response
#: rather than letting someone wonder why nothing changed.
RESTART_REQUIRED = ("backends.python_exe", "daemon.workers")


@router.get("/config")
def get_config(state: State = Depends(get_state)) -> dict[str, Any]:
    """The resolved config. Never contains secrets, by construction."""
    return _payload(state)


@router.put("/config")
def put_config(
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Apply a flat map of dotted settings, then write ``config.toml``.

    Flat rather than nested on purpose: a nested body cannot distinguish
    "leave this alone" from "set this to its default", and a settings
    screen that saves one field should not silently rewrite the rest.
    """
    updates = dict(payload.get("set") or payload or {})
    updates.pop("set", None)
    if not updates:
        raise Denied("nothing to set", status=400)

    for dotted in updates:
        _check_editable(str(dotted))

    # Validate against a copy, so a rejected change leaves the running
    # daemon on the config it started with rather than half a new one.
    from copy import deepcopy

    candidate = deepcopy(state.config)
    for dotted, value in updates.items():
        try:
            candidate.set(str(dotted), value)
        except (KeyError, ConfigError, TypeError, ValueError) as exc:
            raise Denied(f"{dotted}: {exc}", status=422) from exc

    problems = candidate.validate()
    if problems:
        raise Denied("; ".join(problems), status=422)

    for dotted, value in updates.items():
        state.config.set(str(dotted), value)

    try:
        written = state.config.save()
    except OSError as exc:
        raise Denied(f"cannot write the config file: {exc}", status=500) from exc

    body = _payload(state)
    body["written"] = written.as_posix()
    body["changed"] = sorted(str(k) for k in updates)
    body["restart_required"] = [k for k in body["changed"] if k in RESTART_REQUIRED]
    return body


@router.get("/config/secrets")
def get_secrets(state: State = Depends(get_state)) -> dict[str, Any]:
    """Which secrets this node can find, and where it looked. Never values."""
    return {
        "secrets": [
            {"name": name, "found": secret(name) is not None, "env": list(env)}
            for name, env in SECRET_ENV.items()
        ]
    }


# --------------------------------------------------------------------------
# captioners
# --------------------------------------------------------------------------


@router.get("/captioners")
def list_captioners(state: State = Depends(get_state)) -> dict[str, Any]:
    """What this node could caption with, without probing anything.

    ``available`` answers "could this be built", not "would it work" -
    building a captioner must not cost a network round trip. ``POST
    /captioners/test`` is the question that costs one.
    """
    ready = captioners_available()
    labels = captioner_labels()
    return {
        "captioners": [
            {"name": name, "label": labels.get(name, name), "available": ok}
            for name, ok in ready.items()
        ],
        "configured": state.config.captioner.provider,
    }


@router.post("/captioners/test")
def test_captioner(
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Probe a captioner and report what it said, success or not.

    Returns 200 with ``ok: false`` rather than an error status: a stopped
    Ollama daemon is an answer to the question that was asked, and the
    settings screen wants to render the message either way.
    """
    overrides = {
        key: payload[key]
        for key in ("provider", "url", "model", "timeout")
        if payload.get(key) is not None
    }
    try:
        captioner = build_captioner(state.config.captioner, **overrides)
    except CaptionerError as exc:
        return {"ok": False, "message": str(exc), "provider": overrides.get("provider")}
    except TypeError as exc:
        raise Denied(f"option not valid for this captioner: {exc}", status=422) from exc

    ok, message = captioner.test()
    body: dict[str, Any] = {"ok": ok, "message": message, "provider": captioner.name}

    # The one probe worth reporting in detail: knowing which models *are*
    # pulled turns "model not found" from a guessing game into a choice.
    installed = getattr(captioner, "installed_models", None)
    if callable(installed):
        found, models = installed()
        if found:
            body["models"] = models

    captioner.close()
    return body


# --------------------------------------------------------------------------
# saved prompts
# --------------------------------------------------------------------------


@router.get("/captioners/prompts")
def list_prompts(state: State = Depends(get_state)) -> dict[str, Any]:
    """Every prompt this node knows, built-ins included."""
    return {
        "prompts": [prompt.as_dict() for prompt in state.prompts.all()],
        "file": state.prompts.file.as_posix(),
    }


@router.put("/captioners/prompts/{name}")
def put_prompt(
    name: str,
    payload: dict[str, Any] = Body(default={}),
    state: State = Depends(get_state),
) -> dict[str, Any]:
    """Save a prompt under a name, replacing any prompt already there."""
    try:
        saved = state.prompts.save(name, str(payload.get("text", "")))
    except ValueError as exc:
        raise Denied(str(exc), status=422) from exc
    except OSError as exc:
        raise Denied(f"cannot write the prompt file: {exc}", status=500) from exc
    return saved.as_dict()


@router.delete("/captioners/prompts/{name}")
def delete_prompt(name: str, state: State = Depends(get_state)) -> dict[str, Any]:
    """Delete a saved prompt. A shadowed built-in reappears underneath."""
    try:
        removed = state.prompts.delete(name)
    except OSError as exc:
        raise Denied(f"cannot write the prompt file: {exc}", status=500) from exc
    if not removed:
        # Either it never existed or it is a built-in nobody saved over -
        # and a built-in is not deletable, which is worth saying plainly.
        raise Denied(
            f"no saved prompt named {name!r}"
            + (" - built-in prompts cannot be deleted" if name in state.prompts else ""),
            status=404,
        )
    return {"name": name, "deleted": True, "restored": state.prompts.get(name) is not None}


def _payload(state: State) -> dict[str, Any]:
    config = state.config.as_dict()
    return {
        **config,
        "source": str(state.config.source) if state.config.source else None,
        "read_only": _read_only(),
    }


def _read_only() -> list[str]:
    """Settings the API will not write, so the UI can say why, not just refuse."""
    locked = ["daemon.*"]
    locked.extend(f"{section}.{key}" for section, keys in EDITABLE.items() for key in sorted(keys))
    return sorted(locked)


def _check_editable(dotted: str) -> None:
    section, _, name = dotted.partition(".")
    if not name:
        raise Denied(f"{dotted!r} is a section; set individual keys", status=400)
    if section not in EDITABLE:
        raise Denied(
            f"{section}.* is not editable over the API - it decides what this "
            f"daemon will reach and who can reach it. Edit config.toml on the "
            f"node and restart.",
            status=403,
        )
    if name in EDITABLE[section]:
        raise Denied(
            f"{dotted} is not editable over the API - it scopes every path "
            f"check this daemon makes. Edit config.toml on the node and restart.",
            status=403,
        )
