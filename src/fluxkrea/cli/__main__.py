"""``fk`` - a full API client, not a shortcut layer.

Every GUI action has to have a command, because both go through the same
endpoints; a feature reachable only by clicking makes the fleet
second-class (doc 06). Nothing in this module touches a dataset folder
directly - it parses arguments, calls the API, and prints what comes back.

Where the request goes:

* ``--node NAME`` looks the node up in ``fleet.toml``.
* ``--url http://...`` addresses one directly, which over an SSH tunnel is
  a ``localhost`` port.
* Neither, and a daemon is listening locally, uses it.
* Neither, and nothing is listening, starts one for the command's lifetime.

Exit codes:

* ``0`` success
* ``1`` the operation ran and reported a problem (validation errors,
  refused export, failed items)
* ``2`` bad usage, a configuration error, or a node that will not answer
* ``130`` interrupted
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import __version__
from ..core import paths
from ..core.config import Config, ConfigError, load
from .client import ApiError, Client
from .fleet import Fleet
from .output import Console, emit_json, table

OK, PROBLEM, USAGE, INTERRUPTED = 0, 1, 2, 130


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------


def _common_flags() -> argparse.ArgumentParser:
    """Flags accepted on either side of the subcommand.

    ``SUPPRESS`` as the default matters: without it the subparser would
    write its own default over a value the top-level parser already read,
    so ``fk --json dataset scan`` would silently lose the flag.
    """
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", default=argparse.SUPPRESS,
                        help="machine-readable output on stdout")
    common.add_argument("-q", "--quiet", action="store_true", default=argparse.SUPPRESS,
                        help="only warnings and errors")
    common.add_argument("-v", "--verbose", action="store_true", default=argparse.SUPPRESS,
                        help="include debug output")
    common.add_argument("--node", default=argparse.SUPPRESS, help="a node from fleet.toml")
    common.add_argument("--url", default=argparse.SUPPRESS,
                        help="a daemon URL, e.g. http://localhost:8471 over an SSH tunnel")
    return common


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fk",
        description="FluxKrea Trainer - dataset tooling and LoRA training across a fleet",
    )
    parser.add_argument("--version", action="version", version=f"fluxkrea {__version__}")
    parser.add_argument("--config", type=Path, help="config file to use instead of the default")
    parser.add_argument("--json", action="store_true", help="machine-readable output on stdout")
    parser.add_argument("-q", "--quiet", action="store_true", help="only warnings and errors")
    parser.add_argument("-v", "--verbose", action="store_true", help="include debug output")
    parser.add_argument("--node", help="a node from fleet.toml")
    parser.add_argument("--url", help="a daemon URL")

    commands = parser.add_subparsers(dest="command", required=True)
    common = _common_flags()

    _add_serve(commands, common)
    _add_config(commands, common)
    _add_node(commands, common)
    _add_prompts(commands, common)
    _add_dataset(commands, common)
    _add_jobs(commands, common)
    _add_train(commands, common)
    _add_fleet(commands, common)
    return parser


def _add_train(commands: Any, common: argparse.ArgumentParser) -> None:
    train = commands.add_parser("train", parents=[common], help="submit a training run")
    train.add_argument("--spec", type=Path, help="a run spec, as TOML")
    train.add_argument("--model", help="model id, e.g. krea2 or klein-4b")
    train.add_argument("--dataset", help="dataset id or folder on the target node")
    train.add_argument("--name", help="a name for the run")
    train.add_argument("--device", type=int, default=0, help="which GPU to pin to")
    train.add_argument("--steps", type=int, help="training steps")
    train.add_argument("--lr", type=float, help="learning rate")
    train.add_argument("--dim", type=int, help="LoRA rank; FLUX.2 wants 32 or more")
    train.add_argument("--alpha", type=int, help="LoRA alpha; defaults to the rank")
    train.add_argument("--resolution", type=int, help="training resolution")
    train.add_argument("--batch-size", type=int, help="batch size")
    train.add_argument("--save-every", type=int, help="checkpoint interval, in steps")
    train.add_argument("--sample-every", type=int, help="sample interval, in steps")
    train.add_argument("--prompt", action="append", default=[],
                       help="sample prompt; repeatable, needs --sample-every")
    train.add_argument("--masked", action="store_true",
                       help="point the run at the dataset's masks/ folder")
    train.add_argument("--mask-min", type=float,
                       help="ai-toolkit mask_min_value; 0 ignores the region entirely")
    train.add_argument("--watch", action="store_true", help="follow the run instead of returning")


def _add_serve(commands: Any, common: argparse.ArgumentParser) -> None:
    serve = commands.add_parser("serve", parents=[common], help="run the daemon")
    serve.add_argument("--host", help="bind address; a non-loopback bind requires FLUXKREA_TOKEN")
    serve.add_argument("--port", type=int, help="port to listen on")


def _add_config(commands: Any, common: argparse.ArgumentParser) -> None:
    config = commands.add_parser("config", help="show or create the config file")
    actions = config.add_subparsers(dest="action", required=True)
    actions.add_parser("show", parents=[common],
                       help="the resolved configuration, after every override")
    actions.add_parser("path", parents=[common],
                       help="where config, data, cache and state live")
    written = actions.add_parser("init", parents=[common], help="write a starter config file")
    written.add_argument("--force", action="store_true", help="overwrite an existing file")

    # `set` goes through the daemon rather than editing the file directly,
    # so that --node reaches the machine the setting is actually for.
    setter = actions.add_parser("set", parents=[common],
                                help="change settings on a node and save its config file")
    setter.add_argument("pairs", nargs="+", metavar="KEY=VALUE",
                        help="dotted setting, e.g. captioner.provider=ollama")

    actions.add_parser("secrets", parents=[common],
                       help="which API keys a node can find, and where it looked")


def _add_prompts(commands: Any, common: argparse.ArgumentParser) -> None:
    prompts = commands.add_parser("prompts", help="saved caption prompts")
    actions = prompts.add_subparsers(dest="action", required=True)

    listing = actions.add_parser("list", parents=[common], help="every prompt this node knows")
    listing.add_argument("--full", action="store_true", help="print the whole text, not a preview")

    show = actions.add_parser("show", parents=[common], help="one prompt, in full")
    show.add_argument("name")

    save = actions.add_parser("save", parents=[common], help="save or replace a prompt")
    save.add_argument("name")
    save.add_argument("text", nargs="?", help="the prompt; omit to read stdin")

    remove = actions.add_parser("delete", parents=[common], help="delete a saved prompt")
    remove.add_argument("name")


def _add_node(commands: Any, common: argparse.ArgumentParser) -> None:
    node = commands.add_parser("node", help="what a node can do")
    actions = node.add_subparsers(dest="action", required=True)
    actions.add_parser("status", parents=[common],
                       help="platform, versions, GPUs and available detectors")
    actions.add_parser("gpus", parents=[common], help="per-GPU name, VRAM and capability")
    actions.add_parser("models", parents=[common], help="what this node can train")
    probe = actions.add_parser("captioners", parents=[common],
                               help="which captioners a node has, and whether one answers")
    probe.add_argument("--test", action="store_true",
                       help="actually probe the configured captioner - costs a round trip")
    probe.add_argument("--provider", choices=["ollama", "joycaption", "claude"],
                       help="probe this one instead of the configured one")


def _add_dataset(commands: Any, common: argparse.ArgumentParser) -> None:
    dataset = commands.add_parser("dataset", help="dataset operations")
    actions = dataset.add_subparsers(dest="action", required=True)

    def with_target(name: str, help_text: str) -> Any:
        sub = actions.add_parser(name, parents=[common], help=help_text)
        sub.add_argument("path", help="a dataset folder, or a registered dataset id")
        return sub

    actions.add_parser("list", parents=[common], help="datasets registered on the node")

    with_target("scan", "list the bundles in a dataset")
    with_target("register", "add a folder to the node's dataset list")

    validate_cmd = with_target("validate", "report everything wrong, change nothing")
    validate_cmd.add_argument("--require-masks", action="store_true",
                              help="treat a missing mask as an error, for a masked training run")

    resize_cmd = with_target("resize", "fit every image's longest edge to a target")
    resize_cmd.add_argument("--size", type=int, required=True, help="target longest edge, in pixels")
    resize_cmd.add_argument("--output", help="write to another folder instead of in place")
    resize_cmd.add_argument("--no-upscale", action="store_true",
                            help="leave images smaller than the target alone instead of enlarging")

    rename_cmd = with_target("rename", "renumber a dataset onto a prefix")
    rename_cmd.add_argument("prefix", help="new filename prefix")
    rename_cmd.add_argument("--start", type=int,
                            help="first number; default continues past bystanders")
    rename_cmd.add_argument("--digits", type=int, help="zero-padding width")
    rename_cmd.add_argument("--scramble", action="store_true",
                            help="prefix a random letter, to break capture order")
    rename_cmd.add_argument("--seed", type=int, help="make --scramble reproducible")
    rename_cmd.add_argument("--dry-run", action="store_true",
                            help="print the plan and move nothing")

    augment_cmd = with_target("augment", "write flipped and rotated copies")
    augment_cmd.add_argument("--flip", action="store_true", help="horizontal flip")
    augment_cmd.add_argument("--rot-left", action="store_true", help="90 degrees left")
    augment_cmd.add_argument("--rot-right", action="store_true", help="90 degrees right")
    augment_cmd.add_argument("--rot-180", action="store_true", help="180 degrees")
    augment_cmd.add_argument("--duplicate", action="store_true", help="plain copy, no transform")
    augment_cmd.add_argument("--output", help="write to another folder instead of in place")

    caption_cmd = with_target("caption", "write a .txt caption beside every image")
    caption_cmd.add_argument("--provider", choices=["ollama", "joycaption", "claude"],
                             help="captioner to use; default from config")
    caption_cmd.add_argument("--model", help="vision model; default from config")
    # not --url: that is the daemon's address, on every subcommand.
    caption_cmd.add_argument("--ollama-url", help="Ollama base URL; default from config")
    caption_cmd.add_argument("--prompt", help="what to ask the model; default from config")
    caption_cmd.add_argument("--prompt-name", help="a saved prompt, from `fk prompts list`")
    caption_cmd.add_argument("--prefix", help="prepended to every caption - a trigger token")
    caption_cmd.add_argument("--overwrite", action="store_true",
                             help="re-caption images that already have one")
    caption_cmd.add_argument("--max-tokens", type=int, help="ceiling on one caption")
    caption_cmd.add_argument("--timeout", type=float, help="seconds to wait for one image")

    detect_cmd = with_target("detect", "find faces and record the boxes")
    detect_cmd.add_argument("--detector", help="detector name; default from config")
    detect_cmd.add_argument("--confidence", type=float, help="detection threshold; lower finds more")
    detect_cmd.add_argument("--workers", type=int, help="detection threads")
    detect_cmd.add_argument("--only-missing", action="store_true",
                            help="skip images that already have boxes")

    review = with_target("review", "review progress, and the images that need attention")
    review.add_argument("--mark-all-reviewed", action="store_true",
                        help="accept every detection as-is; only after looking at the previews")
    review.add_argument("--mark", action="append", default=[],
                        help="mark one image reviewed, by filename")

    boxes_cmd = with_target("boxes", "inspect or edit the face boxes for one image")
    boxes_cmd.add_argument("image", help="image basename, e.g. punch_014")
    boxes_cmd.add_argument("--add", action="append", default=[], metavar="X,Y,W,H",
                           help="add a manual box")
    boxes_cmd.add_argument("--clear", action="store_true", help="remove every box for this image")
    boxes_cmd.add_argument("--reviewed", action="store_true", help="mark the image reviewed")

    mask_cmd = with_target("mask", "detect, then export masks/*.png the trainer reads")
    mask_cmd.add_argument("--detector", help="detector name; default from config")
    mask_cmd.add_argument("--confidence", type=float, help="detection threshold")
    mask_cmd.add_argument("--expand", type=float, help="box expansion factor; default 1.6")
    mask_cmd.add_argument("--expand-up", type=float, help="extra upward growth, for the hairline")
    mask_cmd.add_argument("--feather", type=int, help="pixels of gradient at the mask boundary")
    mask_cmd.add_argument("--no-detect", action="store_true",
                          help="re-export from the stored boxes only")
    mask_cmd.add_argument("--no-previews", action="store_true", help="skip the redacted previews")
    mask_cmd.add_argument("--force", action="store_true",
                          help="export unreviewed or zero-box images too")

    manifest_cmd = with_target("manifest", "per-file size, mtime and digest")
    manifest_cmd.add_argument("--quick", action="store_true", help="skip digests")
    manifest_cmd.add_argument("--sidecars-only", action="store_true",
                              help="captions and masks only")

    push_cmd = with_target("push", "send a dataset to a node, by manifest diff")
    push_cmd.add_argument("--to", dest="to", help="target node name from fleet.toml")
    push_cmd.add_argument("--sidecars-only", action="store_true",
                          help="captions and masks only - kilobytes, not gigabytes")
    push_cmd.add_argument("--dry-run", action="store_true", help="show the diff and send nothing")
    push_cmd.add_argument("--transport", choices=["auto", "rsync", "tar"], default="auto")
    push_cmd.add_argument("--ssh", help="user@host for rsync")
    push_cmd.add_argument("--remote-path", help="where the dataset should live on the target")
    push_cmd.add_argument("--quick", action="store_true",
                          help="diff on size and mtime instead of digests")

    export_cmd = with_target("export", "stream a dataset out as a tar")
    export_cmd.add_argument("--out", type=Path, required=True, help="tar file to write")
    export_cmd.add_argument("--sidecars-only", action="store_true")

    where = actions.add_parser("where", parents=[common], help="which nodes have a dataset")
    where.add_argument("dataset", help="dataset id")


def _add_jobs(commands: Any, common: argparse.ArgumentParser) -> None:
    jobs = commands.add_parser("jobs", help="training jobs")
    actions = jobs.add_subparsers(dest="action", required=True)
    actions.add_parser("list", parents=[common], help="the queue and its history")

    watch = actions.add_parser("watch", parents=[common], help="tail a job's event stream")
    watch.add_argument("job", help="job id")

    cancel = actions.add_parser("cancel", parents=[common], help="cancel or dequeue a job")
    cancel.add_argument("job", help="job id")

    loss = actions.add_parser("loss", parents=[common], help="the loss series")
    loss.add_argument("job", help="job id")


def _add_fleet(commands: Any, common: argparse.ArgumentParser) -> None:
    fleet = commands.add_parser("fleet", help="every node in fleet.toml at once")
    actions = fleet.add_subparsers(dest="action", required=True)
    actions.add_parser("status", parents=[common], help="one table, one row per node")
    actions.add_parser("datasets", parents=[common], help="dataset placement and drift")
    actions.add_parser("nodes", parents=[common], help="the configured node list")


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    console = Console(quiet=args.quiet or args.json, verbose=args.verbose)

    try:
        config = load(args.config)
    except ConfigError as exc:
        console.write(f"x {exc}")
        return USAGE

    for problem in config.validate():
        console.write(f"! {problem}")

    # These need no node at all.
    if args.command == "config":
        return _run_config(args, config, console)
    if args.command == "serve":
        return _run_serve(args, config, console)
    if args.command == "fleet":
        return _run_fleet(args, config, console)

    try:
        client = _client(args, config)
    except (ApiError, ValueError, RuntimeError) as exc:
        console.write(f"x {exc}")
        return USAGE

    handlers = {
        "node": _run_node,
        "prompts": _run_prompts,
        "dataset": _run_dataset,
        "jobs": _run_jobs,
        "train": _run_train,
    }
    try:
        return handlers[args.command](args, config, console, client)
    except KeyboardInterrupt:
        console.write("interrupted")
        return INTERRUPTED
    except ApiError as exc:
        console.write(f"x {exc}")
        return PROBLEM if exc.status in (409, 410, 422) else USAGE
    except (NotADirectoryError, FileNotFoundError, ValueError) as exc:
        console.write(f"x {exc}")
        return USAGE
    finally:
        client.close()


def _client(args: argparse.Namespace, config: Config) -> Client:
    url = getattr(args, "url", None)
    node = getattr(args, "node", None)
    if node:
        url = Fleet.load().url_for(node)
    return Client.for_config(config, url)


# --------------------------------------------------------------------------
# config, serve, node
# --------------------------------------------------------------------------


def _run_config(args: argparse.Namespace, config: Config, console: Console) -> int:
    if args.action == "show":
        payload = config.as_dict()
        payload["source"] = str(config.source) if config.source else None
        if args.json:
            emit_json(payload)
        else:
            console.write(f"# from {config.source or 'defaults (no file yet)'}")
            console.write(table(_flatten(config.as_dict())))
        return OK

    if args.action == "path":
        located = paths.describe()
        located["fleet_file"] = str(paths.fleet_file())
        if args.json:
            emit_json(located)
        else:
            console.write(table(sorted(located.items()), headers=("location", "path")))
        return OK

    if args.action in ("set", "secrets"):
        # `config` is dispatched before main()'s client error handling,
        # because most of its actions never touch a daemon. These two do,
        # so they carry their own - a refused setting is a message, not a
        # traceback.
        try:
            client = _client(args, config)
        except (ApiError, ValueError, RuntimeError) as exc:
            console.write(f"x {exc}")
            return USAGE
        try:
            return _run_config_remote(args, console, client)
        except ApiError as exc:
            console.write(f"x {exc}")
            return PROBLEM if exc.status in (403, 409, 422) else USAGE
        finally:
            client.close()

    from ..core.config import example_toml

    target = paths.config_file()
    if target.exists() and not args.force:
        console.write(f"x {target} already exists; pass --force to overwrite")
        return USAGE
    paths.ensure_dir(target.parent)
    target.write_text(example_toml(), encoding="utf-8")
    console.write(f"wrote {target}")
    return OK


def _run_config_remote(args: argparse.Namespace, console: Console, client: Client) -> int:
    """The two config actions that belong to a node rather than to this machine."""
    if args.action == "secrets":
        payload = client.get("/config/secrets")
        if args.json:
            emit_json(payload)
            return OK
        console.write(
            table(
                [
                    (s["name"], "found" if s["found"] else "-", ", ".join(s["env"]))
                    for s in payload["secrets"]
                ],
                headers=("secret", "state", "environment variables tried"),
            )
        )
        return OK

    updates: dict[str, Any] = {}
    for pair in args.pairs:
        key, sep, value = pair.partition("=")
        if not sep:
            console.write(f"x {pair!r} is not KEY=VALUE")
            return USAGE
        updates[key.strip()] = _literal(value)

    payload = client.put("/config", json_body={"set": updates})
    if args.json:
        emit_json(payload)
        return OK

    for key in payload.get("changed", []):
        console.write(f"{key} = {updates[key]}")
    console.write(f"wrote {payload.get('written')}")
    for key in payload.get("restart_required", []):
        console.write(f"! {key} is only read at startup - restart the daemon")
    return OK


def _literal(value: str) -> Any:
    """Parse a command-line value into the type the config expects.

    ``true``/``false`` and numbers are converted; everything else stays a
    string, which the config coerces to the declared type anyway.
    """
    text = value.strip()
    lowered = text.lower()
    if lowered in ("true", "yes", "on"):
        return True
    if lowered in ("false", "no", "off"):
        return False
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            continue
    return text


def _run_serve(args: argparse.Namespace, config: Config, console: Console) -> int:
    from ..daemon.app import serve
    from ..daemon.security import Denied

    if args.host:
        config.daemon.host = args.host
    if args.port:
        config.daemon.port = args.port

    console.write(f"serving on http://{config.daemon.host}:{config.daemon.port}")
    if config.daemon.host in ("127.0.0.1", "localhost", "::1"):
        console.write("reach it from a laptop with:")
        console.write(
            f"  ssh -N -L {config.daemon.port}:localhost:{config.daemon.port} user@this-node"
        )
    try:
        serve(config)
    except Denied as exc:
        console.write(f"x {exc}")
        return USAGE
    except KeyboardInterrupt:
        console.write("stopped")
    return OK


def _run_prompts(
    args: argparse.Namespace, config: Config, console: Console, client: Client
) -> int:
    """Saved caption prompts, on whichever node the client points at."""
    if args.action == "list":
        payload = client.get("/captioners/prompts")
        if args.json:
            emit_json(payload)
            return OK
        rows = [
            (
                prompt["name"],
                "built-in" if prompt["builtin"] else "saved",
                prompt["text"] if args.full else _preview(prompt["text"]),
            )
            for prompt in payload["prompts"]
        ]
        console.write(table(rows, headers=("name", "source", "prompt")))
        console.write("")
        console.write(str(payload["file"]))
        return OK

    if args.action == "show":
        found = next(
            (p for p in client.get("/captioners/prompts")["prompts"] if p["name"] == args.name),
            None,
        )
        if found is None:
            console.write(f"x no prompt named {args.name!r}")
            return USAGE
        if args.json:
            emit_json(found)
        else:
            console.write(found["text"])
        return OK

    if args.action == "save":
        # Reading stdin lets a long prompt come from a file or a heredoc
        # rather than being fought with through shell quoting.
        text = args.text if args.text is not None else sys.stdin.read()
        saved = client.put(f"/captioners/prompts/{args.name}", json_body={"text": text})
        if args.json:
            emit_json(saved)
            return OK
        console.write(f"saved {saved['name']!r}")
        if saved.get("shadows_builtin"):
            console.write("  (standing over the built-in of the same name; delete to restore it)")
        return OK

    payload = client.delete(f"/captioners/prompts/{args.name}")
    if args.json:
        emit_json(payload)
        return OK
    console.write(f"deleted {args.name!r}")
    if payload.get("restored"):
        console.write("  (the built-in of the same name is back)")
    return OK


def _preview(text: str, width: int = 64) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= width else flat[: width - 3] + "..."


def _run_node(args: argparse.Namespace, config: Config, console: Console, client: Client) -> int:
    if args.action == "gpus":
        payload = client.get("/gpus")
        if args.json:
            emit_json(payload)
            return OK
        gpus = payload.get("gpus") or []
        if not gpus:
            console.write("no CUDA devices visible from this node")
            return OK
        console.write(
            table(
                [
                    (
                        str(g["index"]),
                        g["name"],
                        g.get("capability", "-"),
                        _gb(g.get("vram_free")),
                        _gb(g.get("vram_total")),
                    )
                    for g in gpus
                ],
                headers=("#", "name", "cc", "free", "total"),
            )
        )
        return OK

    if args.action == "status":
        health = client.get("/health")
        if health.get("stale"):
            console.write(
                "! this daemon started before the code it is running was last "
                "changed.\n  Restart it, or it will keep behaving like the "
                "old version: fk serve"
            )

    if args.action == "captioners":
        payload = client.get("/captioners")
        if args.test or args.provider:
            body = {"provider": args.provider} if args.provider else {}
            probe = client.post("/captioners/test", json_body=body)
        else:
            probe = {}

        if args.json:
            emit_json({**payload, "probe": probe} if probe else payload)
            return OK

        console.write(
            table(
                [
                    (
                        c["name"],
                        c["label"],
                        "yes" if c["available"] else "no",
                        "*" if c["name"] == payload.get("configured") else "",
                    )
                    for c in payload["captioners"]
                ],
                headers=("name", "backend", "installed", "in use"),
            )
        )
        if probe:
            console.write("")
            console.write(("ok  " if probe.get("ok") else "x   ") + str(probe.get("message", "")))
            for model in probe.get("models", []):
                console.write(f"    {model}")
            return OK if probe.get("ok") else PROBLEM
        return OK

    if args.action == "models":
        payload = client.get("/models")
        if args.json:
            emit_json(payload)
            return OK
        console.write(
            table(
                [
                    (m["id"], m["arch"] or "-", m["label"], str(m["network_dim"]), m["notes"])
                    for m in payload["models"]
                ],
                headers=("model", "arch", "name", "dim", "notes"),
            )
        )
        console.write()
        for name, backend in payload["backends"].items():
            state = "ready" if backend["ready"] else "no checkout configured"
            console.write(f"  backend {name}: {state}")
        return OK

    info = client.get("/node")
    if args.json:
        emit_json(info)
        return OK

    console.write(f"# {client.where}")
    rows = [
        ("node", info["name"]),
        ("version", info["version"]),
        ("os", f"{info['os']} {info['os_release']} ({info['machine']})"),
        ("python", info["python"]),
        ("opencv", info["opencv"]),
        ("torch", info.get("torch") or "not installed"),
        ("cuda", info.get("cuda") or "-"),
        ("driver", info.get("driver") or "-"),
    ]
    free = info.get("disk_free") or {}
    if free:
        rows.append(("disk free", _gb(free.get("free"))))
    console.write(table(rows, headers=("field", "value")))

    console.write()
    for name, ready in (info.get("detectors") or {}).items():
        console.write(f"  detector {name}: {'ready' if ready else 'unavailable'}")
    if not (info.get("detectors") or {}).get("yunet"):
        console.write("! YuNet weights are missing; see assets/models/README.md")

    for name, backend in (info.get("backends") or {}).items():
        state = "ready" if backend.get("ready") else "no checkout configured"
        console.write(f"  backend {name}: {state} ({len(backend.get('models', []))} models)")

    for gpu in info.get("gpus") or []:
        console.write(f"  gpu {gpu['index']}: {gpu['name']} ({_gb(gpu.get('vram_total'))})")
    return OK


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------


def _run_dataset(args: argparse.Namespace, config: Config, console: Console, client: Client) -> int:
    if args.action == "list":
        payload = client.get("/datasets")
        if args.json:
            emit_json(payload)
        else:
            rows = [
                (d["id"], d["path"], "yes" if d["exists"] else "MISSING")
                for d in payload.get("datasets", [])
            ]
            console.write(
                table(rows, headers=("id", "path", "there")) if rows else "no datasets registered"
            )
        return OK

    if args.action == "where":
        return _run_where(args, console)

    if args.action == "push":
        return _run_push(args, console, client)

    dataset = client.resolve(args.path)
    dataset_id = dataset["id"]

    if args.action == "register":
        if args.json:
            emit_json(dataset)
        else:
            console.write(f"{dataset_id} -> {dataset['path']}")
        return OK

    if args.action == "scan":
        payload = client.get(f"/datasets/{dataset_id}/items")
        if args.json:
            emit_json(payload)
            return OK
        rows = [
            (
                item["filename"],
                "yes" if item["has_caption"] else "-",
                "yes" if item["has_mask"] else "-",
                str(item["boxes"]) if item["boxes"] else "-",
                "yes" if item["reviewed"] else "-",
                item["quality"] or "-",
            )
            for item in payload["items"]
        ]
        console.write(table(rows, headers=("image", "caption", "mask", "boxes", "seen", "quality")))
        console.write(f"\n{len(rows)} items - {payload['review']['summary']}")
        return OK

    if args.action == "validate":
        report = client.get(
            f"/datasets/{dataset_id}/validate", params={"require_masks": args.require_masks}
        )
        if args.json:
            emit_json(report)
        else:
            for problem in report["problems"]:
                where = f"{problem['stem']}: " if problem["stem"] else ""
                console.write(f"  [{problem['severity']}] {where}{problem['message']}")
            console.write(
                f"\n{report['items']} items, {len(report['problems'])} problems"
                if report["problems"]
                else f"\n{report['items']} items, no problems"
            )
        return OK if report["ok"] else PROBLEM

    if args.action == "review":
        return _run_review(args, console, client, dataset_id)

    if args.action == "boxes":
        return _run_boxes(args, console, client, dataset_id)

    if args.action == "manifest":
        payload = client.get(
            f"/datasets/{dataset_id}/manifest",
            params={"digests": not args.quick, "sidecars_only": args.sidecars_only},
            timeout=600.0,
        )
        if args.json:
            emit_json(payload)
        else:
            console.write(
                table(
                    [
                        (e["path"], str(e["size"]), e.get("digest", "-")[:12])
                        for e in payload["entries"]
                    ],
                    headers=("path", "bytes", "digest"),
                )
            )
            console.write(f"\n{payload['files']} files, {payload['bytes']} bytes")
        return OK

    if args.action == "export":
        client.download(
            f"/datasets/{dataset_id}/export",
            args.out,
            params={"sidecars_only": args.sidecars_only},
        )
        console.write(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
        return OK

    return _run_operation(args, console, client, dataset_id)


def _run_operation(
    args: argparse.Namespace,
    console: Console,
    client: Client,
    dataset_id: str,
) -> int:
    """Submit a dataset operation and follow its event stream to the end."""
    task = client.post(f"/datasets/{dataset_id}/ops/{args.action}", json_body=_options(args))
    task_id = task["id"]

    for payload in client.stream(f"/tasks/{task_id}/events"):
        console.event(payload)

    final = client.get(f"/tasks/{task_id}")
    result = final.get("result") or {}

    if args.json:
        emit_json(final)
    elif args.action == "mask" and final["status"] == "done":
        # The one line that has to end up in a training config, spelled out.
        masks = f"{_dataset_path(client, dataset_id)}/masks"
        console.write(f"\nmasks are in {masks}")
        console.write("point the trainer at them with:")
        console.write(f"  mask_path: {masks}")

    ok = final["status"] == "done" and result.get("ok", True)
    return OK if ok else PROBLEM


def _options(args: argparse.Namespace) -> dict[str, Any]:
    """Turn parsed flags into the operation payload the API expects."""
    if args.action == "resize":
        return {"size": args.size, "output": args.output, "upscale": not args.no_upscale}
    if args.action == "caption":
        return {
            "provider": args.provider,
            "model": args.model,
            "url": args.ollama_url,
            "prompt": args.prompt,
            "prompt_name": args.prompt_name,
            "prefix": args.prefix,
            "overwrite": args.overwrite,
            "max_tokens": args.max_tokens,
            "timeout": args.timeout,
        }
    if args.action == "rename":
        return {
            "prefix": args.prefix,
            "start": args.start,
            "digits": args.digits,
            "scramble": args.scramble,
            "seed": args.seed,
            "dry_run": args.dry_run,
        }
    if args.action == "augment":
        return {
            "transforms": [
                name
                for name, wanted in (
                    ("flip_horizontal", args.flip),
                    ("rotate_90_left", args.rot_left),
                    ("rotate_90_right", args.rot_right),
                    ("rotate_180", args.rot_180),
                    ("duplicate", args.duplicate),
                )
                if wanted
            ],
            "output": args.output,
        }
    if args.action == "detect":
        return {
            "detector": args.detector,
            "confidence": args.confidence,
            "workers": args.workers or 4,
            "only_missing": args.only_missing,
        }
    if args.action == "mask":
        payload: dict[str, Any] = {
            "detect": not args.no_detect,
            "force": args.force,
            "previews": not args.no_previews,
        }
        for key, value in (
            ("detector", args.detector),
            ("confidence", args.confidence),
            ("expand", args.expand),
            ("expand_up", args.expand_up),
            ("feather", args.feather),
        ):
            if value is not None:
                payload[key] = value
        return payload
    return {}


def _run_review(args: argparse.Namespace, console: Console, client: Client, dataset_id: str) -> int:
    if args.mark_all_reviewed or args.mark:
        items = client.get(f"/datasets/{dataset_id}/items")["items"]
        wanted = (
            [i["stem"] for i in items]
            if args.mark_all_reviewed
            else [Path(name).stem for name in args.mark]
        )
        for stem in wanted:
            current = client.get(f"/datasets/{dataset_id}/items/{stem}/boxes")
            client.put(
                f"/datasets/{dataset_id}/items/{stem}/boxes",
                json_body={"boxes": current["boxes"], "reviewed": True},
            )
        console.write(f"marked {len(wanted)} images reviewed")

    progress = client.get(f"/datasets/{dataset_id}/review")
    if args.json:
        emit_json(progress)
        return OK if progress["complete"] else PROBLEM

    console.write(progress["summary"])
    if progress["empty"]:
        console.write("\nno detections - look at these first:")
        for name in progress["empty"][:40]:
            console.write(f"  {name}")
        if len(progress["empty"]) > 40:
            console.write(f"  ... and {len(progress['empty']) - 40} more")
    return OK if progress["complete"] else PROBLEM


def _run_boxes(args: argparse.Namespace, console: Console, client: Client, dataset_id: str) -> int:
    stem = Path(args.image).stem
    path = f"/datasets/{dataset_id}/items/{stem}/boxes"
    current = client.get(path)

    if args.clear or args.add or args.reviewed:
        boxes = [] if args.clear else list(current["boxes"])
        boxes.extend(_parse_box(raw) for raw in args.add)
        current = client.put(
            path,
            json_body={"boxes": boxes, "reviewed": args.reviewed or current.get("reviewed", False)},
        )

    if args.json:
        emit_json(current)
        return OK

    console.write(f"{stem}: {len(current['boxes'])} boxes, reviewed={current['reviewed']}")
    console.write(
        table(
            [
                (
                    str(b["x"]),
                    str(b["y"]),
                    str(b["w"]),
                    str(b["h"]),
                    b.get("src", "-"),
                    "-" if b.get("conf") is None else f"{b['conf']:.2f}",
                )
                for b in current["boxes"]
            ],
            headers=("x", "y", "w", "h", "src", "conf"),
        )
    )
    return OK


def _run_push(args: argparse.Namespace, console: Console, client: Client) -> int:
    from .push import push

    target = args.to or getattr(args, "node", None)
    if target:
        remote = Client.remote(Fleet.load().url_for(target))
    elif getattr(args, "url", None):
        remote = client
    else:
        console.write("x push needs a target: --to NODE, or --url")
        return USAGE

    try:
        result = push(
            remote,
            args.path,
            sidecars_only=args.sidecars_only,
            dry_run=args.dry_run,
            transport=args.transport,
            ssh_target=args.ssh,
            remote_path=args.remote_path,
            digests=not args.quick,
            emit=console,
        )
    finally:
        if remote is not client:
            remote.close()

    if args.json:
        emit_json(result.as_dict())
    else:
        console.write(result.summary())
    return OK if result.ok else PROBLEM


def _run_where(args: argparse.Namespace, console: Console) -> int:
    """Which nodes have a dataset, and whether their copies agree."""
    rows = Fleet.load().where(args.dataset)
    if args.json:
        emit_json(rows)
        return OK
    if not rows:
        console.write(f"no node in fleet.toml has {args.dataset!r}")
        return PROBLEM
    console.write(
        table(
            [
                (r["node"], r.get("path", "-"), str(r.get("files", "-")), r.get("state", ""))
                for r in rows
            ],
            headers=("node", "path", "files", "state"),
        )
    )
    return OK


# --------------------------------------------------------------------------
# jobs and fleet
# --------------------------------------------------------------------------


def _run_jobs(args: argparse.Namespace, config: Config, console: Console, client: Client) -> int:
    if args.action == "list":
        payload = client.get("/jobs")
        if args.json:
            emit_json(payload)
            return OK
        rows = [
            (
                job["id"],
                job["status"],
                str(job["device"]),
                job["spec"]["model"],
                job["spec"]["dataset"],
                f"{job['progress']['step']}/{job['progress']['total']}",
            )
            for job in payload["jobs"]
        ]
        console.write(
            table(rows, headers=("id", "status", "gpu", "model", "dataset", "progress"))
            if rows
            else "no jobs"
        )
        if not payload["runner"]:
            console.write("\n! no training backend is registered on this node yet")
        return OK

    if args.action == "watch":
        for payload in client.stream(f"/jobs/{args.job}/events"):
            console.event(payload)
        final = client.get(f"/jobs/{args.job}")
        if args.json:
            emit_json(final)
        return OK if final["status"] == "done" else PROBLEM

    if args.action == "cancel":
        payload = client.delete(f"/jobs/{args.job}")
        state = "dequeued" if payload["dequeued"] else "cancelling"
        console.write(f"{args.job}: {state} (was {payload['was']})")
        return OK

    payload = client.get(f"/jobs/{args.job}/loss")
    if args.json:
        emit_json(payload)
    else:
        points = payload["points"]
        console.write(
            table([(str(p["step"]), f"{p['value']:.5f}") for p in points], headers=("step", "loss"))
            if points
            else "no loss recorded"
        )
    return OK


def _run_train(args: argparse.Namespace, config: Config, console: Console, client: Client) -> int:
    """Submit a run. Placement is explicit - this node, or --node/--url."""
    spec = _run_spec(args, client)
    job = client.post("/jobs", json_body=spec)

    if warning := job.get("warning"):
        console.write(f"! {warning}")

    if args.json and not args.watch:
        emit_json(job)
    else:
        console.write(f"queued {job['id']} on gpu {job['device']} at {client.where}")
        console.write(f"follow it with:  fk jobs watch {job['id']}")

    if not args.watch:
        return OK

    for payload in client.stream(f"/jobs/{job['id']}/events"):
        console.event(payload)
    final = client.get(f"/jobs/{job['id']}")
    if args.json:
        emit_json(final)
    return OK if final["status"] == "done" else PROBLEM


def _run_spec(args: argparse.Namespace, client: Client) -> dict[str, Any]:
    """Build the POST body from a spec file, flags, or both."""
    import tomllib

    spec: dict[str, Any] = {}
    if args.spec:
        try:
            spec.update(tomllib.loads(args.spec.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot read {args.spec}: {exc}") from exc

    for key, value in (
        ("model", args.model),
        ("dataset", args.dataset),
        ("name", args.name),
        ("device", args.device),
        ("steps", args.steps),
        ("learning_rate", args.lr),
        ("network_dim", args.dim),
        ("network_alpha", args.alpha),
        ("resolution", args.resolution),
        ("batch_size", args.batch_size),
        ("save_every", args.save_every),
        ("sample_every", args.sample_every),
        ("mask_min_value", args.mask_min),
    ):
        if value is not None:
            spec[key] = value

    if args.prompt:
        extra = dict(spec.get("extra") or {})
        extra["sample_prompts"] = list(args.prompt)
        spec["extra"] = extra

    if not spec.get("model") or not spec.get("dataset"):
        raise ValueError("a run needs --model and --dataset, or a --spec file with both")

    if args.masked and not spec.get("mask_path"):
        # Resolve the dataset on the target node, so the path in the config is
        # the node's path rather than one that only exists on this laptop.
        dataset = client.resolve(str(spec["dataset"]))
        spec["mask_path"] = f"{dataset['path'].rstrip('/')}/{paths.MASKS_DIRNAME}"
        spec["dataset"] = dataset["id"]

    return spec


def _run_fleet(args: argparse.Namespace, config: Config, console: Console) -> int:
    fleet = Fleet.load()
    if not fleet.nodes:
        console.write(f"x no nodes configured. Write {paths.fleet_file()}:")
        console.write(fleet.example())
        return USAGE

    if args.action == "nodes":
        if args.json:
            emit_json([n.as_dict() for n in fleet.nodes])
        else:
            console.write(table([(n.name, n.url) for n in fleet.nodes], headers=("name", "url")))
        return OK

    if args.action == "status":
        rows = fleet.status()
        if args.json:
            emit_json(rows)
            return OK
        console.write(
            table(
                [
                    (
                        r["node"],
                        r["state"],
                        r.get("version") or "-",
                        str(r.get("queue_depth", "-")),
                        r.get("gpu") or "-",
                        r.get("torch") or "-",
                    )
                    for r in rows
                ],
                headers=("node", "state", "version", "queued", "gpu", "torch"),
            )
        )
        return OK if all(r["state"] == "up" for r in rows) else PROBLEM

    placement = fleet.datasets()
    if args.json:
        emit_json(placement)
        return OK
    rows = []
    for dataset_id, entries in sorted(placement.items()):
        for entry in entries:
            rows.append(
                (dataset_id, entry["node"], str(entry.get("files", "-")), entry.get("state", ""))
            )
    console.write(table(rows, headers=("dataset", "node", "files", "state")) if rows else "no datasets")
    return OK


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def _parse_box(raw: str) -> dict[str, Any]:
    parts = [p.strip() for p in raw.replace(" ", ",").split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError(f"a box is X,Y,W,H - got {raw!r}")
    try:
        x, y, w, h = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError(f"a box is four integers - got {raw!r}") from exc
    if w <= 0 or h <= 0:
        raise ValueError(f"a box needs a positive width and height - got {raw!r}")
    return {"x": x, "y": y, "w": w, "h": h, "src": "manual"}


def _dataset_path(client: Client, dataset_id: str) -> str:
    for dataset in client.get("/datasets").get("datasets", []):
        if dataset["id"] == dataset_id:
            return str(dataset["path"])
    return dataset_id


def _flatten(data: dict[str, Any], prefix: str = "") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for key, value in data.items():
        dotted = f"{prefix}{key}"
        if isinstance(value, dict):
            rows.extend(_flatten(value, f"{dotted}."))
        else:
            rows.append((dotted, str(value)))
    return rows


def _gb(value: Any) -> str:
    if not value:
        return "-"
    return f"{int(value) / (1024 ** 3):.1f}GB"


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
