"""``fk train`` and the models endpoints, over HTTP.

Uses the same stand-in trainer as ``test_training_run``: a node with a
folder that looks like an ai-toolkit checkout and a script that prints
training-shaped output.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from fluxkrea.cli.__main__ import OK, PROBLEM, main
from tests.backends.test_training_run import FAKE_TRAINER, toolkit_with


@pytest.fixture
def node(tmp_path: Path, dataset: Path) -> Iterator[str]:
    """A node that can train: a checkout, an interpreter, an output root."""
    from fluxkrea.cli.embedded import EmbeddedDaemon
    from fluxkrea.core.config import load
    from fluxkrea.daemon.registry import Registry
    from fluxkrea.daemon.state import State

    config = load(use_file=False)
    config.dataset.min_resolution = 0
    config.backends.aitoolkit_path = toolkit_with(FAKE_TRAINER, tmp_path)
    config.backends.python_exe = sys.executable
    config.backends.output_root = tmp_path / "runs"

    state = State(config=config, registry=Registry(file=tmp_path / "registry.json"))
    daemon = EmbeddedDaemon(config, state)
    url = daemon.start()
    yield url
    daemon.stop()
    state.shutdown()


@pytest.fixture(autouse=True)
def use_node(node: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FLUXKREA_URL", node)


def run(*args: str) -> int:
    return main([str(a) for a in args])


def payload(capsys: pytest.CaptureFixture[str]):  # noqa: ANN201
    return json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------


def test_node_models_lists_the_flux2_family(capsys: pytest.CaptureFixture[str]) -> None:
    assert run("--json", "node", "models") == OK
    listed = payload(capsys)

    ids = {m["id"] for m in listed["models"]}
    assert {"flux2", "flux2-klein-4b", "flux2-klein-9b", "krea2"} <= ids
    assert listed["backends"]["aitoolkit"]["ready"] is True


def test_a_node_without_a_checkout_says_so(tmp_path: Path, monkeypatch, capsys) -> None:
    from fluxkrea.cli.embedded import EmbeddedDaemon
    from fluxkrea.core.config import load
    from fluxkrea.daemon.registry import Registry
    from fluxkrea.daemon.state import State

    config = load(use_file=False)
    state = State(config=config, registry=Registry(file=tmp_path / "bare.json"))
    daemon = EmbeddedDaemon(config, state)
    monkeypatch.setenv("FLUXKREA_URL", daemon.start())
    try:
        assert run("--json", "node", "models") == OK
        assert payload(capsys)["backends"]["aitoolkit"]["ready"] is False
    finally:
        daemon.stop()
        state.shutdown()


# --------------------------------------------------------------------------
# submitting
# --------------------------------------------------------------------------


def test_a_run_needs_a_model_and_a_dataset(dataset: Path) -> None:
    from fluxkrea.cli.__main__ import USAGE

    assert run("train", "--model", "flux2") == USAGE
    assert run("train", "--dataset", str(dataset)) == USAGE


def test_an_unknown_model_is_refused_at_submission(dataset: Path) -> None:
    """Not queued and then failed - refused, with the list of what is known."""
    assert run("train", "--model", "sdxl", "--dataset", str(dataset)) == PROBLEM


def test_a_run_trains_and_reports(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert run(
        "train", "--model", "flux2", "--dataset", str(dataset),
        "--name", "probe", "--steps", "6", "--watch", "--json",
    ) == OK

    final = payload(capsys)
    assert final["status"] == "done"
    assert final["progress"] == {"step": 6, "total": 6}
    assert final["spec"]["model"] == "flux2"


def test_the_flags_reach_the_generated_config(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    run(
        "train", "--model", "flux2-klein-4b", "--dataset", str(dataset),
        "--name", "flagged", "--steps", "2", "--lr", "0.0003",
        "--dim", "64", "--alpha", "16", "--resolution", "768",
        "--sample-every", "100", "--prompt", "a test prompt",
        "--watch", "--json",
    )
    final = payload(capsys)
    rendered = yaml.safe_load(Path(final["config_path"]).read_text(encoding="utf-8"))
    process = rendered["config"]["process"][0]

    assert process["model"]["arch"] == "flux2_klein_4b"
    assert process["train"]["lr"] == 0.0003
    assert process["network"] == {"type": "lora", "linear": 64, "linear_alpha": 16}
    assert process["datasets"][0]["resolution"] == 768
    assert process["sample"]["samples"][0]["prompt"] == "a test prompt"
    assert process["sample"]["guidance_scale"] == 3.5, "Klein is not guidance-distilled"


def test_a_spec_file_and_flags_compose(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The spec carries the run; flags override for a one-off."""
    import yaml

    spec = tmp_path / "run.toml"
    spec.write_text(
        'model = "flux2"\nname = "from-spec"\nsteps = 100\nlearning_rate = 0.0002\n'
        'network_dim = 48\n',
        encoding="utf-8",
    )

    run("train", "--spec", str(spec), "--dataset", str(dataset), "--steps", "2",
        "--watch", "--json")
    final = payload(capsys)

    rendered = yaml.safe_load(Path(final["config_path"]).read_text(encoding="utf-8"))
    train = rendered["config"]["process"][0]["train"]
    assert train["steps"] == 2, "the flag should win over the spec"
    assert train["lr"] == 0.0002, "the spec should supply what the flag did not"
    assert rendered["config"]["process"][0]["network"]["linear"] == 48


def test_a_masked_run_points_at_the_nodes_own_masks(
    dataset: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import yaml

    from fluxkrea.core import paths
    from tests.conftest import make_mask

    for index in range(1, 5):
        make_mask(paths.masks_dir(dataset) / f"punch_{index:03d}.png", size=(64 + index, 48 + index))

    run("train", "--model", "flux2", "--dataset", str(dataset), "--name", "masked",
        "--steps", "2", "--masked", "--watch", "--json")
    final = payload(capsys)

    assert final["status"] == "done"
    rendered = yaml.safe_load(Path(final["config_path"]).read_text(encoding="utf-8"))
    assert rendered["config"]["process"][0]["datasets"][0]["mask_path"] == (
        paths.masks_dir(dataset).as_posix()
    )


def test_a_masked_run_refuses_a_dataset_that_is_missing_masks(dataset: Path) -> None:
    """Only one of the four fixtures has a mask."""
    assert run("train", "--model", "flux2", "--dataset", str(dataset), "--name", "bad",
               "--steps", "2", "--masked", "--watch") == PROBLEM


def test_jobs_list_shows_the_run(dataset: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run("train", "--model", "krea2", "--dataset", str(dataset), "--name", "listed",
        "--steps", "2", "--watch")
    capsys.readouterr()

    assert run("--json", "jobs", "list") == OK
    listing = payload(capsys)
    assert listing["runner"] is True
    assert listing["jobs"][0]["spec"]["model"] == "krea2"


def test_the_loss_series_survives_the_run(dataset: Path, capsys) -> None:
    run("train", "--model", "flux2", "--dataset", str(dataset), "--name", "lossy",
        "--steps", "6", "--watch", "--json")
    job_id = payload(capsys)["id"]

    assert run("--json", "jobs", "loss", job_id) == OK
    points = payload(capsys)["points"]
    assert len(points) == 6
    assert points[0]["value"] > points[-1]["value"]
