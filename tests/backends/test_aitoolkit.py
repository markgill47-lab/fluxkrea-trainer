"""Config generation and output parsing for the ai-toolkit backend.

The config tests are worth more than they look. Every quirk pinned here
was learned by somebody watching a run do nothing for an hour, and none of
it is visible from the generated YAML unless you already know.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fluxkrea.core.backends import BackendError, supported_by
from fluxkrea.core.backends.aitoolkit import AIToolkitBackend, OutputParser
from fluxkrea.core.backends.models import UnknownModel
from fluxkrea.core.backends.spec import RunSpec
from fluxkrea.core.events import Collector, LossPoint, Progress


@pytest.fixture
def backend(tmp_path: Path) -> AIToolkitBackend:
    return AIToolkitBackend(tmp_path / "ai-toolkit", output_root=tmp_path / "runs")


def spec_for(model: str = "flux2", **kwargs) -> RunSpec:
    payload = {
        "model": model,
        "dataset": "/srv/datasets/poses",
        "name": "blizzard",
        "steps": 2400,
        "learning_rate": 0.0002,
        **kwargs,
    }
    return RunSpec(**payload)


def process_of(backend: AIToolkitBackend, spec: RunSpec) -> dict:
    return backend.build(spec)["config"]["process"][0]


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_the_backend_is_registered_for_the_flux2_family() -> None:
    for model in ("flux2", "flux2-klein-4b", "flux2-klein-9b", "krea2"):
        backend = supported_by(model)
        assert backend is not None and backend.name == "aitoolkit"


def test_one_class_covers_all_of_them(backend: AIToolkitBackend) -> None:
    """v1 has two classes differing mostly in which strings they emit."""
    archs = {process_of(backend, spec_for(m))["model"].get("arch") for m in
             ("flux2", "flux2-klein-4b", "krea2")}
    assert archs == {"flux2", "flux2_klein_4b", "krea2"}


def test_an_unknown_model_raises(backend: AIToolkitBackend) -> None:
    with pytest.raises(UnknownModel):
        backend.build(spec_for("stable-diffusion-xl"))


# --------------------------------------------------------------------------
# the quirks
# --------------------------------------------------------------------------


def test_lr_is_written_as_well_as_learning_rate(backend: AIToolkitBackend) -> None:
    """`lr` is the key ai-toolkit reads; `learning_rate` alone trains at ~1e-6."""
    train = process_of(backend, spec_for(learning_rate=0.0003))["train"]
    assert train["lr"] == 0.0003
    assert train["learning_rate"] == 0.0003


def test_caching_is_a_dataset_setting_not_a_train_setting(backend: AIToolkitBackend) -> None:
    """Under `train` it silently does nothing, and the VRAM never comes back."""
    process = process_of(backend, spec_for())

    dataset = process["datasets"][0]
    assert dataset["cache_latents_to_disk"] is True
    assert dataset["cache_text_embeddings"] is True
    assert "cache_latents" not in process["train"]
    assert "cache_text_encoder_outputs" not in process["train"]


def test_skip_first_sample_is_a_train_key(backend: AIToolkitBackend) -> None:
    """`sample_at_first` in the sample block is ignored by current upstream."""
    process = process_of(backend, spec_for(sample_every=100, extra={"sample_prompts": ["x"]}))
    assert process["train"]["skip_first_sample"] is True
    assert "sample_at_first" not in process.get("sample", {})


def test_layer_offloading_is_off(backend: AIToolkitBackend) -> None:
    """v1 found it too slow even at 30%. Off unless explicitly asked for."""
    assert process_of(backend, spec_for())["model"]["layer_offloading"] is False
    assert process_of(backend, spec_for(extra={"layer_offloading": True}))["model"][
        "layer_offloading"
    ] is True


def test_krea2_refuses_a_vae_path(backend: AIToolkitBackend) -> None:
    """The arch resolves its own VAE; setting one fights it."""
    with pytest.raises(BackendError, match="resolves its own VAE"):
        backend.build(spec_for("krea2", extra={"vae_path": "/models/ae.safetensors"}))


# --------------------------------------------------------------------------
# the mask line
# --------------------------------------------------------------------------


def test_mask_path_reaches_the_dataset_block(backend: AIToolkitBackend) -> None:
    """The one line the whole masking feature exists to produce."""
    dataset = process_of(backend, spec_for(mask_path="/srv/datasets/poses/masks"))["datasets"][0]

    assert dataset["mask_path"] == "/srv/datasets/poses/masks"
    assert dataset["mask_min_value"] == 0.0


def test_an_unmasked_run_writes_no_mask_keys(backend: AIToolkitBackend) -> None:
    dataset = process_of(backend, spec_for())["datasets"][0]
    assert "mask_path" not in dataset
    assert "mask_min_value" not in dataset


def test_mask_min_value_is_carried(backend: AIToolkitBackend) -> None:
    """0.05-0.1 lets the region contribute a trace, if a hard zero artefacts."""
    dataset = process_of(
        backend, spec_for(mask_path="/srv/poses/masks", mask_min_value=0.05)
    )["datasets"][0]
    assert dataset["mask_min_value"] == 0.05


# --------------------------------------------------------------------------
# per-model differences
# --------------------------------------------------------------------------


def test_klein_9b_gets_low_vram_and_4b_does_not(backend: AIToolkitBackend) -> None:
    assert process_of(backend, spec_for("flux2-klein-9b"))["model"]["low_vram"] is True
    assert process_of(backend, spec_for("flux2-klein-4b"))["model"]["low_vram"] is False


def test_network_dim_defaults_per_model_and_can_be_overridden(backend: AIToolkitBackend) -> None:
    assert process_of(backend, spec_for("flux2"))["network"]["linear"] == 32
    assert process_of(backend, spec_for("flux1"))["network"]["linear"] == 16
    assert process_of(backend, spec_for("flux2", network_dim=64))["network"]["linear"] == 64


def test_klein_samples_at_its_own_guidance(backend: AIToolkitBackend) -> None:
    klein = process_of(
        backend, spec_for("flux2-klein-4b", sample_every=50, extra={"sample_prompts": ["x"]})
    )
    flux2 = process_of(
        backend, spec_for("flux2", sample_every=50, extra={"sample_prompts": ["x"]})
    )
    assert klein["sample"]["guidance_scale"] == 3.5
    assert flux2["sample"]["guidance_scale"] == 4.0


def test_the_device_pins_the_gpu(backend: AIToolkitBackend) -> None:
    assert process_of(backend, spec_for(device=1))["device"] == "cuda:1"


# --------------------------------------------------------------------------
# the artifact
# --------------------------------------------------------------------------


def test_the_config_is_written_as_yaml_with_a_do_not_edit_header(
    backend: AIToolkitBackend,
) -> None:
    path = backend.generate_config(spec_for())

    assert path.suffix == ".yaml"
    text = path.read_text(encoding="utf-8")
    assert "Do not hand-edit" in text

    payload = yaml.safe_load(text)
    assert payload["job"] == "extension"
    assert payload["config"]["process"][0]["type"] == "sd_trainer"


def test_regenerating_overwrites_rather_than_accumulating(backend: AIToolkitBackend) -> None:
    """A generated artifact, never a source of truth."""
    first = backend.generate_config(spec_for(steps=100))
    second = backend.generate_config(spec_for(steps=999))

    assert first == second
    assert yaml.safe_load(second.read_text(encoding="utf-8"))["config"]["process"][0]["train"][
        "steps"
    ] == 999


def test_paths_are_posix_in_the_config(backend: AIToolkitBackend) -> None:
    """The same config is read on the Windows desk and the Linux node."""
    text = backend.generate_config(
        spec_for(dataset="D:/data/poses", mask_path="D:/data/poses/masks")
    ).read_text(encoding="utf-8")
    assert "\\" not in text


def test_no_toolkit_path_is_a_clear_error(tmp_path: Path) -> None:
    bare = AIToolkitBackend(output_root=tmp_path)
    assert bare.available() is False
    with pytest.raises(BackendError, match="aitoolkit_path is not set"):
        bare.runner_script()


def test_available_reports_whether_the_checkout_is_there(tmp_path: Path) -> None:
    backend = AIToolkitBackend(tmp_path / "ai-toolkit", output_root=tmp_path)
    assert backend.available() is False

    (tmp_path / "ai-toolkit").mkdir()
    (tmp_path / "ai-toolkit" / "run.py").write_text("", encoding="utf-8")
    assert backend.available() is True


# --------------------------------------------------------------------------
# output parsing
# --------------------------------------------------------------------------


def test_parses_an_explicit_step_line() -> None:
    collector = Collector()
    parser = OutputParser(collector)

    parser("step: 42/2400")

    progress = collector.of(Progress)[0]
    assert (progress.step, progress.total) == (42, 2400)


def test_parses_a_tqdm_bar() -> None:
    collector = Collector()
    parser = OutputParser(collector)

    consumed = parser("blizzard:  10%|#         | 240/2400 [01:12<10:48,  3.33s/it]")

    assert collector.of(Progress)[0].step == 240
    assert consumed, "a redrawn bar should not also be logged verbatim"


def test_parses_loss_including_scientific_notation() -> None:
    collector = Collector()
    parser = OutputParser(collector)

    parser("step: 10/100 loss: 3.698e-01")
    parser("loss=0.2431")

    values = [e.value for e in collector.of(LossPoint)]
    assert values == pytest.approx([0.3698, 0.2431])
    assert collector.of(LossPoint)[0].step == 10, "loss is attributed to the current step"


def test_a_bar_for_something_else_does_not_move_the_run() -> None:
    """Caching and sampling have their own bars with their own totals."""
    collector = Collector()
    parser = OutputParser(collector, total=2400)

    parser("Caching latents: 100%|##########| 40/40 [00:03<00:00]")

    assert parser.step == 0
    assert not collector.of(Progress)


def test_trouble_is_raised_to_error() -> None:
    collector = Collector()
    parser = OutputParser(collector)

    parser("torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB")

    assert collector.lines("error")


def test_ordinary_lines_are_left_for_the_caller_to_log() -> None:
    parser = OutputParser(Collector())
    assert parser("Loading Flux2 model") is False


def test_progress_snapshot() -> None:
    parser = OutputParser(Collector())
    parser("step: 7/99")
    snapshot = parser.progress()
    assert (snapshot.step, snapshot.total) == (7, 99)
