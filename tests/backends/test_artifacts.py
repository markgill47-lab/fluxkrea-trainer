"""Finding a run's LoRA, and publishing it where ComfyUI will load it.

The expensive mistake this file guards is the quiet one: a Krea 2 LoRA
copied into ``models/loras/flux2`` loads, generates noise, and reads as a
bad training run rather than as a misfiled file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core.backends.artifacts import (
    PublishError,
    final_artifact,
    find_artifacts,
    lora_destination,
    lora_family,
    publish,
)


def lora(folder: Path, name: str, body: bytes = b"weights") -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / name
    target.write_bytes(body)
    return target


# --------------------------------------------------------------------------
# finding
# --------------------------------------------------------------------------


def test_the_final_weights_come_before_the_rotated_checkpoints(tmp_path: Path) -> None:
    """ai-toolkit keeps three checkpoints plus the final one, all one stem."""
    lora(tmp_path, "femj_000000500.safetensors")
    lora(tmp_path, "femj_000001000.safetensors")
    lora(tmp_path, "femj.safetensors")

    found = find_artifacts(tmp_path)
    assert [a.name for a in found] == [
        "femj.safetensors",
        "femj_000001000.safetensors",
        "femj_000000500.safetensors",
    ]
    assert found[0].final and found[0].step is None
    assert found[1].step == 1000


def test_final_is_decided_by_the_name_not_the_mtime(tmp_path: Path) -> None:
    """A sample render touching a file last must not make it the answer."""
    final = lora(tmp_path, "femj.safetensors")
    checkpoint = lora(tmp_path, "femj_000001000.safetensors")
    checkpoint.touch()  # newer than the final weights

    assert final_artifact(tmp_path).path == final


def test_weights_one_folder_down_are_still_found(tmp_path: Path) -> None:
    """ai-toolkit appends the job name to ``training_folder`` itself."""
    lora(tmp_path / "femj-flux2", "femj.safetensors")
    assert [a.name for a in find_artifacts(tmp_path)] == ["femj.safetensors"]


def test_nothing_there_is_an_empty_list_not_an_error(tmp_path: Path) -> None:
    assert find_artifacts(tmp_path) == []
    assert find_artifacts(tmp_path / "never-existed") == []
    assert find_artifacts("") == []
    assert final_artifact(tmp_path) is None


def test_only_safetensors_count(tmp_path: Path) -> None:
    lora(tmp_path, "config_fluxkrea.yaml")
    lora(tmp_path, "sample_000100.jpg")
    assert find_artifacts(tmp_path) == []


# --------------------------------------------------------------------------
# where it goes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("model_id", "folder"),
    [
        ("flux2", "flux2"),
        ("flux2-klein-4b", "flux2"),
        ("flux2-klein-9b", "flux2"),
        ("klein-9b", "flux2"),  # via an alias
        ("krea2", "krea2"),
        ("flux1", "flux1"),
    ],
)
def test_each_model_names_its_own_folder(model_id: str, folder: str) -> None:
    assert lora_family(model_id) == folder


def test_an_unknown_model_refuses_rather_than_guessing() -> None:
    """A LoRA in the wrong family folder loads and produces noise."""
    with pytest.raises(PublishError, match="unknown model"):
        lora_family("krea2-experimental-final-v3")


def test_the_destination_is_under_models_loras(tmp_path: Path) -> None:
    (tmp_path / "models" / "loras").mkdir(parents=True)
    target = lora_destination(tmp_path, "krea2")
    assert target == tmp_path / "models" / "loras" / "krea2"


def test_no_comfyui_path_says_where_to_set_it(tmp_path: Path) -> None:
    with pytest.raises(PublishError, match="backends.comfyui_path"):
        lora_destination(None, "krea2")
    with pytest.raises(PublishError, match="not there"):
        lora_destination(tmp_path / "no-comfy-here", "krea2")


# --------------------------------------------------------------------------
# publishing
# --------------------------------------------------------------------------


def test_publishing_copies_into_the_models_folder(tmp_path: Path) -> None:
    source = lora(tmp_path / "run", "femj.safetensors", b"the weights")
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()

    written = publish(source, comfy, "krea2")

    assert written == comfy / "models" / "loras" / "krea2" / "femj.safetensors"
    assert written.read_bytes() == b"the weights"
    assert source.is_file(), "the run folder is the record and stays intact"


def test_a_flux_lora_and_a_krea_lora_do_not_share_a_folder(tmp_path: Path) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    publish(lora(tmp_path / "a", "one.safetensors"), comfy, "flux2-klein-9b")
    publish(lora(tmp_path / "b", "two.safetensors"), comfy, "krea2")

    loras = comfy / "models" / "loras"
    assert (loras / "flux2" / "one.safetensors").is_file()
    assert (loras / "krea2" / "two.safetensors").is_file()


def test_an_existing_file_is_not_silently_replaced(tmp_path: Path) -> None:
    """On a shared node the other file is somebody else's afternoon."""
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    theirs = lora(comfy / "models" / "loras" / "krea2", "poses.safetensors", b"theirs")
    mine = lora(tmp_path / "run", "poses.safetensors", b"mine")

    with pytest.raises(PublishError, match="already exists"):
        publish(mine, comfy, "krea2")
    assert theirs.read_bytes() == b"theirs"

    publish(mine, comfy, "krea2", overwrite=True)
    assert theirs.read_bytes() == b"mine"


def test_a_supplied_name_cannot_escape_the_folder(tmp_path: Path) -> None:
    """The name arrives over HTTP."""
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    source = lora(tmp_path / "run", "femj.safetensors")

    written = publish(source, comfy, "krea2", name="../../../autoexec")

    assert written.parent == comfy / "models" / "loras" / "krea2"
    assert written.name == "autoexec.safetensors"


def test_a_name_without_the_suffix_gets_one(tmp_path: Path) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    written = publish(lora(tmp_path / "run", "femj.safetensors"), comfy, "krea2", name="tuesday")
    assert written.name == "tuesday.safetensors"


def test_nothing_partial_is_left_in_the_folder_comfyui_scans(tmp_path: Path) -> None:
    """ComfyUI lists that folder; a copy in progress is a file it can find."""
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    publish(lora(tmp_path / "run", "femj.safetensors"), comfy, "krea2")

    folder = comfy / "models" / "loras" / "krea2"
    assert [p.name for p in folder.iterdir()] == ["femj.safetensors"]


def test_publishing_something_that_is_not_a_lora_is_refused(tmp_path: Path) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    config = lora(tmp_path / "run", "config_fluxkrea.yaml")

    with pytest.raises(PublishError, match="not a .safetensors"):
        publish(config, comfy, "krea2")
    with pytest.raises(PublishError, match="not there to publish"):
        publish(tmp_path / "run" / "gone.safetensors", comfy, "krea2")
