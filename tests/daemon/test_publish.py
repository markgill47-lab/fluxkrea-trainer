"""Getting the LoRA off the node: download, and publish into ComfyUI.

The endpoints a student uses once a run finishes. Over a tunnel or a LAN
URL the browser is not on the machine that trained anything, so "the file
is in your output folder" is not an answer.
"""

from __future__ import annotations

from pathlib import Path

import httpx

from fluxkrea.daemon.queue import RunSpec


def finished_job(api: httpx.Client, output: Path, model: str = "krea2") -> str:
    """A job whose output folder holds a final LoRA and one checkpoint."""
    output.mkdir(parents=True, exist_ok=True)
    (output / "poses.safetensors").write_bytes(b"the final weights")
    (output / "poses_000001000.safetensors").write_bytes(b"an earlier checkpoint")

    state = api.app_state  # type: ignore[attr-defined]
    state.jobs.runner = None  # nothing should pick this up and change it
    job = state.jobs.submit(
        RunSpec(model=model, dataset="poses", name="poses", output=output.as_posix())
    )
    return job.id


def test_the_artifacts_are_listed_final_first(api: httpx.Client, tmp_path: Path) -> None:
    job = finished_job(api, tmp_path / "run")
    payload = api.get(f"/jobs/{job}/artifacts").json()

    assert [a["name"] for a in payload["artifacts"]] == [
        "poses.safetensors",
        "poses_000001000.safetensors",
    ]
    assert payload["artifacts"][0]["final"] is True
    assert payload["family"] == "krea2"


def test_the_lora_downloads(api: httpx.Client, tmp_path: Path) -> None:
    job = finished_job(api, tmp_path / "run")
    response = api.get(f"/jobs/{job}/artifacts/poses.safetensors")

    assert response.status_code == 200
    assert response.content == b"the final weights"


def test_a_download_can_only_name_a_file_this_run_produced(
    api: httpx.Client, tmp_path: Path
) -> None:
    """The name arrives over HTTP; it selects from the listing, never a path."""
    job = finished_job(api, tmp_path / "run")
    (tmp_path / "elsewhere.safetensors").write_bytes(b"not this run's")

    assert api.get(f"/jobs/{job}/artifacts/elsewhere.safetensors").status_code == 404
    assert api.get(f"/jobs/{job}/artifacts/..%2Felsewhere.safetensors").status_code == 404


def test_publishing_puts_a_krea_lora_in_the_krea_folder(
    api: httpx.Client, tmp_path: Path
) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    api.app_state.config.backends.comfyui_path = comfy  # type: ignore[attr-defined]
    job = finished_job(api, tmp_path / "run", model="krea2")

    response = api.post(f"/jobs/{job}/publish", json={})
    assert response.status_code == 200, response.text

    written = Path(response.json()["published"])
    assert written == comfy / "models" / "loras" / "krea2" / "poses.safetensors"
    assert written.read_bytes() == b"the final weights"


def test_publishing_puts_a_flux_lora_in_the_flux2_folder(
    api: httpx.Client, tmp_path: Path
) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    api.app_state.config.backends.comfyui_path = comfy  # type: ignore[attr-defined]
    job = finished_job(api, tmp_path / "run", model="flux2-klein-9b")

    published = api.post(f"/jobs/{job}/publish", json={}).json()["published"]
    assert Path(published).parent == comfy / "models" / "loras" / "flux2"


def test_an_earlier_checkpoint_can_be_published_by_name(
    api: httpx.Client, tmp_path: Path
) -> None:
    """"The run at step 1000 was better than the one it finished on" is real."""
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    api.app_state.config.backends.comfyui_path = comfy  # type: ignore[attr-defined]
    job = finished_job(api, tmp_path / "run")

    response = api.post(
        f"/jobs/{job}/publish",
        json={"artifact": "poses_000001000.safetensors", "name": "poses-step1000"},
    )
    written = Path(response.json()["published"])
    assert written.name == "poses-step1000.safetensors"
    assert written.read_bytes() == b"an earlier checkpoint"


def test_a_collision_refuses_and_says_so(api: httpx.Client, tmp_path: Path) -> None:
    comfy = tmp_path / "ComfyUI"
    comfy.mkdir()
    api.app_state.config.backends.comfyui_path = comfy  # type: ignore[attr-defined]
    job = finished_job(api, tmp_path / "run")

    assert api.post(f"/jobs/{job}/publish", json={}).status_code == 200
    second = api.post(f"/jobs/{job}/publish", json={})
    assert second.status_code == 409
    assert "already exists" in second.json()["error"]

    assert api.post(f"/jobs/{job}/publish", json={"overwrite": True}).status_code == 200


def test_no_comfyui_configured_is_a_409_that_names_the_setting(
    api: httpx.Client, tmp_path: Path
) -> None:
    api.app_state.config.backends.comfyui_path = None  # type: ignore[attr-defined]
    job = finished_job(api, tmp_path / "run")

    response = api.post(f"/jobs/{job}/publish", json={})
    assert response.status_code == 409
    assert "backends.comfyui_path" in response.json()["error"]

    # And the listing says up front that the button cannot work yet.
    assert api.get(f"/jobs/{job}/artifacts").json()["publishable"] is False


def test_a_run_with_nothing_to_publish_says_which_state_it_is_in(
    api: httpx.Client, tmp_path: Path
) -> None:
    state = api.app_state  # type: ignore[attr-defined]
    state.jobs.runner = None
    job = state.jobs.submit(
        RunSpec(model="krea2", dataset="poses", output=(tmp_path / "empty").as_posix())
    )

    response = api.post(f"/jobs/{job.id}/publish", json={})
    assert response.status_code == 409
    assert "queued" in response.json()["error"]
