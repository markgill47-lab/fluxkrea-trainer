"""Projects: the grouping a room of students works in.

Two properties this file is really about. A project never owns the files -
deleting one must leave every dataset registered and every image where it
was - and a rename never moves the id, because the id is what every open
browser and every queued job is holding.
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from fluxkrea.daemon.projects import ProjectError, ProjectStore
from tests.daemon.conftest import register

# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------


def test_ids_are_derived_from_the_name(tmp_path: Path) -> None:
    store = ProjectStore(file=tmp_path / "projects.json")
    assert store.create("Tuesday Portraits").id == "tuesday-portraits"


def test_two_projects_may_share_a_name_but_never_an_id(tmp_path: Path) -> None:
    """Two students both typing "My Project" is the normal case, not the odd one."""
    store = ProjectStore(file=tmp_path / "projects.json")
    first = store.create("My Project")
    second = store.create("My Project")

    assert first.id != second.id
    assert first.name == second.name == "My Project"


def test_a_rename_does_not_move_the_id(tmp_path: Path) -> None:
    """The id is what jobs and open browsers hold. Deriving a new one orphans them."""
    store = ProjectStore(file=tmp_path / "projects.json")
    project = store.create("Untitled")
    renamed = store.rename(project.id, "Fight Choreography")

    assert renamed.id == project.id == "untitled"
    assert renamed.name == "Fight Choreography"


def test_an_empty_name_is_refused(tmp_path: Path) -> None:
    store = ProjectStore(file=tmp_path / "projects.json")
    with pytest.raises(ProjectError, match="needs a name"):
        store.create("   ")


def test_the_shared_config_is_replaced_not_merged(tmp_path: Path) -> None:
    """A cleared field must not read as an unmentioned one."""
    store = ProjectStore(file=tmp_path / "projects.json")
    project = store.create("Poses")
    store.set_config(project.id, {"model": "krea2", "steps": 2000})
    store.set_config(project.id, {"model": "flux2"})

    assert store.get(project.id).config == {"model": "flux2"}


def test_projects_survive_a_reload(tmp_path: Path) -> None:
    file = tmp_path / "projects.json"
    store = ProjectStore(file=file)
    project = store.create("Poses")
    store.set_config(project.id, {"resolution": 1024})
    store.add_dataset(project.id, "punches")

    reloaded = ProjectStore(file=file)
    assert reloaded.get(project.id).config == {"resolution": 1024}
    assert reloaded.get(project.id).datasets == ["punches"]


def test_one_malformed_project_does_not_lose_the_rest(tmp_path: Path) -> None:
    file = tmp_path / "projects.json"
    file.write_text(
        '{"version": 1, "projects": ['
        '{"name": "no id here"},'
        '{"id": "good", "name": "Good"}]}',
        encoding="utf-8",
    )
    assert [p.id for p in ProjectStore(file=file).list()] == ["good"]


def test_a_dataset_is_added_once_however_many_times_it_is_added(tmp_path: Path) -> None:
    store = ProjectStore(file=tmp_path / "projects.json")
    project = store.create("Poses")
    store.add_dataset(project.id, "punches")
    store.add_dataset(project.id, "punches")

    assert store.get(project.id).datasets == ["punches"]


# --------------------------------------------------------------------------
# the endpoints
# --------------------------------------------------------------------------


def test_a_project_starts_empty_and_is_listed(api: httpx.Client) -> None:
    assert api.get("/projects").json() == {"projects": [], "count": 0}

    created = api.post("/projects", json={"name": "Tuesday"})
    assert created.status_code == 201, created.text
    assert created.json()["id"] == "tuesday"
    assert api.get("/projects").json()["count"] == 1


def test_a_nameless_project_is_refused(api: httpx.Client) -> None:
    assert api.post("/projects", json={"name": ""}).status_code == 422


def test_datasets_join_and_leave_a_project(api: httpx.Client, dataset: Path) -> None:
    project = api.post("/projects", json={"name": "Tuesday"}).json()["id"]
    dataset_id = register(api, dataset)

    added = api.post(f"/projects/{project}/datasets", json={"dataset": dataset_id})
    assert added.status_code == 201, added.text
    assert added.json()["datasets"] == [dataset_id]

    removed = api.delete(f"/projects/{project}/datasets/{dataset_id}")
    assert removed.json()["datasets"] == []


def test_a_project_cannot_hold_a_dataset_the_node_does_not_have(api: httpx.Client) -> None:
    """A dangling id in a project is a row that 404s on click. Refuse it up front."""
    project = api.post("/projects", json={"name": "Tuesday"}).json()["id"]
    response = api.post(f"/projects/{project}/datasets", json={"dataset": "nope"})
    assert response.status_code == 404


def test_a_deregistered_dataset_is_reported_missing_not_hidden(
    api: httpx.Client, dataset: Path
) -> None:
    project = api.post("/projects", json={"name": "Tuesday"}).json()["id"]
    dataset_id = register(api, dataset)
    api.post(f"/projects/{project}/datasets", json={"dataset": dataset_id})

    api.delete(f"/datasets/{dataset_id}")

    payload = api.get(f"/projects/{project}").json()
    assert payload["datasets"] == []
    assert payload["missing"] == [dataset_id]


def test_deleting_a_project_keeps_every_dataset(api: httpx.Client, dataset: Path) -> None:
    """The promise this endpoint makes. Closing a project is not a way to lose work."""
    project = api.post("/projects", json={"name": "Tuesday"}).json()["id"]
    dataset_id = register(api, dataset)
    api.post(f"/projects/{project}/datasets", json={"dataset": dataset_id})

    deleted = api.delete(f"/projects/{project}")
    assert deleted.json()["deleted"] is True

    assert api.get("/projects").json()["count"] == 0
    assert [d["id"] for d in api.get("/datasets").json()["datasets"]] == [dataset_id]
    assert any(dataset.iterdir()), "the folder itself must be untouched"


def test_the_shared_config_round_trips_over_http(api: httpx.Client) -> None:
    project = api.post("/projects", json={"name": "Tuesday"}).json()["id"]
    saved = api.patch(
        f"/projects/{project}",
        json={"config": {"model": "krea2", "resolution": 1024}},
    )
    assert saved.status_code == 200, saved.text
    assert api.get(f"/projects/{project}").json()["config"]["model"] == "krea2"


def test_a_rename_and_a_config_save_are_independent(api: httpx.Client) -> None:
    project = api.post("/projects", json={"name": "Untitled"}).json()["id"]
    api.patch(f"/projects/{project}", json={"config": {"model": "krea2"}})
    api.patch(f"/projects/{project}", json={"name": "Fight Choreography"})

    payload = api.get(f"/projects/{project}").json()
    assert payload["name"] == "Fight Choreography"
    assert payload["config"] == {"model": "krea2"}, "a rename must not clear the config"


def test_a_config_that_is_not_an_object_is_refused(api: httpx.Client) -> None:
    project = api.post("/projects", json={"name": "Tuesday"}).json()["id"]
    assert api.patch(f"/projects/{project}", json={"config": "nope"}).status_code == 422


def test_an_unknown_project_is_a_404(api: httpx.Client) -> None:
    assert api.get("/projects/nope").status_code == 404
    assert api.delete("/projects/nope").status_code == 404
    assert api.patch("/projects/nope", json={"name": "x"}).status_code == 404
