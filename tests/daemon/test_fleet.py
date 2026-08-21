"""Client-side fleet aggregation. No coordinator, so this is where it lives."""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.cli.fleet import Fleet, Node
from fluxkrea.core import paths


def write_fleet(text: str) -> Path:
    target = paths.fleet_file()
    paths.ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def test_no_fleet_file_is_an_empty_fleet() -> None:
    assert Fleet.load().nodes == []


def test_loading_the_node_list() -> None:
    write_fleet(
        """
[[node]]
name = "olympus-1"
url  = "http://localhost:8471"

[[node]]
name = "olympus-2"
url  = "http://localhost:8472/"
ssh  = "mark@olympus-2"
"""
    )
    fleet = Fleet.load()

    assert [n.name for n in fleet.nodes] == ["olympus-1", "olympus-2"]
    assert fleet.nodes[1].url == "http://localhost:8472", "the trailing slash is stripped"
    assert fleet.nodes[1].ssh == "mark@olympus-2"


def test_incomplete_entries_are_skipped() -> None:
    write_fleet('[[node]]\nname = "nameless"\n\n[[node]]\nname = "ok"\nurl = "http://x"\n')
    assert [n.name for n in Fleet.load().nodes] == ["ok"]


def test_url_for_a_known_and_an_unknown_node() -> None:
    write_fleet('[[node]]\nname = "olympus-1"\nurl = "http://localhost:8471"\n')
    fleet = Fleet.load()

    assert fleet.url_for("olympus-1") == "http://localhost:8471"
    with pytest.raises(ValueError, match="known: olympus-1"):
        fleet.url_for("olympus-9")


def test_a_broken_fleet_file_is_reported() -> None:
    write_fleet("[[node]\nbroken")
    with pytest.raises(ValueError, match="cannot read"):
        Fleet.load()


def test_a_node_that_is_down_is_one_row_not_a_failure() -> None:
    """Doc 06: a node being down degrades to a missing row."""
    write_fleet('[[node]]\nname = "gone"\nurl = "http://127.0.0.1:1"\n')

    rows = Fleet.load().status()

    assert len(rows) == 1
    assert rows[0]["node"] == "gone"
    assert rows[0]["state"] == "down"
    assert "error" in rows[0]


def test_status_reports_a_live_node(live_node) -> None:
    write_fleet(f'[[node]]\nname = "live"\nurl = "{live_node.base_url}"\n')

    rows = Fleet.load().status()

    assert rows[0]["state"] == "up"
    assert rows[0]["version"]
    assert rows[0]["queue_depth"] == 0


def test_datasets_are_gathered_from_every_node(live_node, dataset: Path) -> None:
    live_node.register(dataset)
    write_fleet(
        f'[[node]]\nname = "live"\nurl = "{live_node.base_url}"\n'
        '[[node]]\nname = "gone"\nurl = "http://127.0.0.1:1"\n'
    )

    placement = Fleet.load().datasets()

    assert "poses" in placement
    assert placement["poses"][0]["node"] == "live"
    assert placement["poses"][0]["files"] > 0
    assert placement["poses"][0]["state"] == "ok"


def test_where_names_the_nodes_that_have_a_dataset(live_node, dataset: Path) -> None:
    live_node.register(dataset)
    write_fleet(f'[[node]]\nname = "live"\nurl = "{live_node.base_url}"\n')

    assert [r["node"] for r in Fleet.load().where("poses")] == ["live"]
    assert Fleet.load().where("nothing-like-this") == []


def test_drift_is_reported_not_reconciled(live_node, second_live_node, dataset: Path, tmp_path: Path) -> None:
    """Copies are independent and therefore drift. Say so; change nothing."""
    import shutil

    other = tmp_path / "other-copy"
    shutil.copytree(dataset, other)
    (other / "extra.txt").write_text("this copy has more in it", encoding="utf-8")

    live_node.register(dataset)
    second_live_node.register(other, "poses")
    write_fleet(
        f'[[node]]\nname = "a"\nurl = "{live_node.base_url}"\n'
        f'[[node]]\nname = "b"\nurl = "{second_live_node.base_url}"\n'
    )

    placement = Fleet.load().datasets()

    states = {row["state"] for row in placement["poses"]}
    assert states == {"drift"}
    assert (other / "extra.txt").is_file(), "reconciliation is not in scope"


def test_node_serialises_for_json_output() -> None:
    assert Node(name="a", url="http://x").as_dict() == {"name": "a", "url": "http://x", "ssh": ""}
