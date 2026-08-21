"""The saved prompt library.

The behaviour worth pinning is the shadowing rule: a built-in can be saved
over but never destroyed, and deleting your version brings the original
back. Everything else here is a JSON file.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from fluxkrea.core.captioners.base import DEFAULT_PROMPT
from fluxkrea.core.captioners.prompts import BUILTIN, PROSE, PromptLibrary


@pytest.fixture
def library(tmp_path: Path) -> PromptLibrary:
    return PromptLibrary(file=tmp_path / "prompts.json")


# --------------------------------------------------------------------------
# built-ins
# --------------------------------------------------------------------------


def test_the_builtins_are_there_before_anything_is_saved(library: PromptLibrary) -> None:
    names = [prompt.name for prompt in library.all()]
    assert "default" in names and "person" in names
    assert all(prompt.builtin for prompt in library.all())


def test_the_default_prompt_is_the_one_captioners_use(library: PromptLibrary) -> None:
    """One default, not two that drift apart."""
    assert library.get("default") == DEFAULT_PROMPT


def test_every_prompt_that_lists_fields_also_asks_for_prose() -> None:
    """Asked to cover a list, a model may answer with a labelled form.

    Two captions out of forty-two came back as "**Pose:** ... **Expression:**
    ..." before these prompts said how to write, not just what to cover.
    """
    for name in ("person", "clothing", "style"):
        assert PROSE in BUILTIN[name]


# --------------------------------------------------------------------------
# saving
# --------------------------------------------------------------------------


def test_a_saved_prompt_survives_a_reload(library: PromptLibrary, tmp_path: Path) -> None:
    library.save("mara", "Describe her jacket.")
    reopened = PromptLibrary(file=tmp_path / "prompts.json").load()
    assert reopened.get("mara") == "Describe her jacket."


def test_saving_trims_and_refuses_nothing(library: PromptLibrary) -> None:
    saved = library.save("  spaced  ", "  some text  ")
    assert saved.name == "spaced"
    assert saved.text == "some text"

    with pytest.raises(ValueError):
        library.save("", "text")
    with pytest.raises(ValueError):
        library.save("named", "   ")


def test_saving_twice_replaces_rather_than_duplicates(library: PromptLibrary) -> None:
    library.save("mara", "first")
    library.save("mara", "second")
    assert library.get("mara") == "second"
    assert [p.name for p in library.all()].count("mara") == 1


def test_a_saved_prompt_is_listed_after_the_builtins(library: PromptLibrary) -> None:
    library.save("zzz", "text")
    names = [prompt.name for prompt in library.all()]
    assert names[-1] == "zzz"
    assert names[0] == "default"


# --------------------------------------------------------------------------
# shadowing
# --------------------------------------------------------------------------


def test_saving_over_a_builtin_shadows_it(library: PromptLibrary) -> None:
    library.save("person", "my own version")

    found = next(p for p in library.all() if p.name == "person")
    assert found.text == "my own version"
    assert found.builtin is False
    assert found.shadows_builtin is True


def test_deleting_a_shadow_restores_the_builtin(library: PromptLibrary) -> None:
    """A bad edit must not cost a prompt you cannot get back."""
    library.save("person", "my own version")
    assert library.delete("person") is True
    assert library.get("person") == BUILTIN["person"]


def test_a_builtin_that_was_never_shadowed_cannot_be_deleted(library: PromptLibrary) -> None:
    assert library.delete("person") is False
    assert library.get("person") == BUILTIN["person"]


def test_deleting_something_that_does_not_exist_says_so(library: PromptLibrary) -> None:
    assert library.delete("nothing-here") is False


# --------------------------------------------------------------------------
# resolving
# --------------------------------------------------------------------------


def test_resolve_prefers_the_named_prompt(library: PromptLibrary) -> None:
    library.save("mara", "her jacket")
    assert library.resolve("mara", "fallback text") == "her jacket"


def test_resolve_falls_back_to_the_given_text(library: PromptLibrary) -> None:
    assert library.resolve(None, "fallback text") == "fallback text"
    assert library.resolve("", "fallback text") == "fallback text"


def test_an_unknown_name_gives_the_default_not_an_empty_prompt(
    library: PromptLibrary,
) -> None:
    """A typo should produce ordinary captions, not 200 images described by nothing."""
    assert library.resolve("typo", "") == DEFAULT_PROMPT


def test_resolve_with_nothing_at_all_still_returns_a_prompt(library: PromptLibrary) -> None:
    assert library.resolve(None, "") == DEFAULT_PROMPT


# --------------------------------------------------------------------------
# robustness
# --------------------------------------------------------------------------


def test_a_corrupt_file_is_ignored_rather_than_fatal(tmp_path: Path) -> None:
    """Losing a prompt list is annoying; refusing to caption is worse."""
    target = tmp_path / "prompts.json"
    target.write_text("{not json at all", encoding="utf-8")

    library = PromptLibrary(file=target).load()
    assert library.get("default") == DEFAULT_PROMPT
    assert [p.name for p in library.all() if not p.builtin] == []


def test_a_missing_file_is_not_an_error(tmp_path: Path) -> None:
    library = PromptLibrary(file=tmp_path / "never-written.json").load()
    assert library.names()


def test_junk_entries_are_dropped_on_load(tmp_path: Path) -> None:
    target = tmp_path / "prompts.json"
    target.write_text('{"prompts": {"good": "text", "empty": "  ", "": "x"}}', encoding="utf-8")

    library = PromptLibrary(file=target).load()
    saved = [p.name for p in library.all() if not p.builtin]
    assert saved == ["good"]


def test_the_file_is_written_atomically(library: PromptLibrary) -> None:
    """A half-written list read back as corrupt would empty the library."""
    library.save("mara", "text")
    assert library.file.is_file()
    assert not library.file.with_suffix(".json.tmp").exists()
