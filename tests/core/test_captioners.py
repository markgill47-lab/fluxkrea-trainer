"""Captioners and the batch that drives them.

Everything here runs against a fake backend. A test suite that needs a
vision model running is a test suite nobody runs, and the interesting
behaviour is not "does llama3.2-vision describe a photograph" - it is what
the batch does when the backend refuses, dies, or was never there.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from fluxkrea.core.captioners import (
    Captioner,
    CaptionerError,
    available,
    from_config,
    get_captioner,
    labels,
    names,
)
from fluxkrea.core.captioners.joycaption import JoyCaptionCaptioner
from fluxkrea.core.captioners.ollama import OllamaCaptioner
from fluxkrea.core.config import CaptionerConfig
from fluxkrea.core.dataset.ops.caption import PREAMBLES, CaptionResult, _clean, caption


class Fake(Captioner):
    """A captioner whose answers the test writes in advance."""

    name = "fake"
    label = "Fake"

    def __init__(self, answers: list[tuple[bool, str]] | None = None, ready: bool = True) -> None:
        self.answers = answers or []
        self.ready = ready
        self.seen: list[Path] = []
        self.prompts: list[str] = []
        self.closed = False

    def test(self) -> tuple[bool, str]:
        return (True, "fake ready") if self.ready else (False, "fake is not running")

    def describe(self, image: Path, prompt: str, max_tokens: int = 400) -> tuple[bool, str]:
        self.seen.append(image)
        self.prompts.append(prompt)
        if self.answers:
            return self.answers.pop(0)
        return True, f"a description of {image.stem}"

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "set"
    root.mkdir()
    for index in range(4):
        Image.new("RGB", (64, 64), (index * 40, 60, 90)).save(root / f"img_{index:03d}.png")
    return root


# --------------------------------------------------------------------------
# registry
# --------------------------------------------------------------------------


def test_the_registry_builds_by_name() -> None:
    assert isinstance(get_captioner("ollama"), OllamaCaptioner)


def test_an_unknown_captioner_names_the_ones_that_exist() -> None:
    with pytest.raises(CaptionerError) as raised:
        get_captioner("blip2")
    assert "ollama" in str(raised.value)
    assert "joycaption" in str(raised.value)


def test_joycaption_is_registered_and_configured_from_its_own_fields() -> None:
    """The model this project was built around, and not an Ollama model."""
    built = from_config(
        CaptionerConfig(
            provider="joycaption",
            joycaption_model="fancyfeast/llama-joycaption-beta-one-hf-llava",
            joycaption_quantize=True,
        )
    )
    assert isinstance(built, JoyCaptionCaptioner)
    assert built.model_id.startswith("fancyfeast/")
    assert built.quantize is True


def test_joycaption_says_what_to_install_rather_than_raising() -> None:
    """A missing torch is an answer to `test()`, not an exception."""
    ok, message = JoyCaptionCaptioner().test()
    if not ok:
        # On a node without the extra installed - which is most of them.
        assert "joycaption" in message or "HuggingFace cache" in message


def test_joycaption_closes_cleanly_when_it_never_loaded() -> None:
    """close() runs after every batch, including ones that never started."""
    JoyCaptionCaptioner().close()  # must not raise


def test_availability_is_reported_without_probing_anything() -> None:
    """``GET /node`` must not make a network call to answer this."""
    ready = available()
    assert ready["ollama"] is True
    assert set(ready) == set(names())
    assert set(labels()) == set(names())


def test_config_feeds_the_right_fields_to_each_backend() -> None:
    settings = CaptionerConfig(
        provider="ollama",
        ollama_url="http://box:11434",
        ollama_model="llava",
        timeout=12.0,
    )
    built = from_config(settings)
    assert isinstance(built, OllamaCaptioner)
    assert built.url == "http://box:11434"
    assert built.model == "llava"
    assert built.timeout == 12.0


def test_switching_provider_does_not_lose_the_other_model() -> None:
    """The reason there are two model fields rather than one."""
    settings = CaptionerConfig(ollama_model="llava", claude_model="claude-opus-5")
    assert from_config(settings, provider="ollama").model == "llava"
    assert settings.claude_model == "claude-opus-5"


def test_a_trailing_slash_on_the_url_does_not_double_up() -> None:
    assert OllamaCaptioner(url="http://localhost:11434/").url == "http://localhost:11434"


def test_ollama_tolerates_a_model_named_without_its_tag(monkeypatch: pytest.MonkeyPatch) -> None:
    """People write ``llava``; ``/api/tags`` says ``llava:latest``."""
    captioner = OllamaCaptioner(model="llava")
    monkeypatch.setattr(captioner, "installed_models", lambda: (True, ["llava:latest"]))
    ok, message = captioner.test()
    assert ok, message


def test_a_missing_model_says_which_ones_are_there(monkeypatch: pytest.MonkeyPatch) -> None:
    """"Model not found" without the list is a guessing game."""
    captioner = OllamaCaptioner(model="llava")
    monkeypatch.setattr(captioner, "installed_models", lambda: (True, ["qwen2.5vl:7b"]))
    ok, message = captioner.test()
    assert not ok
    assert "ollama pull llava" in message
    assert "qwen2.5vl:7b" in message


def test_a_stopped_daemon_and_a_missing_model_are_different_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captioner = OllamaCaptioner()
    monkeypatch.setattr(captioner, "installed_models", lambda: (False, "connection refused"))
    ok, message = captioner.test()
    assert not ok
    assert "ollama serve" in message


def test_describe_refuses_a_missing_file(tmp_path: Path) -> None:
    ok, message = OllamaCaptioner().describe(tmp_path / "nope.png", "describe")
    assert not ok and "not found" in message


# --------------------------------------------------------------------------
# the batch
# --------------------------------------------------------------------------


def test_every_image_gets_a_caption(dataset: Path) -> None:
    backend = Fake()
    result = caption(dataset, backend)

    assert result.ok
    assert result.captioned == 4
    assert len(list(dataset.glob("*.txt"))) == 4
    assert (dataset / "img_000.txt").read_text(encoding="utf-8").strip()


def test_an_existing_caption_wins(dataset: Path) -> None:
    """A caption someone edited by hand is worth more than a new one."""
    (dataset / "img_001.txt").write_text("written by a person", encoding="utf-8")

    result = caption(dataset, Fake())
    assert result.captioned == 3
    assert result.skipped == 1
    assert (dataset / "img_001.txt").read_text(encoding="utf-8") == "written by a person"


def test_overwrite_replaces_them(dataset: Path) -> None:
    (dataset / "img_001.txt").write_text("written by a person", encoding="utf-8")

    result = caption(dataset, Fake(), overwrite=True)
    assert result.captioned == 4
    assert result.skipped == 0
    assert "person" not in (dataset / "img_001.txt").read_text(encoding="utf-8")


def test_a_dead_backend_is_reported_once_not_once_per_image(dataset: Path) -> None:
    """The whole reason the backend is probed before the loop."""
    backend = Fake(ready=False)
    result = caption(dataset, backend)

    assert not result.ok
    assert "not running" in result.aborted
    assert backend.seen == []  # nothing was even attempted
    assert result.failed == []  # and no image is blamed for it


def test_one_refusal_does_not_stop_the_batch(dataset: Path) -> None:
    backend = Fake([(False, "declined to describe this image"), (True, "fine"), (True, "fine")])
    result = caption(dataset, backend)

    assert result.captioned == 3
    assert len(result.failed) == 1
    assert not result.aborted


def test_failures_in_a_row_abort(dataset: Path) -> None:
    """Past a handful it is the backend that is broken, not the images."""
    backend = Fake([(False, "connection reset")] * 4)
    result = caption(dataset, backend, abort_after=2)

    assert "in a row" in result.aborted
    assert len(backend.seen) == 2  # stopped rather than grinding through
    assert not result.ok


def test_a_success_resets_the_failure_run(dataset: Path) -> None:
    backend = Fake([(False, "a"), (True, "ok"), (False, "b"), (True, "ok")])
    result = caption(dataset, backend, abort_after=2)

    assert not result.aborted
    assert result.captioned == 2
    assert len(result.failed) == 2


def test_cancelling_stops_the_loop(dataset: Path) -> None:
    import threading

    cancel = threading.Event()
    cancel.set()
    result = caption(dataset, Fake(), cancel=cancel)

    assert "cancelled" in result.aborted
    assert result.total == 0
    assert result.captioned == 0


def test_the_result_describes_the_whole_dataset_even_when_it_stops_early(
    dataset: Path,
) -> None:
    """A partial run still has to say what the dataset now contains."""
    backend = Fake([(False, "x")] * 2)
    result = caption(dataset, backend, abort_after=2)

    assert len(result.items) == 4
    assert len({item.stem for item in result.items}) == 4


def test_an_empty_folder_says_so_rather_than_probing(tmp_path: Path) -> None:
    backend = Fake(ready=False)
    result = caption(tmp_path, backend)
    assert result.total == 0
    assert result.ok


def test_a_prefix_is_prepended_to_every_caption(dataset: Path) -> None:
    """Where a LoRA trigger token goes."""
    caption(dataset, Fake(), prefix="mara_ohara")
    text = (dataset / "img_000.txt").read_text(encoding="utf-8")
    assert text.startswith("mara_ohara, ")


def test_the_prompt_reaches_the_backend(dataset: Path) -> None:
    backend = Fake()
    caption(dataset, backend, prompt="describe the lighting")
    assert set(backend.prompts) == {"describe the lighting"}


def test_the_backend_is_closed_when_the_run_ends(dataset: Path) -> None:
    """JoyCaption holds VRAM on the card that also runs training."""
    backend = Fake()
    caption(dataset, backend)
    assert backend.closed


def test_the_backend_is_closed_even_when_the_run_fails(dataset: Path) -> None:
    backend = Fake(ready=False)
    caption(dataset, backend)
    assert backend.closed


def test_the_result_serialises_for_the_api(dataset: Path) -> None:
    payload = caption(dataset, Fake()).as_dict()
    for key in ("root", "captioned", "skipped", "failed", "aborted", "ok"):
        assert key in payload


# --------------------------------------------------------------------------
# cleaning
# --------------------------------------------------------------------------


@pytest.mark.parametrize("preamble", PREAMBLES)
def test_every_known_preamble_is_stripped(preamble: str) -> None:
    cleaned = _clean(f"{preamble} a woman standing by a window")
    assert not cleaned.lower().startswith(preamble)
    assert "woman standing" in cleaned


def test_cleaning_collapses_whitespace_and_quotes() -> None:
    assert _clean('  "a  woman\n  standing"  ') == "a woman standing"


def test_cleaning_capitalises_what_it_uncovers() -> None:
    assert _clean("This image shows a woman standing").startswith("A woman")


def test_markdown_emphasis_is_stripped() -> None:
    """A LoRA trained on asterisks learns asterisks."""
    assert _clean("**Pose:** Standing. **Expression:** Neutral.") == (
        "Pose: Standing. Expression: Neutral."
    )
    assert _clean("a woman in a __red__ coat") == "a woman in a red coat"


def test_the_prefix_keeps_its_own_markdown() -> None:
    """Cleaning is for the model's answer; the prefix is the operator's."""
    assert _clean("a woman", prefix="**Mara**:") == "**Mara**: a woman"


def test_every_shipped_prompt_that_lists_fields_asks_for_prose() -> None:
    """Two captions in forty-two came back as forms. This is why."""
    from fluxkrea.core.captioners.prompts import BUILTIN, PROSE

    for name in ("person", "clothing", "style"):
        assert PROSE in BUILTIN[name], f"{name} lists what to cover but not how"


def test_a_prefix_keeps_the_separator_it_was_written_with() -> None:
    """Both conventions are deliberate; only the missing space is a typo."""
    assert _clean("a woman", prefix="mara") == "mara, a woman"
    assert _clean("a woman", prefix="mara,") == "mara, a woman"
    assert _clean("a woman", prefix="**Mara**:") == "**Mara**: a woman"


def test_a_prefix_survives_an_empty_caption() -> None:
    assert _clean("", prefix="mara") == "mara"


def test_the_result_summary_reads_as_a_sentence() -> None:
    result = CaptionResult(root=Path("."), captioned=3, skipped=1)
    result.failed.append(("img_009", "declined"))
    assert "3 captioned" in result.summary()
    assert "1 failed" in result.summary()
