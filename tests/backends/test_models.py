"""The model registry - explicit dispatch, replacing v1's string sniffing."""

from __future__ import annotations

import pytest

from fluxkrea.core.backends import models


def test_the_flux2_family_is_present() -> None:
    ids = [m.id for m in models.MODELS]
    assert "flux2" in ids
    assert "flux2-klein-4b" in ids
    assert "flux2-klein-9b" in ids
    assert "krea2" in ids


@pytest.mark.parametrize(
    ("model_id", "arch"),
    [
        ("flux2", "flux2"),
        ("flux2-klein-4b", "flux2_klein_4b"),
        ("flux2-klein-9b", "flux2_klein_9b"),
        ("krea2", "krea2"),
    ],
)
def test_arch_strings_are_the_ones_ai_toolkit_registers(model_id: str, arch: str) -> None:
    """Checked against extensions_built_in/diffusion_models/, not guessed.

    v1 emits ``flux2_klein``, which is not an architecture ai-toolkit has -
    the real ones are per-size - so its Klein configs fail at load.
    """
    assert models.get(model_id).arch == arch


def test_flux1_predates_arch_and_uses_is_flux() -> None:
    flux1 = models.get("flux1")
    assert flux1.arch == ""
    assert flux1.is_flux is True


def test_the_newer_models_do_not_set_is_flux() -> None:
    for model_id in ("flux2", "flux2-klein-4b", "krea2"):
        assert models.get(model_id).is_flux is False


@pytest.mark.parametrize(
    ("alias", "expected"),
    [
        ("FLUX2", "flux2"),
        ("flux.2", "flux2"),
        ("klein-4b", "flux2-klein-4b"),
        ("flux2_klein_9b", "flux2-klein-9b"),
        ("Krea-2", "krea2"),
        ("flux", "flux1"),
    ],
)
def test_aliases_and_case(alias: str, expected: str) -> None:
    assert models.get(alias).id == expected


def test_an_unknown_model_raises_and_says_what_is_known() -> None:
    """No fallback. v1 returns 'kohya' for anything it does not recognise."""
    with pytest.raises(models.UnknownModel) as exc:
        models.get("stable-diffusion-xl")

    message = str(exc.value)
    assert "unknown model" in message
    assert "flux2" in message, "the error should list what it does know"


def test_nothing_is_inferred_from_a_checkpoint_path() -> None:
    """v1 reads 'klein' and '4b' out of the filename; that is the bug."""
    assert models.find("my_flux_experiment_4b.safetensors") is None
    assert models.find("D:/models/krea2_raw.safetensors") is None


def test_klein_9b_keeps_its_text_encoder_off_the_gpu() -> None:
    """The 9B encoder does not fit beside the transformer on 16GB; 4B does."""
    assert models.get("flux2-klein-9b").low_vram is True
    assert models.get("flux2-klein-4b").low_vram is False


def test_klein_is_not_guidance_distilled_so_it_samples_lower() -> None:
    assert models.get("flux2-klein-4b").guidance_scale == 3.5
    assert models.get("flux2").guidance_scale == 4.0


def test_flux2_wants_a_larger_rank_than_flux1() -> None:
    """The fused transformer needs more than FLUX.1's default."""
    assert models.get("flux2").network_dim >= 32
    assert models.get("flux1").network_dim == 16


def test_listing_serialises_for_the_api() -> None:
    listing = models.listing()
    assert {entry["id"] for entry in listing} >= {"flux2", "krea2"}
    assert all("arch" in entry and "notes" in entry for entry in listing)
