"""Generated configs, loaded by the real ai-toolkit.

Doc 02 asks for "one integration test per backend that generates a config
and asserts its shape, without launching training". Asserting the shape
against our own expectations only proves we are self-consistent, so this
goes further: it hands the generated YAML to ai-toolkit's own
``get_job`` and checks that it resolves to a trainer with the right
architecture.

That is the check that would have caught v1's ``arch: flux2_klein``, which
is not an architecture ai-toolkit has.

Skipped unless a checkout is present. Point ``FLUXKREA_AITOOLKIT`` at one,
or leave it and these skip. It never launches training and never
downloads weights - ``get_job`` parses and instantiates the process;
weights load in ``process.run()``, which is not called.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fluxkrea.core.backends.aitoolkit import AIToolkitBackend
from fluxkrea.core.backends.spec import RunSpec

#: Where to look for a checkout, in order.
CANDIDATES = (
    os.environ.get("FLUXKREA_AITOOLKIT", ""),
    "D:/Projects_26/AI_Image_Trainer/ai-toolkit-krea2",
    "D:/Projects_26/AI_Image_Trainer/ai-toolkit",
)

#: The interpreter that has ai-toolkit's dependencies. Ours does not - no
#: torch, deliberately, since the laptop driving the fleet has none.
INTERPRETERS = (
    os.environ.get("FLUXKREA_AITOOLKIT_PYTHON", ""),
    "D:/Projects_26/AI_Image_Trainer/.venv/Scripts/python.exe",
    "D:/Projects_26/AI_Image_Trainer/.venv/bin/python",
)

#: get_job imports torch and the whole extension tree. Generous.
TIMEOUT = 300


def _first_existing(candidates: tuple[str, ...], suffix: str = "") -> Path | None:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if (path / suffix if suffix else path).exists():
            return path
    return None


TOOLKIT = _first_existing(CANDIDATES, "run.py")
PYTHON = _first_existing(INTERPRETERS)

needs_toolkit = pytest.mark.skipif(
    TOOLKIT is None or PYTHON is None,
    reason="no ai-toolkit checkout with a usable interpreter; set FLUXKREA_AITOOLKIT",
)

#: Runs inside ai-toolkit's interpreter, prints one JSON line per config.
PROBE = r"""
import contextlib, io, json, sys
sys.path.insert(0, ".")
from toolkit.job import get_job

out = []
for path in sys.argv[1:]:
    record = {"config": path}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            job = get_job(path)
        process = job.process[0]
        record.update(
            ok=True,
            process=type(process).__name__,
            arch=process.model_config.arch,
            datasets=len(process.datasets),
            mask_path=process.datasets[0].mask_path or "",
            mask_min_value=process.datasets[0].mask_min_value,
            lr=process.train_config.lr,
            unet_lr=getattr(process.train_config, "unet_lr", None),
            steps=process.train_config.steps,
            linear=process.network_config.linear,
        )
    except Exception as exc:
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    out.append(record)
print("PROBE_RESULT " + json.dumps(out))
"""


def probe(configs: list[Path]) -> list[dict]:
    """Load each config with the real ai-toolkit and report what it became."""
    assert TOOLKIT is not None and PYTHON is not None
    script = TOOLKIT / "_fluxkrea_probe.py"
    script.write_text(PROBE, encoding="utf-8")
    try:
        result = subprocess.run(  # noqa: S603 - argument list, no shell
            [str(PYTHON), str(script), *[str(c) for c in configs]],
            cwd=str(TOOLKIT),
            capture_output=True,
            text=True,
            timeout=TIMEOUT,
            check=False,
        )
    finally:
        script.unlink(missing_ok=True)

    for line in result.stdout.splitlines():
        if line.startswith("PROBE_RESULT "):
            return json.loads(line[len("PROBE_RESULT ") :])

    pytest.fail(
        f"the probe produced no result (exit {result.returncode}).\n"
        f"stdout tail:\n{result.stdout[-2000:]}\nstderr tail:\n{result.stderr[-2000:]}"
    )


@pytest.fixture(scope="module")
def loaded(tmp_path_factory: pytest.TempPathFactory) -> dict[str, dict]:
    """Generate a config per model and load them all in one subprocess."""
    if TOOLKIT is None or PYTHON is None:
        pytest.skip("no ai-toolkit checkout")

    root = tmp_path_factory.mktemp("aitoolkit-probe")
    dataset = root / "poses"
    (dataset / "masks").mkdir(parents=True)

    backend = AIToolkitBackend(TOOLKIT, output_root=root / "runs")
    configs: dict[str, Path] = {}
    for model in ("flux2", "flux2-klein-4b", "flux2-klein-9b", "krea2"):
        configs[model] = backend.generate_config(
            RunSpec(
                model=model,
                dataset=dataset.as_posix(),
                name=f"probe-{model}",
                steps=100,
                learning_rate=0.0002,
                network_dim=32,
                mask_path=(dataset / "masks").as_posix(),
                extra={"model_path": "/models/placeholder.safetensors"},
            )
        )

    records = probe(list(configs.values()))
    by_model = {}
    for model, path in configs.items():
        by_model[model] = next(r for r in records if Path(r["config"]) == path)
    return by_model


@needs_toolkit
@pytest.mark.slow
@pytest.mark.parametrize(
    ("model", "arch"),
    [
        ("flux2", "flux2"),
        ("flux2-klein-4b", "flux2_klein_4b"),
        ("flux2-klein-9b", "flux2_klein_9b"),
        ("krea2", "krea2"),
    ],
)
def test_the_config_loads_as_the_right_architecture(
    loaded: dict[str, dict], model: str, arch: str
) -> None:
    record = loaded[model]
    assert record["ok"], record.get("error")
    assert record["process"] == "SDTrainer"
    assert record["arch"] == arch


@needs_toolkit
@pytest.mark.slow
def test_ai_toolkit_derives_unet_lr_from_lr(loaded: dict[str, dict]) -> None:
    """The quirk, proved by the thing that has it.

    ai-toolkit reads ``lr`` and derives ``unet_lr``; a config that sets only
    ``learning_rate`` trains at a near-zero default.
    """
    record = loaded["flux2"]
    assert record["lr"] == 0.0002
    assert record["unet_lr"] == 0.0002


@needs_toolkit
@pytest.mark.slow
def test_the_mask_path_survives_into_the_dataset_config(loaded: dict[str, dict]) -> None:
    """The one line the masking feature exists to produce, as ai-toolkit sees it."""
    for model in ("flux2", "krea2"):
        record = loaded[model]
        assert record["mask_path"].endswith("/masks"), record
        assert record["mask_min_value"] == 0.0


@needs_toolkit
@pytest.mark.slow
def test_the_rest_of_the_config_arrives_intact(loaded: dict[str, dict]) -> None:
    record = loaded["flux2"]
    assert record["steps"] == 100
    assert record["datasets"] == 1
    assert record["linear"] == 32
