"""What a running daemon holds. One object, built once, injected everywhere.

Kept separate from ``app.py`` so routes can be tested against a state
object without an HTTP layer, and so the app factory has nothing in it but
wiring.
"""

from __future__ import annotations

import platform
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .. import __version__
from ..core import paths
from ..core.config import Config, load
from ..core.dataset import scan
from ..core.dataset.item import DatasetItem
from .queue import JobQueue, JobRunner
from .registry import Dataset, Registry
from .security import Denied, check_path
from .tasks import TaskRunner


@dataclass
class State:
    config: Config = field(default_factory=load)
    registry: Registry = field(default_factory=Registry)
    tasks: TaskRunner = field(default=None)  # type: ignore[assignment]
    jobs: JobQueue = field(default=None)  # type: ignore[assignment]
    started: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.tasks is None:
            self.tasks = TaskRunner(workers=self.config.daemon.workers)
        if self.jobs is None:
            self.jobs = JobQueue(devices=max(1, gpu_count()))
        if self.jobs.runner is None:
            # Register this node's configured backends and give the queue
            # something to run. Import here: core must not import the
            # daemon, and this is the daemon side of that boundary.
            from .runner import attach

            attach(self)

    # -- node -------------------------------------------------------------

    @property
    def node_name(self) -> str:
        return self.config.daemon.node_name or platform.node()

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "node": self.node_name,
            "uptime": round(time.time() - self.started, 1),
            "queue_depth": self.jobs.depth(),
            "tasks_active": self.tasks.active(),
        }

    def backends(self) -> dict[str, Any]:
        """Which backends this node has, and whether they could actually run.

        A backend that is registered but has no checkout to launch is worth
        seeing *before* a job is submitted, not after it fails.
        """
        from ..core.backends import registered

        out: dict[str, Any] = {}
        for name, backend in registered().items():
            available = getattr(backend, "available", None)
            out[name] = {
                "ready": bool(available()) if callable(available) else True,
                "models": [m["id"] for m in getattr(backend, "models", lambda: [])()],
            }
        return out

    def node(self) -> dict[str, Any]:
        """Everything doc 06 wants visible *before* a job is submitted.

        The torch/CUDA/driver line is not decoration: Blackwell (sm_120)
        needs torch 2.6+ with CUDA 12.6+, and a fleet-wide version mismatch
        is otherwise invisible until a run dies confusingly.
        """
        import numpy
        from PIL import Image as PILImage

        import cv2

        from ..core.detect import available

        info: dict[str, Any] = {
            "name": self.node_name,
            "version": __version__,
            "hostname": platform.node(),
            "os": platform.system(),
            "os_release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "pillow": PILImage.__version__,
            "numpy": numpy.__version__,
            "detectors": available(),
            "backends": self.backends(),
            "models": _model_listing(),
            "paths": paths.describe(),
            "disk_free": disk_free(paths.data_dir()),
            **torch_versions(),
        }
        return info

    # -- dataset resolution -----------------------------------------------

    def dataset(self, dataset_id: str) -> Dataset:
        found = self.registry.get(dataset_id)
        if found is None:
            raise Denied(f"no dataset registered as {dataset_id!r}", status=404)
        return found

    def dataset_path(self, dataset_id: str) -> Path:
        dataset = self.dataset(dataset_id)
        if not dataset.path.is_dir():
            raise Denied(
                f"{dataset.id} is registered at {dataset.path.as_posix()}, which is not there",
                status=410,
            )
        return dataset.path

    def register(self, path: str, name: str | None = None, *, create: bool = False) -> Dataset:
        """Register a folder, after checking it is inside the allowed roots.

        ``create`` exists for the first push to a node that has never seen
        this dataset. It is still bounded by the root check, so it cannot
        make a directory anywhere interesting.
        """
        target = check_path(self.config, path)
        if create and not target.exists():
            paths.ensure_dir(target)
        if not target.is_dir():
            raise Denied(f"{target.as_posix()} is not a folder", status=400)
        return self.registry.register(target, name)

    def items(self, dataset_id: str) -> list[DatasetItem]:
        return scan(
            self.dataset_path(dataset_id),
            extensions=self.config.dataset.image_extensions,
            caption_ext=self.config.dataset.caption_ext,
        )

    def item(self, dataset_id: str, stem: str) -> DatasetItem:
        for item in self.items(dataset_id):
            if item.stem == stem:
                return item
        raise Denied(f"no item {stem!r} in {dataset_id}", status=404)

    # -- lifecycle --------------------------------------------------------

    def attach_job_runner(self, runner: JobRunner) -> None:
        """Give the queue something to run. The backends do this in P4."""
        self.jobs.runner = runner

    def shutdown(self) -> None:
        self.tasks.shutdown()
        self.jobs.stop()


# --------------------------------------------------------------------------
# node facts
# --------------------------------------------------------------------------


def _model_listing() -> list[dict[str, Any]]:
    from ..core.backends.models import listing

    return listing()


def torch_versions() -> dict[str, Any]:
    """torch, CUDA and driver, or nulls on a node with no torch installed.

    Imported lazily and defensively: the laptop driving the fleet has no
    torch, and asking it for node info must not be an error.
    """
    try:
        import torch
    except Exception:  # noqa: BLE001
        return {"torch": None, "cuda": None, "driver": None, "gpus": []}

    cuda = getattr(torch.version, "cuda", None)
    gpus: list[dict[str, Any]] = []
    try:
        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(index)
                free, total = torch.cuda.mem_get_info(index)
                gpus.append(
                    {
                        "index": index,
                        "name": props.name,
                        "capability": f"{props.major}.{props.minor}",
                        "vram_total": total,
                        "vram_free": free,
                    }
                )
    except Exception:  # noqa: BLE001 - a broken driver must not 500 the endpoint
        pass

    return {"torch": torch.__version__, "cuda": cuda, "driver": _driver_version(), "gpus": gpus}


def gpu_count() -> int:
    return len(torch_versions().get("gpus") or [])


def _driver_version() -> str | None:
    import shutil
    import subprocess

    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    first = out.stdout.strip().splitlines()
    return first[0].strip() if first else None


def disk_free(path: Path) -> dict[str, int] | None:
    import shutil

    try:
        usage = shutil.disk_usage(path if path.exists() else path.anchor or ".")
    except OSError:
        return None
    return {"total": usage.total, "free": usage.free}
