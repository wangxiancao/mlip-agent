"""Compute-environment self-check.

Run with the COMPUTE_VENV interpreter:

    $COMPUTE_VENV/bin/python app/compute/scripts/pinspect.py

Prints a JSON report with versions of the compute stack and CUDA
availability. Exit code 0 when every required package imports.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import sys

# module 名 -> pip 发行版名（不一致时用于元数据回退，如 fairchem -> fairchem-core）
PACKAGES = {
    "torch": "torch",
    "ase": "ase",
    "maple": "maple",
    "nequip": "nequip",
    "fairchem": "fairchem-core",
}


def _version(module: str, dist: str) -> str | None:
    try:
        mod = importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001 - report any import failure
        print(f"[pinspect] WARN: import {module} failed: {exc}", file=sys.stderr)
        return None
    version = getattr(mod, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(dist)
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    return version


def main() -> int:
    report: dict[str, object] = {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {mod: _version(mod, dist) for mod, dist in PACKAGES.items()},
    }
    try:
        import torch

        report["cuda_available"] = bool(torch.cuda.is_available())
        report["cuda_device"] = (
            torch.cuda.get_device_name(0) if report["cuda_available"] else None
        )
    except Exception as exc:  # noqa: BLE001
        report["cuda_available"] = False
        report["cuda_error"] = str(exc)

    print(json.dumps(report, indent=2, ensure_ascii=False))
    missing = [k for k, v in report["packages"].items() if v is None]  # type: ignore[union-attr]
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
