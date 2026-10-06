#!/usr/bin/env python3
"""Load and validate NVIDIA Process Metrics platform definitions."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "packaging" / "supported-platforms.yaml"
REQUIRED_KEYS = {
    "prefix",
    "libdir",
    "systemd_unit_dir",
    "package_format",
    "cuda_root",
    "build_cuda_components",
    "enable_pm_sampling",
    "build_packages",
}


class PlatformConfigError(RuntimeError):
    pass


def _read_os_release(path: Path = Path("/etc/os-release")) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key] = value.strip().strip('"').strip("'")
    return values


def load_registry(path: Path = CONFIG_PATH) -> dict[str, Any]:
    # JSON is valid YAML. Keeping this file in the JSON subset avoids requiring
    # PyYAML on a machine that is only trying to bootstrap the build.
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PlatformConfigError(f"cannot read {path}: {error}") from error
    if registry.get("schema_version") != 1:
        raise PlatformConfigError("unsupported platform schema version")
    if not isinstance(registry.get("defaults"), dict) or not isinstance(
        registry.get("platforms"), dict
    ):
        raise PlatformConfigError("defaults and platforms must be mappings")
    return registry


def resolve_platform(name: str | None = None) -> tuple[str, dict[str, Any]]:
    registry = load_registry()
    platforms = registry["platforms"]
    if name is None:
        os_release = _read_os_release()
        candidates = [os_release.get("ID", "")]
        candidates.extend(os_release.get("ID_LIKE", "").split())
        name = next(
            (
                platform_name
                for candidate in candidates
                for platform_name, values in platforms.items()
                if candidate and candidate in values.get("distro_ids", [])
            ),
            "generic-linux",
        )
    if name not in platforms:
        choices = ", ".join(sorted(platforms))
        raise PlatformConfigError(f"unknown platform {name!r}; choose one of: {choices}")
    resolved = dict(registry["defaults"])
    resolved.update(platforms[name])
    missing = REQUIRED_KEYS.difference(resolved)
    if missing:
        raise PlatformConfigError(
            f"platform {name!r} is missing: {', '.join(sorted(missing))}"
        )
    for key in ("prefix", "cuda_root"):
        if not os.path.isabs(resolved[key]):
            raise PlatformConfigError(f"{name}.{key} must be an absolute path")
    if resolved["package_format"] not in {"none", "rpm", "deb"}:
        raise PlatformConfigError(f"unsupported package format for {name}")
    return name, resolved
