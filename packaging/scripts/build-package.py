#!/usr/bin/env python3
"""Configure, build, or stage the project for a supported Linux platform."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

from platform_config import ROOT, PlatformConfigError, resolve_platform


def meson_arguments(config: dict[str, object], build_dir: Path) -> list[str]:
    enabled = lambda value: "true" if value else "false"
    return [
        "meson",
        "setup",
        str(build_dir),
        str(ROOT),
        f"--prefix={config['prefix']}",
        f"--libdir={config['libdir']}",
        f"-Dsystemd_unit_dir={config['systemd_unit_dir']}",
        f"-Dnvidia_toolkit_root={config['cuda_root']}",
        f"-Dbuild_cuda_components={enabled(config['build_cuda_components'])}",
        f"-Denable_pm_sampling={enabled(config['enable_pm_sampling'])}",
    ]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("detect", "configure", "build", "stage"))
    parser.add_argument("--platform", help="override /etc/os-release detection")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--destdir", type=Path)
    parser.add_argument(
        "--without-cuda",
        action="store_true",
        help="configure only the daemon and launcher (useful for portable validation)",
    )
    args = parser.parse_args()

    name, config = resolve_platform(args.platform)
    if args.without_cuda:
        config["build_cuda_components"] = False
        config["enable_pm_sampling"] = False
    build_dir = (args.build_dir or ROOT / f"build-{name}").resolve()
    destdir = (args.destdir or ROOT / "dist" / name / "root").resolve()

    print(f"platform: {name}")
    for key in (
        "description",
        "package_format",
        "prefix",
        "libdir",
        "systemd_unit_dir",
        "cuda_root",
        "build_packages",
    ):
        print(f"{key}: {config.get(key)}")
    if args.action == "detect":
        return 0

    setup = meson_arguments(config, build_dir)
    if (build_dir / "meson-private" / "coredata.dat").exists():
        setup.insert(2, "--reconfigure")
    run(setup)
    if args.action == "configure":
        return 0
    run(["meson", "compile", "-C", str(build_dir)])
    if args.action == "build":
        return 0
    if destdir.exists():
        shutil.rmtree(destdir)
    run(["meson", "install", "-C", str(build_dir), f"--destdir={destdir}"])
    print(f"staged installation: {destdir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (PlatformConfigError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
