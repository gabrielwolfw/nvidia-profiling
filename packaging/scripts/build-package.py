#!/usr/bin/env python3
"""Configure, build, or stage the project for a supported Linux platform."""

from __future__ import annotations

import argparse
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

from platform_config import ROOT, PlatformConfigError, resolve_platform


RPM_SPEC_TEMPLATE = ROOT / "packaging" / "rpm" / "nvidia-process-metrics.spec.in"
DEB_TEMPLATE_DIR = ROOT / "packaging" / "deb"
RPM_SOURCE_PATHS = (
    "LICENSE",
    "meson.build",
    "meson_options.txt",
    "nvidia",
    "tools",
)


def discover_cuda_root(configured_root: str, override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().absolute()
    configured = Path(configured_root)
    if configured.is_dir():
        return configured
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    return configured


def discover_cuda_target(cuda_root: Path) -> Path | None:
    machine = platform.machine().lower()
    preferred_names = {
        "aarch64": ("aarch64-linux",),
        "arm64": ("aarch64-linux",),
        "x86_64": ("x86_64-linux",),
        "amd64": ("x86_64-linux",),
    }.get(machine, ())
    targets_dir = cuda_root / "targets"
    candidates = [targets_dir / name for name in preferred_names]
    if targets_dir.is_dir():
        candidates.extend(
            candidate
            for candidate in sorted(targets_dir.iterdir())
            if candidate.is_dir() and candidate not in candidates
        )
    for candidate in candidates:
        if (candidate / "include" / "cupti.h").exists() and (candidate / "lib").is_dir():
            return candidate.absolute()
    return None


def resolve_deb_multiarch(config: dict[str, object]) -> None:
    if config["package_format"] != "deb":
        return
    dpkg_architecture = shutil.which("dpkg-architecture")
    if dpkg_architecture is None:
        raise PlatformConfigError("dpkg-architecture is required for DEB packaging")
    multiarch = subprocess.run(
        [dpkg_architecture, "-qDEB_HOST_MULTIARCH"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if not multiarch or "/" in multiarch:
        raise PlatformConfigError(f"invalid Debian multiarch value: {multiarch!r}")
    config["libdir"] = f"lib/{multiarch}"


def validate_cuda(config: dict[str, object]) -> None:
    cuda_root = Path(str(config["cuda_root"]))
    cuda_target = str(config.get("cuda_target_root", ""))
    if cuda_target:
        target = Path(cuda_target)
        cuda_include = target / "include"
        cupti_include = target / "include"
        cupti_libdir = target / "lib"
    else:
        cuda_include = cuda_root / "include"
        cupti_include = cuda_root / "extras" / "CUPTI" / "include"
        cupti_libdir = cuda_root / "extras" / "CUPTI" / "lib64"
    required = (
        cuda_root / "bin" / "nvcc",
        cuda_include / "cuda.h",
        cupti_include / "cupti.h",
        cupti_libdir,
    )
    missing = [str(path) for path in required if not path.exists()]
    if not missing:
        return
    module = config.get("cuda_module")
    hint = f"; try `module load {module}` first" if module else ""
    details = "\n  ".join(missing)
    raise PlatformConfigError(
        f"CUDA toolkit validation failed{hint}. Missing:\n  {details}"
    )


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
        f"-Dnvidia_toolkit_target={config.get('cuda_target_root', '')}",
        f"-Dbuild_cuda_components={enabled(config['build_cuda_components'])}",
        f"-Denable_pm_sampling={enabled(config['enable_pm_sampling'])}",
    ]


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def _rpm_file_list(config: dict[str, object]) -> str:
    prefix = str(config["prefix"]).rstrip("/")
    libdir = str(config["libdir"]).strip("/")
    service_dir = str(config["systemd_unit_dir"]).strip("/")
    files = [
        "%{_bindir}/nvidia-process-metrics-launcher",
        "%{_sbindir}/nvidia-process-metrics-daemon",
        f"{prefix}/{service_dir}/nvidia-process-metrics-daemon.service",
    ]
    if config["build_cuda_components"]:
        component_dir = f"{prefix}/{libdir}/nvidia-process-metrics"
        files.extend(
            (
                f"%dir {component_dir}",
                f"{component_dir}/libnvidia-process-metrics.so",
            )
        )
        if config["enable_pm_sampling"]:
            files.append(f"{component_dir}/pm_sampling_simple")
    return "\n".join(files)


def render_rpm_spec(config: dict[str, object]) -> str:
    values = {
        "PACKAGE_NAME": str(config["package_name"]),
        "PACKAGE_VERSION": str(config["package_version"]),
        "PACKAGE_RELEASE": str(config["package_release"]),
        "PREFIX": str(config["prefix"]),
        "LIBDIR": str(config["libdir"]),
        "SYSTEMD_UNIT_DIR": str(config["systemd_unit_dir"]).strip("/"),
        "CUDA_ROOT": str(config["cuda_root"]),
        "CUDA_TARGET_ROOT": str(config.get("cuda_target_root", "")),
        "BUILD_CUDA_COMPONENTS": "true" if config["build_cuda_components"] else "false",
        "ENABLE_PM_SAMPLING": "true" if config["enable_pm_sampling"] else "false",
        "FILE_LIST": _rpm_file_list(config),
    }
    for key in ("PACKAGE_NAME", "PACKAGE_VERSION", "PACKAGE_RELEASE"):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._+~-]*", values[key]):
            raise PlatformConfigError(f"invalid RPM {key.lower()}: {values[key]!r}")
    spec = RPM_SPEC_TEMPLATE.read_text(encoding="utf-8")
    for key, value in values.items():
        spec = spec.replace(f"@{key}@", value)
    unresolved = re.findall(r"@[A-Z][A-Z0-9_]*@", spec)
    if unresolved:
        raise PlatformConfigError(
            f"unresolved RPM spec placeholders: {', '.join(sorted(set(unresolved)))}"
        )
    return spec


def create_source_archive(destination: Path, name: str, version: str) -> None:
    source_root = f"{name}-{version}"
    with tarfile.open(destination, "w:gz", format=tarfile.PAX_FORMAT) as archive:
        for relative_name in RPM_SOURCE_PATHS:
            source = ROOT / relative_name
            if not source.exists():
                raise PlatformConfigError(f"RPM source is missing: {source}")
            archive.add(
                source,
                arcname=f"{source_root}/{relative_name}",
                filter=lambda info: None
                if "__pycache__" in Path(info.name).parts or info.name.endswith(".pyc")
                else info,
            )


def build_rpm(config: dict[str, object], platform_name: str) -> None:
    if config["package_format"] != "rpm":
        raise PlatformConfigError(
            f"platform {platform_name!r} uses {config['package_format']!r}, not RPM"
        )
    rpmbuild = shutil.which("rpmbuild")
    if rpmbuild is None:
        raise PlatformConfigError(
            "rpmbuild is required; install rpm-build (Fedora/Rocky/RHEL)"
        )

    name = str(config["package_name"])
    version = str(config["package_version"])
    topdir = (ROOT / "dist" / platform_name / "rpmbuild").resolve()
    output_dir = (ROOT / "dist" / platform_name / "rpm").resolve()
    if topdir.exists():
        shutil.rmtree(topdir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for directory in ("BUILD", "BUILDROOT", "RPMS", "SOURCES", "SPECS", "SRPMS"):
        (topdir / directory).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_path = topdir / "SPECS" / f"{name}.spec"
    spec_path.write_text(render_rpm_spec(config), encoding="utf-8")
    create_source_archive(topdir / "SOURCES" / f"{name}-{version}.tar.gz", name, version)
    run(
        [
            rpmbuild,
            "-ba",
            str(spec_path),
            "--define",
            f"_topdir {topdir}",
            "--define",
            f"_rpmdir {output_dir}",
            "--define",
            f"_srcrpmdir {output_dir}",
            "--define",
            "_rpmfilename %{NAME}-%{VERSION}-%{RELEASE}.%{ARCH}.rpm",
        ]
    )
    packages = sorted(output_dir.glob("*.rpm"))
    if not packages:
        raise PlatformConfigError("rpmbuild completed without producing an RPM")
    print("RPM artifacts:")
    for package in packages:
        print(f"  {package}")


def _deb_template_values(config: dict[str, object]) -> dict[str, str]:
    return {
        "PACKAGE_NAME": str(config["package_name"]),
        "PREFIX": str(config["prefix"]),
        "LIBDIR": str(config["libdir"]),
        "SYSTEMD_UNIT_DIR": str(config["systemd_unit_dir"]).strip("/"),
        "CUDA_ROOT": str(config["cuda_root"]),
        "CUDA_TARGET_ROOT": str(config.get("cuda_target_root", "")),
        "BUILD_CUDA_COMPONENTS": "true" if config["build_cuda_components"] else "false",
        "ENABLE_PM_SAMPLING": "true" if config["enable_pm_sampling"] else "false",
        "DH_SHLIBDEPS_ARGS": (
            "-- --ignore-missing-info" if config["build_cuda_components"] else ""
        ),
    }


def render_deb_template(template: Path, config: dict[str, object]) -> str:
    rendered = template.read_text(encoding="utf-8")
    for key, value in _deb_template_values(config).items():
        rendered = rendered.replace(f"@{key}@", value)
    unresolved = re.findall(r"@[A-Z][A-Z0-9_]*@", rendered)
    if unresolved:
        raise PlatformConfigError(
            f"unresolved DEB template placeholders: {', '.join(sorted(set(unresolved)))}"
        )
    return rendered


def _copy_package_sources(destination: Path) -> None:
    for relative_name in RPM_SOURCE_PATHS:
        source = ROOT / relative_name
        target = destination / relative_name
        if not source.exists():
            raise PlatformConfigError(f"package source is missing: {source}")
        if source.is_dir():
            shutil.copytree(
                source,
                target,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def build_deb(config: dict[str, object], platform_name: str) -> None:
    if config["package_format"] != "deb":
        raise PlatformConfigError(
            f"platform {platform_name!r} uses {config['package_format']!r}, not DEB"
        )
    dpkg_buildpackage = shutil.which("dpkg-buildpackage")
    if dpkg_buildpackage is None:
        raise PlatformConfigError(
            "dpkg-buildpackage is required; install dpkg-dev and debhelper"
        )

    name = str(config["package_name"])
    version = str(config["package_version"])
    release = str(config["package_release"])
    for label, value in (("name", name), ("version", version), ("release", release)):
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.+:~-]*", value):
            raise PlatformConfigError(f"invalid Debian package {label}: {value!r}")

    work_dir = (ROOT / "dist" / platform_name / "dpkg-build").resolve()
    output_dir = (ROOT / "dist" / platform_name / "deb").resolve()
    if work_dir.exists():
        shutil.rmtree(work_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    source_dir = work_dir / f"{name}-{version}"
    source_dir.mkdir(parents=True)
    output_dir.mkdir(parents=True)
    _copy_package_sources(source_dir)

    debian_dir = source_dir / "debian"
    (debian_dir / "source").mkdir(parents=True)
    (debian_dir / "control").write_text(
        render_deb_template(DEB_TEMPLATE_DIR / "control.in", config),
        encoding="utf-8",
    )
    rules = debian_dir / "rules"
    rules.write_text(
        render_deb_template(DEB_TEMPLATE_DIR / "rules.in", config),
        encoding="utf-8",
    )
    rules.chmod(0o755)
    shutil.copy2(DEB_TEMPLATE_DIR / "copyright", debian_dir / "copyright")
    shutil.copy2(DEB_TEMPLATE_DIR / "source-format", debian_dir / "source" / "format")
    (debian_dir / "changelog").write_text(
        f"{name} ({version}-{release}) unstable; urgency=medium\n\n"
        "  * Build native Debian package.\n\n"
        " -- NVIDIA Process Metrics maintainers <noreply@example.invalid>  "
        "Fri, 09 Oct 2026 00:00:00 -0600\n",
        encoding="utf-8",
    )

    subprocess.run(
        [dpkg_buildpackage, "--build=binary", "--no-sign"],
        cwd=source_dir,
        check=True,
    )
    artifacts = sorted(
        path
        for path in work_dir.iterdir()
        if path.is_file()
        and path.suffix in {".deb", ".ddeb", ".changes", ".buildinfo"}
    )
    packages = [path for path in artifacts if path.suffix in {".deb", ".ddeb"}]
    if not packages:
        raise PlatformConfigError("dpkg-buildpackage completed without producing a DEB")
    for artifact in artifacts:
        shutil.move(str(artifact), output_dir / artifact.name)
    print("DEB artifacts:")
    for artifact in sorted(output_dir.iterdir()):
        print(f"  {artifact}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", choices=("detect", "configure", "build", "stage", "rpm", "deb")
    )
    parser.add_argument("--platform", help="override /etc/os-release detection")
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--destdir", type=Path)
    parser.add_argument(
        "--cuda-root",
        type=Path,
        help="override the CUDA toolkit root; otherwise use the platform value or loaded nvcc",
    )
    parser.add_argument(
        "--cuda-target-root",
        type=Path,
        help="override the CUDA target prefix containing include/ and lib/",
    )
    parser.add_argument(
        "--without-cuda",
        action="store_true",
        help="configure only the daemon and launcher (useful for portable validation)",
    )
    parser.add_argument(
        "--without-pm-sampling",
        action="store_true",
        help="build CUDA telemetry and kernel activity, but disable privileged PM Sampling",
    )
    args = parser.parse_args()

    name, config = resolve_platform(args.platform)
    config["cuda_root"] = str(
        discover_cuda_root(str(config["cuda_root"]), args.cuda_root)
    )
    cuda_target = (
        args.cuda_target_root.expanduser().absolute()
        if args.cuda_target_root
        else discover_cuda_target(Path(str(config["cuda_root"])))
    )
    config["cuda_target_root"] = str(cuda_target) if cuda_target else ""
    resolve_deb_multiarch(config)
    if args.without_cuda:
        config["build_cuda_components"] = False
        config["enable_pm_sampling"] = False
    elif args.without_pm_sampling:
        config["enable_pm_sampling"] = False
    build_dir = (args.build_dir or ROOT / f"build-{name}").resolve()
    destdir = (args.destdir or ROOT / "dist" / name / "root").resolve()

    print(f"platform: {name}")
    for key in (
        "description",
        "package_format",
        "package_name",
        "package_version",
        "package_release",
        "prefix",
        "libdir",
        "systemd_unit_dir",
        "cuda_root",
        "cuda_target_root",
        "cuda_module",
        "build_cuda_components",
        "enable_pm_sampling",
        "build_packages",
    ):
        print(f"{key}: {config.get(key)}")
    if args.action == "detect":
        return 0

    if args.action == "rpm":
        if config["build_cuda_components"]:
            validate_cuda(config)
        build_rpm(config, name)
        return 0

    if args.action == "deb":
        if config["build_cuda_components"]:
            validate_cuda(config)
        build_deb(config, name)
        return 0

    if config["build_cuda_components"]:
        validate_cuda(config)

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
