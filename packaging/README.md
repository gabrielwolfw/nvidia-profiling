# Platform-aware builds

`supported-platforms.yaml` is the registry for Linux installation layouts.
It uses the JSON subset of YAML so the bootstrap scripts need only Python's
standard library.

Detect the current distribution:

```bash
python3 packaging/scripts/build-package.py detect
```

Configure, build, and stage an installation:

```bash
python3 packaging/scripts/build-package.py stage --platform generic-linux
```

Fedora and Enterprise Linux are separate profiles because their RPM release
tags, dependency versions, and CUDA support can differ. Rocky Linux, RHEL,
CentOS, and AlmaLinux automatically select `rocky`. The ORFEO environment is
an explicit profile because its CUDA path is specific to that cluster:

```bash
module load cuda/12.8
python3 packaging/scripts/build-package.py stage --platform rocky-orfeo
```

ORFEO provides two isolated build profiles:

```bash
# Complete root daemon with PM Sampling
python3 packaging/scripts/build-package.py stage --platform rocky-orfeo

# User daemon and launcher without sudo or PM Sampling
python3 packaging/scripts/build-package.py stage --platform rocky-orfeo-user
```

Do not assign a distribution ID to the ORFEO profiles: a Rocky machine outside
the cluster must not inherit ORFEO paths. For other module-based installations,
load CUDA and override or auto-discover its root:

```bash
python3 packaging/scripts/build-package.py stage \
  --platform rocky \
  --cuda-root "$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
```

When the configured CUDA directory does not exist and `nvcc` is available in
`PATH`, the helper derives the toolkit root automatically. Before configuring,
it checks for NVCC, the CUDA headers, and the CUPTI headers and library folder.

Use `--without-cuda` for layout validation on machines without the NVIDIA CUDA
toolkit. This does not produce a functional profiler installation.

## Build RPM packages

On Fedora, Rocky Linux, RHEL, CentOS Stream, or AlmaLinux, install the packages
listed by the selected profile (including `rpm-build`) and build both the
binary RPM and source RPM with:

```bash
python3 packaging/scripts/build-package.py rpm --platform rocky
```

The artifacts are written to `dist/rocky/rpm/`. The RPM build runs Meson from
the generated source archive rather than packaging an existing build tree.
CUDA-enabled packages require the configured toolkit to be present. For a
layout-only package test on a machine without CUDA, use:

```bash
python3 packaging/scripts/build-package.py rpm --platform rocky --without-cuda
```

Use `--without-pm-sampling` to keep CUDA/NVML telemetry and CUPTI kernel
activity while producing a daemon that a regular user can run. This supports
the launcher without a system installation, but intentionally omits hardware
PM Sampling.

To add a related Linux distribution, add its `/etc/os-release` ID to an
existing `distro_ids` list. Add a new platform entry only when its paths,
dependencies, or native package format differ. The `rpm` and `deb` values are
reserved for their native package backends. RPM and DEB generation are
implemented.

## Build Debian packages

Install the packages listed by the selected profile, then build a binary DEB:

```bash
python3 packaging/scripts/build-package.py deb --platform ubuntu
# Or, when building on Debian:
python3 packaging/scripts/build-package.py deb --platform debian
```

Artifacts are written to `dist/ubuntu/deb/` or `dist/debian/deb/`. Use
`--without-cuda` for a layout-only package, or `--without-pm-sampling` to keep
CUDA telemetry while omitting privileged PM Sampling.

Direct Meson commands remain supported and default to `/usr/local`.

For the complete Slurm allocation, CUDA module, sudo-free measurement, and
administrator installation workflows on ORFEO, see `packaging/ORFEO.md`.
