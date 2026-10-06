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

Use `--without-cuda` for layout validation on machines without the NVIDIA CUDA
toolkit. This does not produce a functional profiler installation.

To add a related Linux distribution, add its `/etc/os-release` ID to an
existing `distro_ids` list. Add a new platform entry only when its paths,
dependencies, or native package format differ. The `rpm` and `deb` values are
reserved for their native package backends; this initial entry point builds
and stages the common Meson payload used by those backends.

Direct Meson commands remain supported and default to `/usr/local`.
