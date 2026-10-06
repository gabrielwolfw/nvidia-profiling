#!/usr/bin/env python3
"""Validate every platform entry and print its resolved configuration."""

from platform_config import load_registry, resolve_platform


def main() -> None:
    registry = load_registry()
    for name in sorted(registry["platforms"]):
        _, config = resolve_platform(name)
        print(f"{name}: {config['package_format']} {config['prefix']}/{config['libdir']}")


if __name__ == "__main__":
    main()
