"""Guard the package's structural contracts: CLI/config parity and layering."""

import argparse
import ast
import sys
import unittest
from dataclasses import MISSING, fields
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from energy_attribution import config as config_module
from energy_attribution.allocation import posterior_energy_allocation
from energy_attribution.cli import add_config_arguments, config_from_namespace

PACKAGE = Path(__file__).resolve().parents[1]
ANALYZERS_ROOT = PACKAGE.parent
CONFIG_TYPES = [
    config_module.SingleSessionConfig,
    config_module.MultiSessionConfig,
    config_module.CrossGpuConfig,
    config_module.TelemetrySessionConfig,
    config_module.TelemetryMultiSessionConfig,
]
# Modules that make up the numerical core; they must not depend on the
# orchestration, presentation or command-line layers.
UPPER_LAYERS = ("workflows", "reporting", "commands", "cli", "api")


def required_arguments(config_type):
    argv = []
    for field in fields(config_type):
        if field.default is MISSING:
            argv += ["--" + field.name.replace("_", "-"), "placeholder"]
    return argv


def package_imports(path):
    imports = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
        elif isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
    return imports


class CommandLineParityTests(unittest.TestCase):
    """Command-line parsing matches the configuration dataclasses."""

    def test_parsed_defaults_equal_dataclass_defaults(self):
        for config_type in CONFIG_TYPES:
            with self.subTest(config=config_type.__name__):
                parser = argparse.ArgumentParser()
                add_config_arguments(parser, config_type)
                namespace = parser.parse_args(required_arguments(config_type))
                parsed = config_from_namespace(config_type, namespace)
                direct = config_type(
                    **{
                        field.name: getattr(parsed, field.name)
                        for field in fields(config_type)
                        if field.default is MISSING
                    }
                )
                self.assertEqual(parsed, direct)

    def test_namespace_must_match_fields_exactly(self):
        namespace = argparse.Namespace(telemetry=Path("t"), kernels=Path("k"))
        with self.assertRaisesRegex(ValueError, "missing"):
            config_from_namespace(config_module.SingleSessionConfig, namespace)


class LayeringTests(unittest.TestCase):
    """Core modules do not depend on upper layers."""

    def test_core_modules_do_not_import_upper_layers(self):
        for path in PACKAGE.glob("*.py"):
            if path.stem in UPPER_LAYERS or path.stem == "__init__":
                continue
            for module in package_imports(path):
                with self.subTest(module=path.stem, imports=module):
                    layer = module.removeprefix("energy_attribution.").split(
                        "."
                    )[0]
                    self.assertNotIn(layer, UPPER_LAYERS)

    def test_package_never_imports_repository_scripts(self):
        scripts = {path.stem for path in ANALYZERS_ROOT.glob("*.py")}
        for path in PACKAGE.rglob("*.py"):
            self.assertFalse(package_imports(path) & scripts, path)

    def test_analyzer_scripts_only_delegate_to_the_package(self):
        scripts = {path.stem for path in ANALYZERS_ROOT.glob("*.py")}
        self.assertTrue(scripts)
        for path in ANALYZERS_ROOT.glob("*.py"):
            self.assertFalse(package_imports(path) & scripts, path)


class AllocationTests(unittest.TestCase):
    """Energy allocation conserves window energy."""

    def test_family_and_baseline_shares_conserve_window_energy(self):
        rng = np.random.default_rng(5)
        draws = rng.uniform(0.0, 50.0, size=(200, 4))
        activity_ns = rng.uniform(0.0, 2e8, size=(30, 3))
        activity_ns[3] = 0.0
        pool = rng.uniform(1_000.0, 9_000.0, size=30)
        durations = np.full(30, 2e8)
        for mode in config_module.ATTRIBUTION_MODES:
            with self.subTest(mode=mode):
                allocation = posterior_energy_allocation(
                    draws, activity_ns, pool, durations, mode
                )
                np.testing.assert_allclose(
                    allocation.mean.sum(axis=1) + allocation.baseline_mean, pool
                )
                self.assertEqual(allocation.mean[3].sum(), 0.0)


if __name__ == "__main__":
    unittest.main()
