"""End-to-end regression tests for the Bayesian kernel-energy commands.

Each scenario runs a CLI script on deterministic synthetic sessions and
compares every CSV and JSON output with the golden copies in ``golden/``:
column names and order must match exactly, numbers within 1e-9 relative.

After an intended behaviour change, regenerate the golden files with:

    UPDATE_GOLDEN=1 python3 -m pytest \
        analyzers/energy_attribution/tests/test_regression.py
"""

from argparse import Namespace
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from energy_attribution.tests.synthetic import (  # pylint: disable=wrong-import-position
    EXTERNAL_SESSION,
    write_fixture,
)
from energy_attribution import (  # pylint: disable=wrong-import-position
    CrossGpuConfig,
    MultiSessionConfig,
    SingleSessionConfig,
    analyze_single_session,
    run_single_session,
)
from energy_attribution.cli import (  # pylint: disable=wrong-import-position
    config_from_namespace,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ANALYZERS_ROOT = REPO_ROOT / "analyzers"
GOLDEN_ROOT = Path(__file__).resolve().parent / "golden"
UPDATE_GOLDEN = os.environ.get("UPDATE_GOLDEN") == "1"
RELATIVE_TOLERANCE = 1e-9
ABSOLUTE_TOLERANCE = 1e-9
# Summary keys holding temporary paths, which differ between runs.
PATH_KEYS = {"sessions_root", "training_sessions_root", "external_test_session"}


class ConfigurationTests(unittest.TestCase):
    """Exercise the typed public configuration boundary."""

    def test_namespace_conversion_preserves_typed_paths(self):
        namespace = Namespace(
            telemetry=Path("telemetry.csv"),
            kernels=Path("kernels.csv"),
            output_dir=Path("output"),
            window_ms=250.0,
            idle_power_mw=1_500.0,
            max_families=8,
            train_fraction=0.75,
            prior_scale_w=100.0,
            noise_prior_scale_w=10.0,
            burn_in=20,
            posterior_draws=100,
            thin=1,
            seed=7,
            attribution="model",
            min_windows_per_parameter=2.0,
            allow_underdetermined=False,
            min_window_fraction=0.9,
            plot=False,
            skip_event_output=True,
        )
        config = config_from_namespace(SingleSessionConfig, namespace)
        config.validate()
        self.assertEqual(config.window_ns, 250_000_000)
        self.assertEqual(config.telemetry, Path("telemetry.csv"))
        self.assertTrue(config.skip_event_output)

    def test_workflow_specific_validation(self):
        with self.assertRaisesRegex(ValueError, "train_fraction"):
            MultiSessionConfig(
                sessions_root=Path("sessions"),
                output_dir=Path("output"),
                train_fraction=1.0,
            ).validate()
        with self.assertRaisesRegex(ValueError, "calibration_fraction"):
            CrossGpuConfig(
                train_sessions_root=Path("train"),
                external_test_session=Path("test"),
                output_dir=Path("output"),
                calibration_fraction=0.9,
            ).validate()


def scenarios(fixture):
    single = fixture / "train" / "s3-mixed"
    session_files = [
        "--telemetry",
        str(single / "gpu_telemetry.csv"),
        "--kernels",
        str(single / "kernel_activity.csv"),
    ]
    return {
        "single": [
            "analyze_bayesian_kernel_energy.py",
            *session_files,
            "--max-families",
            "5",
            "--plot",
        ],
        "single_legacy": [
            "analyze_bayesian_kernel_energy.py",
            *session_files,
            "--max-families",
            "5",
            "--attribution",
            "active-fraction",
            "--min-window-fraction",
            "0",
            "--skip-event-output",
        ],
        "multisession": [
            "analyze_bayesian_kernel_energy_multisession.py",
            "--sessions-root",
            str(fixture / "train"),
            "--max-families",
            "5",
        ],
        "transfer": [
            "analyze_bayesian_kernel_energy_transfer.py",
            "--train-sessions-root",
            str(fixture / "train"),
            "--external-test-session",
            str(fixture / "external" / EXTERNAL_SESSION[0]),
            "--max-families",
            "5",
        ],
        "telemetry_single": [
            "bayesian_energy_model.py",
            *session_files,
            "--plot",
        ],
        "telemetry_multisession": [
            "analyze_bayesian_energy.py",
            "--sessions-root",
            str(fixture / "train"),
            "--plot",
        ],
    }


def as_number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def values_match(expected, actual):
    expected_number, actual_number = as_number(expected), as_number(actual)
    if expected_number is None or actual_number is None:
        return expected == actual
    if math.isnan(expected_number) or math.isnan(actual_number):
        return math.isnan(expected_number) and math.isnan(actual_number)
    return math.isclose(
        expected_number,
        actual_number,
        rel_tol=RELATIVE_TOLERANCE,
        abs_tol=ABSOLUTE_TOLERANCE,
    )


def read_csv_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def compare_json(expected, actual, location="$"):
    """Yield human-readable differences between two JSON documents."""
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(expected) != set(actual):
            yield (
                f"{location}: keys differ "
                f"{sorted(set(expected) ^ set(actual))}"
            )
        for key in set(expected) & set(actual):
            if key not in PATH_KEYS:
                yield from compare_json(
                    expected[key], actual[key], f"{location}.{key}"
                )
    elif isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            yield f"{location}: length {len(expected)} != {len(actual)}"
        for index, (left, right) in enumerate(zip(expected, actual)):
            yield from compare_json(left, right, f"{location}[{index}]")
    elif (
        isinstance(expected, bool)
        or expected is None
        or isinstance(expected, str)
    ):
        if expected != actual:
            yield f"{location}: {expected!r} != {actual!r}"
    elif not values_match(expected, actual):
        yield f"{location}: {expected!r} != {actual!r}"


class WorkflowRegressionTests(unittest.TestCase):
    """Compare current command outputs with stored golden outputs."""

    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        root = Path(cls.temporary.name)
        fixture = write_fixture(root / "fixture")
        cls.fixture = fixture
        cls.root = root
        cls.outputs = {}
        for name, command in scenarios(fixture).items():
            output = root / "outputs" / name
            subprocess.run(
                [
                    sys.executable,
                    str(ANALYZERS_ROOT / command[0]),
                    *command[1:],
                    "--output-dir",
                    str(output),
                ],
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
            )
            cls.outputs[name] = output
            if UPDATE_GOLDEN:
                golden = GOLDEN_ROOT / name
                shutil.rmtree(golden, ignore_errors=True)
                golden.mkdir(parents=True)
                for path in output.iterdir():
                    if path.suffix in (".csv", ".json"):
                        shutil.copy(path, golden / path.name)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def assert_matches_golden(self, name):
        golden = GOLDEN_ROOT / name
        output = self.outputs[name]
        expected_files = sorted(
            path.name
            for path in golden.iterdir()
            if path.suffix in (".csv", ".json")
        )
        actual_files = sorted(
            path.name
            for path in output.iterdir()
            if path.suffix in (".csv", ".json")
        )
        self.assertEqual(
            expected_files, actual_files, f"{name}: output files differ"
        )
        for file_name in expected_files:
            with self.subTest(scenario=name, file=file_name):
                if file_name.endswith(".json"):
                    differences = list(
                        compare_json(
                            json.loads((golden / file_name).read_text()),
                            json.loads((output / file_name).read_text()),
                        )
                    )
                    self.assertFalse(differences, "\n".join(differences[:10]))
                    continue
                expected = read_csv_rows(golden / file_name)
                actual = read_csv_rows(output / file_name)
                self.assertEqual(expected[0], actual[0], "CSV columns differ")
                self.assertEqual(
                    len(expected), len(actual), "CSV row counts differ"
                )
                mismatches = [
                    f"row {row}, {expected[0][column]}: {left!r} != {right!r}"
                    for row, (expected_row, actual_row) in enumerate(
                        zip(expected[1:], actual[1:]), start=1
                    )
                    for column, (left, right) in enumerate(
                        zip(expected_row, actual_row)
                    )
                    if not values_match(left, right)
                ]
                self.assertFalse(mismatches, "\n".join(mismatches[:10]))

    def test_single_session(self):
        self.assert_matches_golden("single")
        for plot in (
            "bayesian_kernel_family_energy.png",
            "bayesian_kernel_power_fit.png",
            "bayesian_kernel_energy_test_scatter.png",
            "bayesian_session_energy.png",
        ):
            self.assertTrue((self.outputs["single"] / plot).is_file(), plot)

    def test_single_session_legacy_attribution(self):
        self.assert_matches_golden("single_legacy")

    def test_multisession(self):
        self.assert_matches_golden("multisession")

    def test_transfer(self):
        self.assert_matches_golden("transfer")

    def test_telemetry_single_session(self):
        self.assert_matches_golden("telemetry_single")

    def test_telemetry_multisession(self):
        self.assert_matches_golden("telemetry_multisession")

    def api_config(self, name, **overrides):
        session = self.fixture / "train" / "s3-mixed"
        settings = {
            "telemetry": session / "gpu_telemetry.csv",
            "kernels": session / "kernel_activity.csv",
            "output_dir": self.root / "outputs" / name,
            "max_families": 5,
            "posterior_draws": 100,
            "burn_in": 50,
            "thin": 1,
            "skip_event_output": True,
        }
        settings.update(overrides)
        return SingleSessionConfig(**settings)

    def test_public_api_returns_typed_result(self):
        """The supported library boundary works without importing a CLI."""
        config = self.api_config("public-api")
        result = run_single_session(config)
        self.assertEqual(result.summary["kernel_families"], 5)
        self.assertGreater(result.summary["test_windows"], 0)
        self.assertEqual(len(result.family_rows), 5)
        self.assertTrue(
            (
                config.output_dir / "bayesian_kernel_energy_summary.json"
            ).is_file()
        )

    def test_analyze_writes_nothing(self):
        config = self.api_config("analyze-only", skip_event_output=False)
        result = analyze_single_session(config)
        self.assertFalse(config.output_dir.exists())
        self.assertIsNotNone(result.events)
        # Events conserve the energy attributed to families.
        self.assertAlmostEqual(
            float(result.events.mean.sum()),
            result.accounting["attributed_posterior_mean_mj"],
            delta=1e-6 * result.accounting["attributed_posterior_mean_mj"],
        )


if __name__ == "__main__":
    unittest.main()
