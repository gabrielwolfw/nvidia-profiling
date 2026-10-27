# GPU analyzers

`energy_attribution/` contains the Python implementation, commands, workflows,
report writers, and regression fixtures. The sibling `analyze_*.py`,
`bayesian_energy_model.py`, and `plot_pm_sampling.py` files are executable
entry points. Run them from any working directory, for example:

```bash
python3 /path/to/nvidia-profiling/analyzers/analyze_bayesian_kernel_energy.py --help
```

Install from the repository root:

```bash
python3 -m pip install -e '.[plots,test]'
```

The root `pyproject.toml` remains the single package configuration. Installed
console commands retain their names. Additional commands expose the PM and
weighted-energy analyzers:

- `nvidia-kernel-pm-metrics`
- `nvidia-kernel-energy-weighted`
- `nvidia-pm-sampling-plot` (requires the `plots` extra)

All analyzer scripts live in this directory. From the repository root, use
`python3 analyzers/<script>.py` or an installed console command. The root
`energy_attribution` directory is a relative symlink to the package here,
preserving imports from source checkouts without changing Python's import path.
Distribute this checkout with symlinks preserved. Wheels contain the actual
package and do not require the compatibility symlink.

Capture filenames, CSV columns, output report filenames, and model calculations
are unchanged. Existing result directories do not need migration.

Run the regression suite from the repository root:

```bash
python3 -m pytest
# Without pytest, the existing unittest suite can also run directly:
python3 -m unittest discover -s analyzers/energy_attribution/tests -t analyzers
```
