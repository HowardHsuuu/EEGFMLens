# Contributing

Use Python 3.10+ and a virtual environment:

```bash
python -m pip install -e '.[dev,labram,eeg]'
ruff check .
ruff format --check .
pytest -m 'not integration'
python examples/quickstart.py
python -m build
```

Default tests never download data/weights. Checkpoint tests skip unless local paths are supplied; see [validation](docs/validation.md). Report platform, PyTorch version, checkpoint hash and exact commands with failures.

CI builds the sdist/wheel, installs that wheel, and runs `tools/check_installed.py` with Python isolated mode. The checker verifies the installed distribution's wheel digest and every package Python file, then runs non-integration tests, quickstart and the 24-effect analytic patching-sweep example in a temporary directory outside the checkout. The report retains both example source hashes; the sweep checks every expected local effect and requires all 24 records. It writes a report only after all checks pass and refuses to overwrite an existing report. This prevents an editable/source import or stale artifact from masquerading as an installed-wheel success.

To reproduce after installing a built wheel into a separate environment:

```bash
python -I tools/check_installed.py --dist dist --output installed-wheel-report.json
```

The environment needs `pytest` and the `labram` extra. Add `--coverage` with `pytest-cov` installed to match CI. The dist directory must contain exactly one wheel. CI is configured for Linux/macOS/Windows and Python 3.10/3.12; configuration alone is not evidence that those jobs have passed. Remote execution has not been claimed by the local release audit.

Each new writable site needs native/identity parity, actual hook hits, meaningful downstream effects, selection semantics and cleanup tests. Registering a module name is insufficient. Include negative tests for unsupported operations. Distinguish random fixtures, official weights and real EEG evidence.

Keep research alignment and preprocessing outside the hook engine. Functional instrumentation requires separate parity evidence. Vendor changes must explain deviations, pin revisions and retain licenses; Ruff excludes vendor code to keep upstream diffs small.

Do not commit weights, recordings, caches, credentials or machine-specific paths. Generate synthetic fixtures in tests. Respect public artifact licenses and avoid untested scientific claims.
