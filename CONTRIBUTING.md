# Contributing

Use Python 3.10+ in a virtual environment:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
mypy
pytest -m 'not integration and not native'
python examples/quickstart.py
python examples/model_catalog.py
python examples/restoration_workflow.py --demo --output /tmp/eegfmlens-restoration
python -m build
python tools/check_release.py --dist dist
```

Run `pre-commit install` once to apply the same Ruff formatting/lint and mypy
checks before each commit.

The core suite uses synthetic fixtures and never downloads models or data.
[Native tests](validation/README.md) additionally use explicit pinned upstream
checkouts; checkpoint tests also require local weights. These tests remain
separate so the core package can be tested without upstream dependencies.

CI checks formatting and source type consistency, builds
and tests an installed wheel on Linux/macOS/Windows with Python 3.10/3.12, and
runs a pinned-source native matrix. The installed suite exercises the
adapter contracts and catalog for all eleven public families. The native-source
jobs check current external-source execution for seven families. The
checkpoint-backed scope is summarized separately and is not presented as a
fresh CI run.
Check the actual workflow result for a commit before attributing platform
coverage to that commit. Weights are not silently downloaded in CI.

To verify a built wheel in a separate environment with pytest installed:

```bash
python -m pip install /path/to/eegfmlens-0.1.0a11-py3-none-any.whl pytest
python -I tools/check_installed.py --dist /path/to/dist --output /path/to/new-report.json
```

The checker validates installed package bytes and dependency consistency, then
runs the core tests, quickstarts, known-answer sweep, and restoration workflow
outside the checkout. The dist
folder must contain exactly one wheel; output reports must not already exist.

Each writable site needs native/identity parity, actual hook hits, meaningful
intervention effects, selection semantics and cleanup tests. Include rejection
cases for unsupported operations. Distinguish synthetic contracts from official
checkpoint and real-data evidence.

Keep native model implementations, dataset-specific analysis and preprocessing
out of the runtime. Supply external constructors to checkpoint helpers. Do not
commit weights, recordings, credentials, caches or personal machine paths.
Generated reports belong in CI artifacts or a separate, clearly scoped
integration result. Keep this repository focused on maintained package code,
portable tests and concise documentation of the supported scope.
