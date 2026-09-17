# Contributing

Use Python 3.10+ in a virtual environment:

```bash
python -m pip install -e '.[dev]'
ruff check .
ruff format --check .
pytest -m 'not integration and not native'
python examples/quickstart.py
python -m build
```

The core suite uses synthetic fixtures and never downloads models or data.
[Native tests](validation/README.md) additionally use explicit pinned upstream
checkouts; checkpoint tests also require local weights. These tests remain
separate so the core package can be tested without upstream dependencies.

CI checks formatting with the same pinned Ruff version as the dev extra, builds
and tests an installed wheel on Linux/macOS/Windows with Python 3.10/3.12, and
runs a separate pinned-source native job. Check the actual workflow result for a
commit before attributing platform coverage to that commit. Weights are not
silently downloaded in CI.

To verify a built wheel in a separate environment with pytest installed:

```bash
python -m pip install /path/to/eeglens-0.1.0a10-py3-none-any.whl pytest
python -I tools/check_installed.py --dist /path/to/dist --output /path/to/new-report.json
```

The checker validates installed package bytes and dependency consistency, then
runs the core tests and both offline examples outside the checkout. The dist
folder must contain exactly one wheel; output reports must not already exist.

Each writable site needs native/identity parity, actual hook hits, meaningful
intervention effects, selection semantics and cleanup tests. Include rejection
cases for unsupported operations. Distinguish synthetic contracts from official
checkpoint and real-data evidence.

Keep native model implementations, dataset-specific analysis and preprocessing
out of the runtime. Supply external constructors to checkpoint helpers. Do not
commit weights, recordings, credentials, caches or personal machine paths.
Historical evidence in `validation/archive/` is preserved; do not relabel it as a
run of current code. New reports belong to CI artifacts or a clearly scoped
integration result.
