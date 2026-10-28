# Releasing EEGFMLens

The release workflow builds a wheel and source archive from a version tag, runs
lint, formatting, type checks and the offline test suite, checks package metadata,
and uploads the verified distributions to GitHub Releases and PyPI. Native model
checkpoints are supplied separately and are not bundled in a release.

Before publishing, configure a PyPI trusted publisher for workflow `release.yml`
and environment `pypi`. Protect that environment with the project's required
review policy. The workflow uses short-lived trusted-publisher credentials.

1. Update the version in `pyproject.toml` and its changelog entry.
2. In a clean checkout with development dependencies installed, run:

   ```sh
   ruff check .
   ruff format --check .
   mypy
   pytest -m 'not integration and not native' --cov=eegfmlens --cov-fail-under=80
   python -m build
   python tools/check_release.py --tag v0.1.0a30 --dist dist
   twine check dist/*
   ```

   Replace the example tag with the exact version being released. Check the
   wheel in an isolated environment with `tools/check_installed.py`; verify
   the examples as well as imports and tests.
3. Review the distribution contents, dependency metadata, documentation and
   license inventory. The wheel and source archive must correspond to the
   reviewed source, with no recordings, model weights or machine-specific files.
4. When the release is approved, create and push its signed version tag. A pushed
   `v*` tag triggers the publication workflow; local builds do not publish.

The tag must match the package version. Corrections to a published version require
a new version rather than replacement of the existing distribution or tag.
Experiments that used an earlier version retain their original execution
provenance; a new software release does not change those measurements.
