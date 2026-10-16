# Releasing EEGFMLens

Releases are built from a clean version tag. GitHub Actions builds one wheel and
one source archive, runs lint, formatting, mypy, the core test suite with an 80%
coverage floor, audits both archives, and checks their package metadata. The same
audited files become the GitHub release assets and the PyPI upload.

Before the first release, configure a PyPI trusted publisher for this repository,
workflow `release.yml`, and environment `pypi`. Protect that GitHub environment
with the desired reviewer policy. No long-lived PyPI token is required.

The reviewed a25 wheel used for the EEGPT cohort has SHA-256
`b805c6286e35d425167dcb99f99adc93a0fb75454d8e032ee0e6d9e9069a1407`.
Its 40 package Python files are AST-equivalent to this release source; two
files received formatting-only edits to satisfy the release lint gate. Its
embedded README predates the a25 documentation update. Keep that digest as
scientific execution provenance; build a separate release wheel from the final
clean source. Before publishing, verify AST equivalence with the studied wheel,
byte identity with the release source, and metadata description equality with
this README.

1. Update the version in `pyproject.toml` and add a matching changelog heading.
2. Run the local release gates:

   ```bash
   ruff check .
   ruff format --check .
   mypy
   pytest -m 'not integration and not native' --cov=eegfmlens --cov-fail-under=80
   python -m build
   python tools/check_release.py --tag v0.1.0a25 --dist dist
   twine check dist/*
   ```

3. Review the two files in `dist/` and the changelog. Create and push the exact
   version tag only after that review:

   ```bash
   git tag -s v0.1.0a25
   git push origin v0.1.0a25
   ```

The tag triggers publication. A tag that differs from the package version fails
before any artifact is uploaded. Delete neither a published PyPI version nor its
Git tag; issue a new version for corrections.
