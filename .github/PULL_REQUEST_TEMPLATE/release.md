---
name: Release request
about: PR template for publishing a new release to PyPI
title: "[RELEASE] - vX.Y.Z"
labels: release
assignees: ''

---

## Release Summary

Briefly describe what's included in this release:

---

## Pre-Release Checklist

### Local Testing

Please confirm the following tests pass **locally** before requesting review:

- [ ] All unit tests pass (`pytest tests/`)
- [ ] All integration tests pass (`pytest tests/ -m integration`)
- [ ] All slow/extended tests pass (`pytest tests/ -m slow`)

### Examples & Notebooks

- [ ] All example notebooks have been tested and run without errors
- [ ] All package examples execute correctly

### Version & Dependencies

- [ ] Version number has been updated in `pyproject.toml` (or equivalent)
- [ ] Version number has been updated in `__init__.py` (if applicable)
- [ ] All dependencies are correct

### Documentation

- [ ] README reflects any new features or breaking changes
- [ ] API documentation is up to date (if applicable)

---

## CI Verification

The following will be verified by GitHub Actions:

- [ ] Unit tests pass in CI
- [ ] Integration tests pass
- [ ] Extended tests pass

---

## Release Process

After approval, follow the release process documented in [RELEASE.md](../../RELEASE.md) (or your project's release documentation):

1. [ ] Merge this PR to main
2. [ ] Create a GitHub release with tag `vX.Y.Z`
3. [ ] Publish to TestPyPI first and verify installation
4. [ ] Publish to PyPI
5. [ ] Verify installation from PyPI (`pip install scorebook==X.Y.Z`)

---

## Approvals Required

This release requires approval from at least one maintainer before merging.

---

## Additional Notes

Any extra context about this release (breaking changes, migration notes, known issues, etc.).
