---
title: Configure Renovate bot policies
description: Add Renovate configuration for dependency, pre-commit, and GitHub Actions updates.
date: 2026-01-26
---

## Goal
Configure Renovate to manage updates with patch-only core deps, full updates for dependency-groups, and latest pre-commit/GitHub Actions.

## References
- `renovate.json`
- `pyproject.toml`
- `.pre-commit-config.yaml`
- `.github/workflows` (if present)

## Design
- Add a `renovate.json` configuration that:
  - Enables `pep621`, `pre-commit`, and `github-actions` managers.
  - Keeps core/optional dependencies patch-only.
  - Allows dependency-groups (dev/test/docs) to update to latest.
  - Pins GitHub Actions to digests with readable versions.
  - Groups updates to reduce PR noise.
  - Splits PRs into groups:
    - Pre-commit hooks (latest) in one PR.
    - GitHub Actions (latest, digest-pinned) in one PR.
    - Dependency-groups (dev/test/docs, all updates) in one PR.
    - Core/optional dependencies (patch-only) in one PR.

## How to test
- Renovate will validate configuration on first run.
