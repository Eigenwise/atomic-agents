---
name: release
description: Release a new version of atomic-agents to PyPI and GitHub. Use when the user asks to "release", "publish", "deploy", or "bump version" for atomic-agents.
allowed-tools: Read, Bash, Grep, Glob
metadata:
  internal: true
---

# Release Process for atomic-agents

## Overview

This skill guides the release process for atomic-agents, including version bumping, PyPI publishing, and GitHub release creation.

## Prerequisites

- Must be on `main` branch with clean working directory
- `.env` file must contain `PYPI_TOKEN` environment variable
- Must have push access to the repository

## Release Types

| Type | When to Use | Example |
|------|-------------|---------|
| `major` | Breaking changes | 2.5.0 → 3.0.0 |
| `minor` | New features (backwards compatible) | 2.5.0 → 2.6.0 |
| `patch` | Bug fixes only | 2.5.0 → 2.5.1 |

## Step-by-Step Process

### 1. Prepare the Branch

```bash
git checkout main
git pull
git status  # Ensure clean working directory
```

### 2. Run Build and Deploy Script

**Important**: The script bumps versions, so if it fails partway through, reset to main before retrying.

```powershell
powershell -ExecutionPolicy Bypass -File build_and_deploy.ps1 <major|minor|patch>
```

This script will:
- Read current version from `pyproject.toml`
- Increment version based on release type
- Update `pyproject.toml` with new version
- Run `uv sync` to update dependencies
- Run `uv build` to create distribution packages
- Run `uv publish` to upload to PyPI

### 3. If Script Fails - Reset and Retry

```bash
git checkout main
git reset --hard origin/main
# Fix the issue, then run script again
```

### 4. Commit and Push Version Bump

```bash
git add pyproject.toml uv.lock
git commit -m "Release vX.Y.Z"
git push
```

### 5. Create GitHub Release

```bash
gh release create vX.Y.Z --title "vX.Y.Z" --notes "RELEASE_NOTES_HERE"
```

## Release Notes Template

```markdown
## What's New

### Feature Name
Brief description of the feature.

#### Features
- Feature 1
- Feature 2

#### Improvements
- Improvement 1

#### Bug Fixes
- Fix 1

### Full Changelog
https://github.com/eigenwise/atomic-agents/compare/vOLD...vNEW
```

## Checklist

- [ ] On main branch with clean working directory
- [ ] `.env` file has `PYPI_TOKEN`
- [ ] Determined correct release type (major/minor/patch)
- [ ] Build and deploy script completed successfully
- [ ] Version bump committed and pushed
- [ ] GitHub release created with release notes
