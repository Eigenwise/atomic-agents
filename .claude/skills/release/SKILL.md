---
name: release
description: Prepare atomic-agents versions and releases. A version bump ends at a local diff. Use the publication steps only for a maintainer's explicit request to publish.
allowed-tools: Read, Edit, Bash, Grep, Glob
metadata:
  internal: true
---

# Release atomic-agents

## Choose the stopping point

| Request | Stop after |
| --- | --- |
| Bump or set the version | Show the local version and lockfile diff. Commit only if requested. |
| Prepare or build a release | Show the local wheel and sdist with their SHA-256 hashes. |
| Release, publish, or deploy | Prepare the requested targets, show the publication plan, and wait for explicit approval of those actions. |
| Continue or retry | Recover the exact version, commit and artifacts from the previous attempt. Check what already exists. |

The maintainer's `build_and_deploy.ps1` command combines a bump, build and PyPI upload. Keep it out of every path in this skill, including retries. Stop on the first failed command.

## Set the version locally

1. From the repository root inspect the branch, `git status --short`, the version diff (including staged changes), and `uv version --short`. Start a new full release from clean, up-to-date `main`. Preserve unrelated work. For a continuation, use the existing prepared version even if the version files are already committed.
2. For a new bump, record the absolute target once. Increment the requested major, minor or patch component, zeroing the components to its right. If the request could refer to a previously prepared version and the target is unclear, resolve that before editing.
3. Set just two values to that target: `[project].version` in the root `pyproject.toml` and the version of the `atomic-agents` package with `source = { editable = "." }` in `uv.lock`. This preserves the other workspace packages and the existing lockfile format. If either value is already correct, leave it in place.
4. Run `uv lock --check`, inspect `git diff HEAD -- pyproject.toml uv.lock`, and verify both values. **End a version-only request here.** Leave credentials, builds, pushes, tags and releases alone.

## Build locally

For a new release use the target set above. For a prepared version, start here. Run `uv sync --locked`, then `uv build --out-dir "$artifacts"` from the root, where `$artifacts` is a newly created, empty directory outside the checkout. Keep the original files through verification and retries. Never clear a previous attempt's directory.

Check that the build produced exactly the expected wheel and source archive. Verify the package name and target version in the wheel's METADATA and the sdist's PKG-INFO. Record each full path and SHA-256. **End a prepare-only request here.**

## Confirm publication

For publication, commit the reviewed version and release notes locally as needed. Build the final artifacts from the clean release commit. Reuse artifacts already built from that exact clean commit. Record the full commit SHA and review the changes since the previous tag.

Before any external write, show the maintainer:

- Exact package/version, PyPI upload endpoint, and both artifact paths and hashes.
- GitHub repository, branch or PR push if needed, tag, target commit, release title and actual notes.
- Which actions remain for each requested destination after the read-only checks below. Omit destinations outside the request.

Wait for explicit approval of that plan. Silence, a declined plan, or an ambiguous reply leaves all external state untouched. Approval can cover both destinations together or just one. Retain explicit approval for the same plan on continuation. If the target, artifacts, commit, notes or scope change, show the revised plan for approval.

## Perform the approved actions

1. Verify the clean source commit and artifact hashes again. If pushing the release commit is approved, use the normal branch/PR process for `Eigenwise/atomic-agents`. Verify the exact source commit on remote main. If merging changed the commit used for the build, prepare and review the final artifacts at the merged commit before upload.
2. For approved PyPI publication, supply credentials only to the upload process (the existing `PYPI_TOKEN` can be mapped to `UV_PUBLISH_TOKEN` without printing it). Upload the verified missing files by exact path. For two missing files the command is `uv publish --publish-url https://upload.pypi.org/legacy/ "$wheel" "$sdist"`. Verify the remote SHA-256 for each file.
3. For approved GitHub publication, create the missing tag at the approved commit with a normal, non-forced push to the verified repository. Verify the remote tag's peeled commit, then run `gh release create "$tag" --repo Eigenwise/atomic-agents --verify-tag --title "$tag" --notes-file "$notes"` if the release is missing. Verify the public release URL and notes. Report each destination separately.

## Reconcile a partial attempt

Before publishing, after any failure or timeout, and on retry, inspect local state and read the applicable remote state:

- **PyPI** — Compare the filenames and `digests.sha256` at `https://pypi.org/pypi/atomic-agents/<version>/json` with the saved files. Skip matching files. A version-specific 404 or a successful response missing an expected filename identifies a missing file. It can be uploaded from the original build under the approved plan. Stop on an unexpected file, hash mismatch, or inconclusive response.
- **GitHub** — Distinguish a confirmed missing ref or release from network and authentication errors. Check the remote branch, the tag (including the peeled target for an annotated tag), and the release by tag. Skip matching completed steps. A matching tag with no release needs only release creation. Stop on a conflicting commit, draft, or unexpected release contents.
- Preserve the version diff, commits, notes and artifacts. Resume the first missing authorized action. Never reset hard, force-move a tag, increment the version again, or rebuild over a partially published version. If original bytes are lost, report what is live and obtain a recovery decision.

[Non-publishing regression scenarios](regression.md)
