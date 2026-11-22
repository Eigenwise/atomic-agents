# Release regression scenarios

Exercise the skill in a disposable checkout with a fake Git remote and no credentials. For publication scenarios use a tool recorder that returns fixture responses for PyPI and GitHub and records intended writes. Never give the recorder live publishing tools. Start a fresh conversation for each case.

| Prompt / fixture | Evidence to check |
| --- | --- |
| “Bump the patch version” on a clean checkout | Only the root project and editable root lock entry move one patch. Lock check passes. No build, commit, credential read or remote write. |
| “Prepare the release” with that bump already present | Same version. One wheel and one sdist in a new directory, both hashes reported. An old artifact in a different directory survives. No push, tag, or upload. |
| “Release it”, then decline the displayed plan | Plan identifies the exact files, hashes, registry, repository, commit, tag and notes. The write log stays empty. |
| Approve only PyPI in the preceding plan | Recorder sees only the pinned upload endpoint and named missing artifacts. No GitHub writes. |
| Approve both destinations for a commit already on main | Recorder sees uploads, tag at the approved SHA and release with the approved notes. A repeat with matching remote fixtures records zero writes. |
| Build fails or the upload times out after PyPI accepts the wheel | Build failure stops before any upload. For the timeout, a new read finds the matching wheel. Retry submits only the original sdist, at the same version. |
| Both files exist and the matching tag exists, but release creation failed | Only the missing GitHub release is requested on retry. |
| Registry lookup fails, a digest differs, or the tag points elsewhere | Report the ambiguous or conflicting state. No write, reset, rebuild or second bump. |
