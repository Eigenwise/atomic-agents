import subprocess
from pathlib import Path

import pytest
from git.exc import GitCommandError

from atomic_assembler.utils import AtomicToolManager, GithubRepoCloner


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


@pytest.fixture
def catalogue(tmp_path):
    repository = tmp_path / "catalogue"
    subprocess.run(["git", "init", "-qb", "main", str(repository)], check=True)
    git(repository, "config", "user.name", "Catalogue test")
    git(repository, "config", "user.email", "catalogue@example.test")
    tool = repository / "atomic-forge/tools/example"
    tool.mkdir(parents=True)
    source = tool / "example.py"
    obsolete = repository / "obsolete.txt"
    obsolete.write_text("Historical content not needed to install today's tools.\n")
    source.write_text("VERSION = 0\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "initial catalogue")
    old_blob = git(repository, "rev-parse", "HEAD:obsolete.txt")
    obsolete.unlink()
    for version in range(1, 12):
        source.write_text(f"VERSION = {version}\n")
        git(repository, "add", "-A")
        git(repository, "commit", "-qm", f"catalogue version {version}")
    git(repository, "checkout", "-qb", "preview")
    source.write_text("VERSION = 'preview'\n")
    git(repository, "add", ".")
    git(repository, "commit", "-qm", "preview catalogue")
    git(repository, "checkout", "-q", "main")
    return repository, old_blob


@pytest.mark.parametrize("branch", ["main", "preview"])
def test_catalogue_fetches_only_selected_tip_without_losing_installable_files(catalogue, tmp_path, branch):
    repository, old_blob = catalogue
    # file:// uses the Git transport, so depth is exercised rather than ignored by a local copy.
    cloner = GithubRepoCloner(repository.as_uri(), branch=branch)
    try:
        cloner.clone()
        checkout = Path(cloner.repo_path)
        assert git(checkout, "rev-parse", "HEAD") == git(repository, "rev-parse", branch)
        assert git(checkout, "rev-list", "--count", "HEAD") == "1"
        assert git(checkout, "rev-parse", "--is-shallow-repository") == "true"
        remote_refs = git(checkout, "for-each-ref", "--format=%(refname)", "refs/remotes/origin").splitlines()
        assert [ref for ref in remote_refs if not ref.endswith("/HEAD")] == [f"refs/remotes/origin/{branch}"]
        assert subprocess.run(["git", "-C", str(checkout), "cat-file", "-e", old_blob], capture_output=True).returncode != 0
        tools = AtomicToolManager.get_atomic_tools(cloner.tools_path)
        assert len(tools) == 1
        destination = tmp_path / "installed"
        destination.mkdir()
        installed = Path(AtomicToolManager.copy_atomic_tool(tools[0]["path"], destination))
        expected = "VERSION = 11\n" if branch == "main" else "VERSION = 'preview'\n"
        assert (installed / "example.py").read_text() == expected
    finally:
        cloner.cleanup()
    assert not Path(cloner.temp_dir).exists()


def test_missing_branch_still_cleans_the_temporary_checkout(catalogue):
    repository, _ = catalogue
    cloner = GithubRepoCloner(repository.as_uri(), branch="does-not-exist")
    with pytest.raises(GitCommandError):
        cloner.clone()
    assert not Path(cloner.temp_dir).exists()
