import subprocess
from pathlib import Path

import git
import pytest

from atomic_assembler.utils import GithubRepoCloner, AtomicToolManager


def test_custom_branch(tmp_path):
    repository = tmp_path / "remote"
    subprocess.run(["git", "init", "-qb", "custom", repository], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            repository,
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.org",
            "commit",
            "--allow-empty",
            "-qm",
            "sample",
        ],
        check=True,
    )
    cloner = GithubRepoCloner(str(repository), branch="custom")
    try:
        cloner.clone()
        assert (
            subprocess.check_output(["git", "-C", cloner.repo_path, "branch", "--show-current"], text=True).strip() == "custom"
        )
    finally:
        cloner.cleanup()


def test_failed_clone_releases_temp_folder(tmp_path):
    cloner = GithubRepoCloner(str(tmp_path / "missing"))
    try:
        with pytest.raises(git.GitCommandError):
            cloner.clone()
        assert not Path(cloner.temp_dir).exists()
    finally:
        cloner.cleanup()


def test_failed_clone_leaves_an_empty_catalogue(tmp_path):
    assert AtomicToolManager.get_atomic_tools(tmp_path / "no-checkout") == []


def test_copy_failure_can_be_retried(tmp_path, monkeypatch):
    tool = tmp_path / "tool"
    tool.mkdir()
    (tool / "sample").touch()
    destination = tmp_path / "destination"
    destination.mkdir()

    def interrupt_copy(source, target, **kwargs):
        partial = Path(target)
        partial.mkdir(exist_ok=True)
        (partial / "partial").touch()
        raise OSError("copy interrupted")

    with monkeypatch.context() as patcher:
        patcher.setattr("atomic_assembler.utils.shutil.copytree", interrupt_copy)
        with pytest.raises(Exception, match="copy interrupted"):
            AtomicToolManager.copy_atomic_tool(tool, destination)
    assert not (destination / "tool").exists()
    copied_tool = Path(AtomicToolManager.copy_atomic_tool(tool, destination))
    assert (copied_tool / "sample").exists()
    (copied_tool / "keep").touch()
    with pytest.raises(Exception, match="File exists"):
        AtomicToolManager.copy_atomic_tool(tool, destination)
    assert (copied_tool / "keep").exists()
