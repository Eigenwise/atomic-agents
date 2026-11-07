import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from atomic_assembler.utils import GithubRepoCloner, AtomicToolManager


class AssemblerTests(unittest.TestCase):
    def setUp(self):
        t = tempfile.TemporaryDirectory()
        self.addCleanup(t.cleanup)
        self.tmp = Path(t.name)

    def test_custom_branch(self):
        repo = self.tmp / "remote"
        subprocess.run(["git", "init", "-qb", "custom", repo], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                repo,
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
        cloner = GithubRepoCloner(str(repo), branch="custom")
        self.addCleanup(cloner.cleanup)
        cloner.clone()
        self.assertEqual(
            subprocess.check_output(["git", "-C", cloner.repo_path, "branch", "--show-current"], text=True).strip(), "custom"
        )

    def test_failed_clone_releases_temp_folder(self):
        cloner = GithubRepoCloner(str(self.tmp / "missing"))
        self.addCleanup(cloner.cleanup)
        with self.assertRaises(Exception):
            cloner.clone()
        self.assertFalse(Path(cloner.temp_dir).exists())

    def test_failed_clone_leaves_an_empty_catalogue(self):
        self.assertEqual(AtomicToolManager.get_atomic_tools(self.tmp / "no-checkout"), [])

    def test_copy_trailing_separator(self):
        tool = self.tmp / "tool"
        tool.mkdir()
        (tool / "sample").touch()
        dest = self.tmp / "dest"
        dest.mkdir()
        out = AtomicToolManager.copy_atomic_tool(str(tool) + "/", str(dest))
        self.assertTrue((Path(out) / "sample").exists())

    def test_copy_failure_can_be_retried(self):
        tool = self.tmp / "tool"
        tool.mkdir()
        (tool / "sample").touch()
        dest = self.tmp / "dest"
        dest.mkdir()

        def fail(src, dst, **kwargs):
            partial = Path(dst)
            partial.mkdir(exist_ok=True)
            (partial / "partial").touch()
            raise OSError("copy interrupted")

        with patch("shutil.copytree", side_effect=fail), self.assertRaises(Exception):
            AtomicToolManager.copy_atomic_tool(tool, dest)
        self.assertFalse((dest / "tool").exists())
        out = Path(AtomicToolManager.copy_atomic_tool(tool, dest))
        self.assertTrue((out / "sample").exists())
        (out / "keep").touch()
        with self.assertRaises(Exception):
            AtomicToolManager.copy_atomic_tool(tool, dest)
        self.assertTrue((out / "keep").exists())


if __name__ == "__main__":
    unittest.main()
