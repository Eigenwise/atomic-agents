import logging
import os
import shutil
import tempfile

import git
import yaml
from pathlib import Path
from atomic_assembler.constants import TOOLS_SUBFOLDER


class GithubRepoCloner:
    def __init__(self, base_url: str, branch: str = "main"):
        self.branch = branch
        self.repo_url = base_url
        self.temp_dir = tempfile.mkdtemp()
        self.repo_path = os.path.join(self.temp_dir, os.path.basename(base_url).replace(".git", ""))
        self.tools_path = os.path.join(self.repo_path, TOOLS_SUBFOLDER)

    def clone(self):
        try:
            git.Repo.clone_from(self.repo_url, self.repo_path, branch=self.branch)
            logging.info(f"Repository cloned to {self.repo_path} on branch {self.branch}")
        except git.GitCommandError as e:
            self.cleanup()
            logging.error(f"Failed to clone repository: {e}")
            raise

    def cleanup(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class AtomicToolManager:
    @staticmethod
    def read_tool_config(tool_path):
        config_path = os.path.join(tool_path, "config.yaml")
        try:
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            return None
        except Exception as e:
            return f"Error reading config file: {e}"

    @staticmethod
    def get_atomic_tools(tools_path: str) -> list[dict]:
        """
        Get a list of atomic tools from the given tools path.

        Args:
            tools_path (str): The path to the directory containing atomic tools.

        Returns:
            list[dict]: A list of dictionaries containing tool information.
        """
        tools = []
        try:
            entries = os.listdir(tools_path)
        except FileNotFoundError:
            return tools
        for item in entries:
            item_path = os.path.join(tools_path, item)
            if os.path.isdir(item_path):
                # Convert snake_case to Title Case
                tool_name = " ".join(word.capitalize() for word in item.split("_"))
                tools.append(
                    {
                        "name": tool_name,
                        "path": item_path,
                    }
                )
        return tools

    @staticmethod
    def copy_atomic_tool(tool_path, destination):
        logging.info(f"copy_atomic_tool called with tool_path: {tool_path}, destination: {destination}")
        try:
            tool_name = os.path.basename(os.path.abspath(tool_path))
            local_tool_path = os.path.join(destination, tool_name)
            logging.info(f"Copying tool from {tool_path} to {local_tool_path}")

            if not os.path.exists(tool_path):
                logging.error(f"Source path does not exist: {tool_path}")
                raise FileNotFoundError(f"Source path does not exist: {tool_path}")

            if not os.path.exists(destination):
                logging.error(f"Destination path does not exist: {destination}")
                raise FileNotFoundError(f"Destination path does not exist: {destination}")

            # Reserve the destination so a failed copy can remove just the files it created.
            os.mkdir(local_tool_path)
            try:
                shutil.copytree(
                    tool_path,
                    local_tool_path,
                    dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("requirements.txt", "pyproject.toml", ".coveragerc", "uv.lock"),
                )
            except Exception:
                shutil.rmtree(local_tool_path)
                raise
            logging.info(f"Tool successfully copied to {local_tool_path}")
            return local_tool_path
        except Exception as e:
            logging.error(f"Error copying tool: {str(e)}", exc_info=True)
            raise Exception(f"Error copying tool: {e}")

    @staticmethod
    def load_env_file(file_path: Path) -> dict:
        """Load environment variables from a .env file."""
        env_vars = {}
        if file_path.exists():
            with open(file_path, "r") as file:
                for line in file:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, value = line.split("=", 1)
                        env_vars[key.strip()] = value.strip()
        return env_vars

    @staticmethod
    def read_readme(tool_path: str) -> str:
        """
        Read the README.md file from the tool directory.

        Args:
            tool_path (str): The path to the tool directory.

        Returns:
            str: The contents of the README.md file, or an error message if not found.
        """
        readme_path = os.path.join(tool_path, "README.md")
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            return "README.md not found for this tool."
        except Exception as e:
            return f"Error reading README.md: {str(e)}"
