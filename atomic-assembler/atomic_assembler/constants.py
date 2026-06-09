from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

# Color configuration
PRIMARY_COLOR: str = "#AAAA00"
SECONDARY_COLOR: str = "#AA00AA"
BORDER_STYLE: str = f"bold {SECONDARY_COLOR}"

# Title configuration
TITLE_FONT: str = "big"

# Repository configuration
TOOLS_SUBFOLDER: str = "atomic-forge/tools"

# Base URL for GitHub repository
GITHUB_BASE_URL: str = "https://github.com/eigenwise/atomic-agents.git"
GITHUB_BRANCH: str = "main"


@dataclass
class MenuOption:
    """Dataclass representing a menu option."""

    label: str
    action: str
    params: Optional[Dict[str, Any]] = None


# Define menu options as a list of MenuOption instances
MENU_OPTIONS: List[MenuOption] = [
    MenuOption("Download Tools", "download_tools"),
    MenuOption("Open Atomic Agents on GitHub", "open_github"),
    MenuOption("Quit", "exit"),
]


class Mode(Enum):
    FILE_MODE = "file_mode"
    DIRECTORY_MODE = "directory_mode"
