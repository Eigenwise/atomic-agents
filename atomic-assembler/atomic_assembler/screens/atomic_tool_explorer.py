from textual.app import ComposeResult
from textual.widgets import Static, Footer
from textual.binding import Binding
from textual.screen import Screen
from textual.message import Message
from textual import on, work
from pathlib import Path
import logging
import threading

from atomic_assembler.constants import BORDER_STYLE, GITHUB_BRANCH, PRIMARY_COLOR, Mode, GITHUB_BASE_URL
from atomic_assembler.screens.file_explorer import FileExplorerScreen
from atomic_assembler.utils import AtomicToolManager, GithubRepoCloner
from atomic_assembler.widgets.generic_list import GenericList
from atomic_assembler.screens.tool_info_screen import ToolInfoScreen
from atomic_assembler.widgets.confirmation_modal import ConfirmationModal


class AtomicToolExplorerScreen(Screen):
    """Screen for exploring and downloading atomic tools."""

    CSS = f"""
    Screen {{
        align: center middle;
    }}

    #title {{
        dock: top;
        padding: 1 2;
        color: $text;
        text-align: center;
        text-style: bold;
        border: solid {PRIMARY_COLOR};
    }}

    #current-tool {{
        padding: 1 2;
    }}

    Footer {{
        color: $text;
    }}
    """

    BINDINGS = [
        Binding("escape", "app.pop_screen", "Exit", show=True),
        Binding("i", "show_tool_info", "Tool Info"),
    ]

    class ToolSelected(Message):
        """Message emitted when a tool is selected."""

        def __init__(self, tool_info: dict) -> None:
            self.tool_info = tool_info
            super().__init__()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._initialize_components()
        self.github_repo_cloner = GithubRepoCloner(GITHUB_BASE_URL, branch=GITHUB_BRANCH)
        self._catalog_ready = False
        self._catalog_closed = threading.Event()
        self._checkout_started = threading.Event()
        self._checkout_finished = threading.Event()

    def _initialize_components(self):
        self.title_widget = Static("Atomic Tool Explorer", id="title")
        self.current_tool_widget = Static("", id="current-tool")
        self.tool_list = GenericList(item_renderer=self._render_tool_item)
        self.footer = Footer()
        self.atomic_tool_manager = AtomicToolManager()
        self.current_tool = None
        self.highlighted_tool = None

    class CatalogLoaded(Message):
        """Carry a completed checkout back to the screen's message loop."""

        def __init__(self, tools: list[dict], error: str | None = None) -> None:
            self.tools = tools
            self.error = error
            super().__init__()

    def on_mount(self) -> None:
        self.tool_list.loading = True
        self._load_catalog()

    @work(thread=True)
    def _load_catalog(self) -> None:
        self._checkout_started.set()
        try:
            if self._catalog_closed.is_set():
                return
            self.github_repo_cloner.clone()
            tools = self.atomic_tool_manager.get_atomic_tools(self.github_repo_cloner.tools_path)
            self.post_message(self.CatalogLoaded(tools))
        except Exception as error:
            logging.exception("Failed to load the tool catalogue")
            self.post_message(self.CatalogLoaded([], str(error)))
        finally:
            # A cancelled worker's Git process can still be writing. Cleanup
            # waits for checkout to finish, even if the screen has already left.
            self._checkout_finished.set()
            if self._catalog_closed.is_set():
                self.github_repo_cloner.cleanup()

    def on_atomic_tool_explorer_screen_catalog_loaded(self, event: CatalogLoaded) -> None:
        if self._catalog_closed.is_set():
            return
        self.tool_list.loading = False
        if event.error is not None:
            self.notify(f"Failed to load tool catalogue: {event.error}", severity="error", markup=False)
            return
        self._catalog_ready = True
        self.tool_list.update_list(event.tools)

    def compose(self) -> ComposeResult:
        yield self.title_widget
        yield self.current_tool_widget
        yield self.tool_list
        yield self.footer

    def on_screen_resume(self) -> None:
        self.refresh_tool_list()
        self.border_color = BORDER_STYLE.split()[-1]

    def on_unmount(self):
        self._catalog_closed.set()
        if not self._checkout_started.is_set() or self._checkout_finished.is_set():
            self.github_repo_cloner.cleanup()

    def refresh_tool_list(self):
        if not self._catalog_ready:
            return
        tools = self.atomic_tool_manager.get_atomic_tools(self.github_repo_cloner.tools_path)
        self.tool_list.update_list(tools)

    def _render_tool_item(self, tool: dict) -> str:
        return f"🔧 {tool['name']}"

    @on(GenericList.Highlighted)
    def handle_tool_highlighted(self, event: GenericList.Highlighted) -> None:
        self.highlighted_tool = event.item

    @on(GenericList.ItemSelected)
    def handle_tool_selected(self, event: GenericList.ItemSelected):
        self.current_tool = event.item
        logging.info(f"Tool selected: {self.current_tool['name']}")
        self.post_message(self.ToolSelected(self.current_tool))
        self._open_file_explorer_for_directory()

    def _open_file_explorer_for_directory(self):
        logging.info("Opening FileExplorerScreen in directory mode")
        self.app.push_screen(FileExplorerScreen(mode=Mode.DIRECTORY_MODE, callback=self.handle_directory_selection))

    def handle_directory_selection(self, selected_dir: Path):
        logging.info(f"Directory selected: {selected_dir}")
        if self.current_tool and selected_dir:
            self._copy_tool_to_directory(selected_dir)
        else:
            logging.warning("No tool selected or no directory chosen")
            self.notify("No tool selected or no directory chosen")

    def _copy_tool_to_directory(self, selected_dir: Path):
        try:
            local_tool_path = self.atomic_tool_manager.copy_atomic_tool(self.current_tool["path"], selected_dir)
            logging.info(f"Tool successfully copied to {local_tool_path}")
            modal = ConfirmationModal(
                f"Tool copied to {local_tool_path}. Press any key to continue.",
                callback=lambda _: None,
                mode="continue",
            )
            self.app.push_screen(modal)
        except Exception as e:
            logging.error(f"Error copying tool: {str(e)}", exc_info=True)
            self.notify(f"Error copying tool: {str(e)}")

    def action_show_tool_info(self):
        if self.highlighted_tool:
            tool_data = self.highlighted_tool.item_data
            readme_content = self.atomic_tool_manager.read_readme(tool_data["path"])
            self.app.push_screen(ToolInfoScreen(tool_data["name"], readme_content))
        else:
            self.notify("No tool highlighted.", title="Warning")

    def update_current_tool(self, tool: dict):
        if tool:
            self.current_tool_widget.update(f"Current tool: [bold {PRIMARY_COLOR}]{tool['name']}[/bold {PRIMARY_COLOR}]")
        else:
            self.current_tool_widget.update("")

    def on_key(self, event):
        if event.key == "escape":
            self.app.pop_screen()
