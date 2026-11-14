import logging
from rich.text import Text
from textual.app import ComposeResult
from textual.widgets import Static, Footer
from textual.binding import Binding
from textual import on
from textual.reactive import reactive
from textual.screen import Screen
from pathlib import Path
from typing import List, Optional, Dict, Callable

from atomic_assembler.constants import BORDER_STYLE, PRIMARY_COLOR, Mode
from atomic_assembler.widgets.generic_list import GenericList
from atomic_assembler.widgets.new_item_input import NewItemInput
from atomic_assembler.widgets.confirmation_modal import ConfirmationModal

from textual.message import Message


class FileExplorerScreen(Screen):
    """Screen for exploring files and directories."""

    class FileSelected(Message):
        """Message emitted when a file is selected."""

        def __init__(self, path: Path) -> None:
            self.path = path
            super().__init__()

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

    #current-path {{
        padding: 1 2;
    }}

    Footer {{
        color: $text;
    }}

    .modal-overlay {{
        width: 100%;
        height: 100%;
        background: rgba(0, 0, 0, 0.5);
        layout: grid;
        grid-size: 1;
        align: center middle;
    }}
    """

    BINDINGS = [
        Binding("escape", "handle_escape", "Exit/Cancel", show=True),
        Binding("n", "new_folder", "New Folder"),
        Binding("f", "new_file", "New File"),
        Binding("left", "go_up_folder", "Go Up", show=True, priority=True),
        Binding("right", "enter_folder", "Enter Folder", show=True, priority=True),
        Binding("enter", "action_select", "Select Item", show=True),
    ]

    current_path: reactive[Path] = reactive(Path.cwd())
    new_item_mode: reactive[bool] = reactive(False)
    selected_file: Optional[Path] = None
    directory_selections: Dict[Path, Optional[Path]] = {}

    def __init__(
        self,
        allowed_extensions: Optional[List[str]] = None,
        enable_folder_creation: bool = True,
        enable_file_creation: bool = True,
        mode: Mode = Mode.FILE_MODE,
        callback: Optional[Callable[[Path], None]] = None,
        title: str = "File Picker",  # New title parameter
        *args,
        **kwargs,
    ):
        """Initialize the FileExplorerScreen."""
        super().__init__(*args, **kwargs)
        self.mode = mode
        self.callback = callback
        self.title_widget = Static(title, id="title")  # Use the title parameter
        self.current_path_widget = Static("", id="current-path")
        self.file_list = GenericList(item_renderer=self._render_file_item)
        self.new_item_input = NewItemInput(id="new-item-input")
        self.footer = Footer()
        self.allowed_extensions = allowed_extensions
        self.enable_folder_creation = enable_folder_creation
        self.enable_file_creation = enable_file_creation
        logging.info("FileExplorerScreen initialized")

    def compose(self) -> ComposeResult:
        """Compose the layout of the screen."""
        yield self.title_widget
        yield self.current_path_widget
        yield self.file_list
        yield self.new_item_input
        yield self.footer

    def on_mount(self):
        """Handler called when the screen is mounted."""
        logging.info("FileExplorerScreen mounted")
        self.refresh_file_list()
        self.border_color = BORDER_STYLE.split()[-1]

    def watch_current_path(self, path: Path):
        """React to changes in the current_path."""
        logging.info(f"Current path changed to: {path}")
        self.refresh_file_list()

    def refresh_file_list(self):
        """Refresh the file list and update the current path display."""
        logging.debug("Refreshing file list")
        self.update_current_path_display()
        items = self._get_file_items()
        self.file_list.update_list(items)

        # Set highlighted index based on stored selection for current directory
        selected_path = self.directory_selections.get(self.current_path)
        if selected_path:
            for index, item in enumerate(items):
                if item["path"] == selected_path:
                    self.file_list.set_highlighted_index(index)
                    break
            else:
                self.file_list.set_highlighted_index(0)
        else:
            self.file_list.set_highlighted_index(0)

    def update_current_path_display(self):
        """Update the display of the current path."""
        self.current_path_widget.update(
            Text.assemble("Current directory: ", (str(self.current_path), f"bold {PRIMARY_COLOR}"))
        )

    @on(GenericList.Highlighted)
    def on_highlighted(self, list_view):
        logging.info(f"Highlighted item: {list_view.item}")
        if list_view.item:
            # self.current_path = str(list_view.highlighted_item.item_data["path"])
            self.current_path_widget.update(
                Text.assemble("Current directory: ", (str(list_view.item.item_data["path"]), f"bold {PRIMARY_COLOR}"))
            )

    def _get_file_items(self):
        """Get the list of file items to display."""
        items = []
        for item in sorted(self.current_path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower())):
            if self.mode == Mode.DIRECTORY_MODE and item.is_file():
                continue  # Skip files in directory mode
            if self._is_allowed_file(item):
                items.append({"path": item, "is_dir": item.is_dir(), "is_parent": False})
        return items

    def _is_allowed_file(self, path: Path) -> bool:
        """Check if the file is allowed based on its extension or name."""
        if path.is_dir():
            return True
        if self.allowed_extensions is None:
            return True

        # Convert allowed extensions to lowercase for case-insensitive comparison
        allowed_exts = [ext.lower() for ext in self.allowed_extensions]

        # Check if the file name (without extension) is in allowed extensions
        # This handles cases like '.env'
        if path.stem.lower() in allowed_exts:
            return True

        # Check if the file extension (including the dot) is in allowed extensions
        if path.suffix.lower() in [f".{ext}" for ext in allowed_exts]:
            return True

        return False

    def _render_file_item(self, item: dict) -> str:
        """Render a file item."""
        icon = "📁 " if item["is_dir"] else "📄 "
        name = ".." if item["is_parent"] else item["path"].name
        return f"{icon}{name}"

    def on_generic_list_item_selected(self, message: GenericList.ItemSelected):
        """Handle item selection from GenericList."""
        item = message.item
        self.handle_item_selection(item)

    def handle_item_selection(self, item: dict):
        """Handle the selection of an item based on the current mode."""
        if (self.mode == Mode.DIRECTORY_MODE and item["is_dir"]) or (self.mode == Mode.FILE_MODE and not item["is_dir"]):
            self.selected_file = item["path"]
            item_type = "folder" if item["is_dir"] else "file"
            logging.info(f"{item_type.capitalize()} selected: {self.selected_file}")
            self.app.push_screen(
                ConfirmationModal(
                    f"Are you sure you want to select this {item_type}: {self.selected_file.name}?",
                    self.handle_confirmation,
                )
            )
        else:
            logging.info("No valid selection made.")

    def action_enter_folder(self):
        """Action to enter the selected folder."""
        highlighted_item = self.file_list.highlighted_child
        if not highlighted_item:
            return
        selected_item = highlighted_item.item_data
        if selected_item and selected_item["is_dir"]:
            # Store the current selection before changing directory
            self.directory_selections[self.current_path] = selected_item["path"]
            self.current_path = selected_item["path"]
            logging.info(f"Entered directory: {self.current_path}")

    def action_select(self):
        """Override the select action to handle Enter key press."""
        highlighted_item = self.file_list.highlighted_child
        if highlighted_item:
            self.handle_item_selection(highlighted_item.item_data)

    def handle_confirmation(self, confirmed: bool):
        """Handle the result of the confirmation modal."""
        logging.info(f"Confirmation result: {confirmed}")
        if confirmed and self.selected_file:
            logging.info(f"Selection confirmed: {self.selected_file}")
            if self.callback:
                logging.info(f"Calling callback with selected file: {self.selected_file}")
                self.app.pop_screen()  # Pop the screen after callback

                self.callback(self.selected_file)  # Ensure this is called
        else:
            logging.info("Selection cancelled")
            self.selected_file = None

    def action_new_folder(self):
        """Action to enter new folder creation mode."""
        logging.info("Entering new folder creation mode")
        self.new_item_mode = True
        self.new_item_input.display = True
        self.file_list.disabled = True
        self.new_item_input.placeholder = "Enter folder name"
        self.new_item_input.focus()

    def action_new_file(self):
        """Action to enter new file creation mode."""
        logging.info("Entering new file creation mode")
        self.new_item_mode = True
        self.new_item_input.display = True
        self.file_list.disabled = True
        self.new_item_input.placeholder = "Enter file name"
        self.new_item_input.focus()

    def create_new_item(self, item_name: str):
        """Create a new item based on the input."""
        logging.info(f"Attempting to create new item: {item_name}")
        if item_name:
            new_item_path = self.current_path / item_name
            try:
                if self.new_item_input.placeholder == "Enter folder name":
                    new_item_path.mkdir(parents=True, exist_ok=False)
                    logging.info(f"New directory created: {new_item_path}")
                else:
                    new_item_path.touch(exist_ok=False)
                    logging.info(f"New file created: {new_item_path}")
                self.refresh_file_list()
            except FileExistsError:
                logging.warning(f"Failed to create item, already exists: {new_item_path}")
                self.bell()
            finally:
                self.new_item_input.value = ""
        self.exit_new_item_mode()

    def exit_new_item_mode(self):
        """Exit the new item creation mode."""
        logging.info("Exiting new item mode")
        self.new_item_mode = False
        self.new_item_input.display = False
        self.file_list.disabled = False
        self.file_list.focus()
        self.refresh_bindings()

    def on_new_item_input_submitted(self, message: NewItemInput.Submitted):
        """Handle the submission of the new item input."""
        item_name = message.value
        self.create_new_item(item_name)

    def on_new_item_input_cancelled(self, message: NewItemInput.Cancelled):
        """Handle the cancellation of the new item input."""
        self.exit_new_item_mode()

    def action_handle_escape(self):
        """Handle the escape key."""
        if self.new_item_mode:
            logging.info("Exiting new item mode via escape key")
            self.exit_new_item_mode()
        else:
            logging.info("Popping screen via escape key")
            self.app.pop_screen()

    def action_go_up_folder(self):
        """Action to go up one folder."""
        if self.current_path != self.current_path.root:
            # Store the current selection before going up
            highlighted_item = self.file_list.highlighted_child
            if highlighted_item:
                self.directory_selections[self.current_path] = highlighted_item.item_data["path"]
            self.current_path = self.current_path.parent
            logging.info(f"Moved up to directory: {self.current_path}")

    def check_action(self, action: str, parameters: tuple[object, ...]) -> bool | None:
        """
        Check if an action may run / be displayed in the footer.
        """
        can_run = None

        if action == "new_folder":
            can_run = self.enable_folder_creation and not self.new_item_mode
        elif action == "new_file":
            can_run = self.enable_file_creation and not self.new_item_mode and self.mode == Mode.FILE_MODE  # Check mode
        elif action == "handle_escape":
            can_run = True
        elif action in ["go_up_folder", "enter_folder"]:
            can_run = not self.new_item_mode

        return can_run  # Return the final value
