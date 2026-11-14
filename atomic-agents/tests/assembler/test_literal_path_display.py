from pathlib import Path

import pytest
from textual.app import App
from textual.color import Color
from textual.widgets import Static

from atomic_assembler.constants import Mode, PRIMARY_COLOR
from atomic_assembler.screens.file_explorer import FileExplorerScreen
from atomic_assembler.widgets.confirmation_modal import ConfirmationModal


class LiteralPickerApp(App):
    def __init__(self, directory: Path, *, mode=Mode.FILE_MODE):
        super().__init__()
        self.selections = []
        self.picker = FileExplorerScreen(mode=mode, callback=self.selections.append)
        self.picker.set_reactive(FileExplorerScreen.current_path, directory)

    def on_mount(self):
        self.push_screen(self.picker)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["[red]notes", "[b]workspace", "[link=example]project", "plain"])
async def test_directory_navigation_displays_literal_path(tmp_path, name):
    directory = tmp_path / name
    directory.mkdir()
    app = LiteralPickerApp(tmp_path)
    async with app.run_test(size=(200, 30)) as pilot:
        picker = app.picker
        await pilot.press("right")
        assert picker.current_path == directory
        rendered = picker.current_path_widget.render()
        assert rendered.plain == f"Current directory: {directory}"
        style = rendered.get_style_at_offset(len("Current directory: "))
        assert style.bold and style.foreground == Color.parse(PRIMARY_COLOR)
        assert style.link is None
        await pilot.press("left")
        assert picker.current_path == tmp_path
        # The selected child path is also shown literally when highlighted.
        assert picker.current_path_widget.render().plain == f"Current directory: {directory}"


@pytest.mark.asyncio
@pytest.mark.parametrize("folder", [False, True])
@pytest.mark.parametrize("name", ["report [red].md", "[b]important", "plain"])
async def test_selection_confirmation_preserves_literal_name(tmp_path, name, folder):
    selected = tmp_path / name
    if folder:
        selected.mkdir()
    else:
        selected.write_text("content", encoding="utf-8")
    app = LiteralPickerApp(tmp_path, mode=Mode.DIRECTORY_MODE if folder else Mode.FILE_MODE)
    async with app.run_test(size=(200, 30)) as pilot:
        await pilot.press("enter")
        assert isinstance(app.screen, ConfirmationModal)
        item_type = "folder" if folder else "file"
        shown = app.screen.query_one("#modal-content", Static).render().plain
        assert shown == f"Are you sure you want to select this {item_type}: {name}?"
        await pilot.press("y")
        assert app.selections == [selected]
        assert app.screen is not app.picker


@pytest.mark.asyncio
async def test_download_receipt_preserves_literal_destination(tmp_path):
    destination = tmp_path / "[red]tools"
    message = f"Tool copied to {destination}. Press any key to continue."
    app = App()
    confirmations = []
    async with app.run_test(size=(200, 30)) as pilot:
        await app.push_screen(ConfirmationModal(message, confirmations.append, mode="continue"))
        assert app.screen.query_one("#modal-content", Static).render().plain == message
        await pilot.press("enter")
        assert confirmations == [True]
