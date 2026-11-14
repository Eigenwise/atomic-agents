import errno
from pathlib import Path
from unittest.mock import Mock

import pytest
from textual.app import App

from atomic_assembler.screens.file_explorer import FileExplorerScreen


class PickerApp(App):
    def __init__(self, directory: Path):
        super().__init__()
        self.picker = FileExplorerScreen()
        self.picker.set_reactive(FileExplorerScreen.current_path, directory)

    def on_mount(self):
        self.push_screen(self.picker)


@pytest.mark.asyncio
@pytest.mark.parametrize("error_number", [errno.EACCES, errno.ENOENT, errno.ENOTDIR])
async def test_unreadable_directory_keeps_picker_usable(tmp_path, monkeypatch, error_number):
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (tmp_path / "available.txt").write_text("kept", encoding="utf-8")
    real_iterdir = Path.iterdir

    def guarded_iterdir(path):
        if path == blocked:
            raise OSError(error_number, "cannot list directory", str(path))
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    app = PickerApp(tmp_path)
    async with app.run_test() as pilot:
        picker = app.picker
        notification = Mock(wraps=picker.notify)
        monkeypatch.setattr(picker, "notify", notification)
        await pilot.press("right")

        assert picker.current_path == blocked
        assert picker.file_list.item_list == []
        assert notification.call_args.kwargs["severity"] == "error"
        assert str(blocked) in notification.call_args.args[0]

        await pilot.press("left")
        assert picker.current_path == tmp_path
        assert {item["path"].name for item in picker.file_list.item_list} == {"blocked", "available.txt"}


@pytest.mark.asyncio
async def test_deleted_current_directory_can_return_to_parent(tmp_path, monkeypatch):
    removed = tmp_path / "removed"
    removed.mkdir()
    app = PickerApp(removed)
    async with app.run_test() as pilot:
        picker = app.picker
        notification = Mock(wraps=picker.notify)
        monkeypatch.setattr(picker, "notify", notification)
        removed.rmdir()
        picker.refresh_file_list()

        assert picker.file_list.item_list == []
        assert notification.call_args.kwargs["severity"] == "error"
        await pilot.press("left")
        assert picker.current_path == tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize("folder", [False, True])
@pytest.mark.parametrize("error_number", [errno.EACCES, errno.EROFS])
async def test_creation_failure_restores_input_and_allows_retry(tmp_path, monkeypatch, folder, error_number):
    blocked = tmp_path / "blocked"
    operation = "mkdir" if folder else "touch"
    original = getattr(Path, operation)

    def guarded_create(path, *args, **kwargs):
        if path == blocked:
            raise OSError(error_number, "cannot create item", str(path))
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, operation, guarded_create)
    app = PickerApp(tmp_path)
    async with app.run_test() as pilot:
        picker = app.picker
        notification = Mock(wraps=picker.notify)
        monkeypatch.setattr(picker, "notify", notification)
        key = "n" if folder else "f"
        await pilot.press(key)
        picker.new_item_input.value = "blocked"
        await pilot.press("enter")

        assert not blocked.exists()
        assert notification.call_args.kwargs["severity"] == "error"
        assert str(blocked) in notification.call_args.args[0]
        assert not picker.new_item_mode
        assert not picker.new_item_input.display
        assert not picker.file_list.disabled

        await pilot.press(key)
        picker.new_item_input.value = "created"
        await pilot.press("enter")
        created = tmp_path / "created"
        assert created.is_dir() if folder else created.is_file()
        assert created in [item["path"] for item in picker.file_list.item_list]


@pytest.mark.asyncio
@pytest.mark.parametrize("folder", [False, True])
async def test_existing_item_is_not_overwritten(tmp_path, folder):
    existing = tmp_path / "existing"
    if folder:
        existing.mkdir()
        protected = existing / "keep.txt"
    else:
        protected = existing
    protected.write_text("original", encoding="utf-8")
    app = PickerApp(tmp_path)
    async with app.run_test() as pilot:
        picker = app.picker
        await pilot.press("n" if folder else "f")
        picker.new_item_input.value = "existing"
        await pilot.press("enter")

        assert protected.read_text(encoding="utf-8") == "original"
        assert not picker.new_item_mode
        assert not picker.file_list.disabled


@pytest.mark.asyncio
@pytest.mark.parametrize("folder", [False, True])
async def test_successful_creation_remains_visible(tmp_path, folder):
    app = PickerApp(tmp_path)
    async with app.run_test() as pilot:
        picker = app.picker
        await pilot.press("n" if folder else "f")
        picker.new_item_input.value = "new-item"
        await pilot.press("enter")

        created = tmp_path / "new-item"
        assert created.is_dir() if folder else created.is_file()
        assert created in [item["path"] for item in picker.file_list.item_list]
        assert not picker.new_item_mode
        assert not picker.file_list.disabled
