"""The tool catalogue loads without holding Textual's event loop."""

import asyncio
import threading
from pathlib import Path

import git
import pytest
from textual.app import App
from textual.screen import Screen
from textual.widgets import Static
from textual.worker import Worker

from atomic_assembler.screens import atomic_tool_explorer as explorer
from atomic_assembler.screens.tool_info_screen import ToolInfoScreen


class CatalogApp(App):
    def __init__(self):
        super().__init__()
        self.catalog = None

    async def on_mount(self):
        self.catalog = explorer.AtomicToolExplorerScreen()
        await self.push_screen(self.catalog)


@pytest.fixture
def repository(tmp_path, monkeypatch):
    path = tmp_path / "remote"
    repo = git.Repo.init(path, initial_branch="main")
    tools = path / "atomic-forge" / "tools"
    for name in ("sample_tool", "second_tool"):
        folder = tools / name
        folder.mkdir(parents=True)
        (folder / "README.md").write_text(f"# {name}\nAvailable tool.\n", encoding="utf-8")
    repo.index.add([str(p.relative_to(path)) for p in path.rglob("README.md")])
    repo.index.commit(
        "fixture", author=git.Actor("Fixture", "fixture@example.test"), committer=git.Actor("Fixture", "fixture@example.test")
    )
    repo.close()
    monkeypatch.setattr(explorer, "GITHUB_BASE_URL", str(path))
    return path


async def wait_for(predicate):
    async def poll():
        while not predicate():
            await asyncio.sleep(0.01)

    await asyncio.wait_for(poll(), timeout=5)


@pytest.fixture
def blocked_clone(monkeypatch):
    original = git.Repo.clone_from
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    threads = []

    def clone(*args, **kwargs):
        threads.append(threading.get_ident())
        started.set()
        try:
            # A bounded guard lets the unchanged synchronous implementation fail
            # the responsiveness assertions instead of hanging the test runner.
            if not release.wait(timeout=2):
                release.set()
            return original(*args, **kwargs)
        finally:
            finished.set()

    monkeypatch.setattr(git.Repo, "clone_from", clone)
    yield started, release, finished, threads
    release.set()
    if started.is_set():
        assert finished.wait(timeout=5)


def test_construction_does_not_start_git(repository, monkeypatch):
    calls = []
    monkeypatch.setattr(git.Repo, "clone_from", lambda *args, **kwargs: calls.append(args))
    screen = explorer.AtomicToolExplorerScreen()
    try:
        assert calls == []
    finally:
        screen.github_repo_cloner.cleanup()


@pytest.mark.asyncio
async def test_real_checkout_runs_off_loop_and_pop_remains_responsive(repository, blocked_clone):
    started, release, finished, threads = blocked_clone
    app = CatalogApp()
    async with app.run_test() as pilot:
        await wait_for(started.is_set)
        screen = app.catalog
        assert len(threads) == 1 and threads[0] != threading.get_ident()
        assert screen.tool_list.loading
        assert not release.is_set()
        temp = Path(screen.github_repo_cloner.temp_dir)
        await pilot.press("escape")
        assert app.screen is not screen
        assert not release.is_set()
        assert temp.exists(), "do not remove a checkout while Git is writing"
        release.set()
        await wait_for(finished.is_set)
        await wait_for(lambda: not temp.exists())
        assert app.screen is not screen


@pytest.mark.asyncio
async def test_app_shutdown_defers_cleanup_until_checkout_finishes(repository, blocked_clone):
    started, release, finished, _ = blocked_clone
    app = CatalogApp()
    async with app.run_test():
        await wait_for(started.is_set)
        assert not release.is_set()
        temp = Path(app.catalog.github_repo_cloner.temp_dir)
    assert not release.is_set(), "closing the application must not join a blocked Git process"
    assert temp.exists()
    release.set()
    await wait_for(finished.is_set)
    await wait_for(lambda: not temp.exists())


@pytest.mark.asyncio
async def test_loaded_tools_readme_resume_and_cleanup(repository, monkeypatch):
    original = git.Repo.clone_from
    calls = []

    def clone(*args, **kwargs):
        calls.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(git.Repo, "clone_from", clone)
    app = CatalogApp()
    async with app.run_test() as pilot:
        screen = app.catalog
        await wait_for(lambda: len(screen.tool_list.item_list) == 2)
        await pilot.pause()
        assert not screen.tool_list.loading
        assert {tool["name"] for tool in screen.tool_list.item_list} == {"Sample Tool", "Second Tool"}
        screen.tool_list.focus()
        screen.tool_list.index = 0
        await pilot.pause()
        await pilot.press("i")
        assert isinstance(app.screen, ToolInfoScreen)
        assert "Available tool." in app.screen.readme_content
        await pilot.press("escape")
        assert app.screen is screen
        assert len(screen.tool_list.item_list) == 2
        assert len(calls) == 1
        temp = Path(screen.github_repo_cloner.temp_dir)
        await pilot.press("escape")
        await wait_for(lambda: not temp.exists())


@pytest.mark.asyncio
async def test_late_result_does_not_replace_a_covering_screen(repository, blocked_clone):
    started, release, finished, _ = blocked_clone
    app = CatalogApp()
    async with app.run_test() as pilot:
        await wait_for(started.is_set)
        assert not release.is_set()
        covering = Screen(Static("Another screen"))
        await app.push_screen(covering)
        release.set()
        await wait_for(finished.is_set)
        await wait_for(lambda: len(app.catalog.tool_list.item_list) == 2)
        assert app.screen is covering
        await app.pop_screen()
        await pilot.pause()
        assert app.screen is app.catalog
        assert len(app.catalog.tool_list.item_list) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["checkout", "catalogue"])
async def test_loading_failure_keeps_screen_usable(repository, monkeypatch, failure):
    if failure == "checkout":
        monkeypatch.setattr(explorer, "GITHUB_BASE_URL", str(repository / "missing"))
    else:

        def fail(_):
            raise PermissionError("[restricted] catalogue")

        monkeypatch.setattr(explorer.AtomicToolManager, "get_atomic_tools", staticmethod(fail))
    notifications = []
    monkeypatch.setattr(
        explorer.AtomicToolExplorerScreen, "notify", lambda self, text, **kwargs: notifications.append((text, kwargs))
    )
    app = CatalogApp()
    async with app.run_test() as pilot:
        await wait_for(lambda: bool(notifications))
        screen = app.catalog
        assert not screen.tool_list.loading
        assert screen.tool_list.item_list == []
        assert notifications[0][1]["severity"] == "error"
        assert notifications[0][1]["markup"] is False
        temp = Path(screen.github_repo_cloner.temp_dir)
        await pilot.press("escape")
        assert app.screen is not screen
        await wait_for(lambda: not temp.exists())


@pytest.mark.asyncio
async def test_pop_before_worker_starts_releases_unused_checkout(repository, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()
    original = Worker._run_threaded
    calls = []
    original_clone = git.Repo.clone_from

    def clone(*args, **kwargs):
        calls.append(args)
        return original_clone(*args, **kwargs)

    async def delay(worker):
        entered.set()
        await release.wait()
        return await original(worker)

    monkeypatch.setattr(Worker, "_run_threaded", delay)
    monkeypatch.setattr(git.Repo, "clone_from", clone)
    app = CatalogApp()
    async with app.run_test() as pilot:
        await asyncio.wait_for(entered.wait(), timeout=5)
        temp = Path(app.catalog.github_repo_cloner.temp_dir)
        await pilot.press("escape")
        assert not temp.exists()
        release.set()
        await pilot.pause()
        assert calls == []
