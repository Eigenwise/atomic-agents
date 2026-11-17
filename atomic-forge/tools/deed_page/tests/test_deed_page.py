import asyncio
import os
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.deed_page import (  # noqa: E402
    DeedPageTool,
    DeedPageToolConfig,
    DeedPageToolInputSchema,
    DeedPageToolOutputSchema,
)


PUBLISHED = {
    "ok": True,
    "url": "https://quiet-fern-abcd.deed.page/",
    "slug": "quiet-fern-abcd",
    "version": "v1",
    "expires_at": None,
    "token": "deed_tok_secret",
}


def _session_class(status: int, payload: dict, reason: str = "OK"):
    """Return a stand-in for aiohttp.ClientSession plus the session it yields."""
    session = MagicMock()
    response = SimpleNamespace(status=status, reason=reason, json=AsyncMock(return_value=payload))
    session.post.return_value.__aenter__.return_value = response
    session.delete.return_value.__aenter__.return_value = response
    cls = MagicMock()
    cls.return_value.__aenter__.return_value = session
    return cls, session


@pytest.fixture(autouse=True)
def no_env_token(monkeypatch):
    monkeypatch.delenv("DEEDPAGE_TOKEN", raising=False)


def test_publish_without_token_keeps_new_token_out_of_output():
    cls, session = _session_class(200, PUBLISHED)
    tool = DeedPageTool()
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        output = tool.run(DeedPageToolInputSchema(files={"index.html": "<h1>hi</h1>"}))

    assert isinstance(output, DeedPageToolOutputSchema)
    assert output.url == "https://quiet-fern-abcd.deed.page/"
    assert output.slug == "quiet-fern-abcd"
    assert output.version == "v1"
    assert "deed_tok_secret" not in output.model_dump_json()
    assert tool.token == "deed_tok_secret"

    headers = cls.call_args.kwargs["headers"]
    assert "Authorization" not in headers
    assert "X-Ttl" not in headers
    assert headers["User-Agent"].startswith("atomic-agents-deed-page-tool/")
    url = session.post.call_args.args[0]
    assert url == "https://deed.page/v1/sites"
    assert session.post.call_args.kwargs["json"] == {"files": {"index.html": "<h1>hi</h1>"}}


def test_publish_with_token_slug_and_ttl():
    cls, session = _session_class(200, {**PUBLISHED, "token": None, "expires_at": "2026-10-27T17:00:00Z"})
    tool = DeedPageTool(DeedPageToolConfig(api_token="deed_tok_existing", base_url="https://example.test/"))
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        output = tool.run(
            DeedPageToolInputSchema(files={"index.html": "<h1>hi</h1>"}, slug="quiet-fern-abcd", ttl_seconds=600)
        )

    headers = cls.call_args.kwargs["headers"]
    assert headers["Authorization"] == "Bearer deed_tok_existing"
    assert headers["X-Ttl"] == "600"
    assert session.post.call_args.args[0] == "https://example.test/v1/sites"
    assert session.post.call_args.kwargs["json"]["slug"] == "quiet-fern-abcd"
    assert output.expires_at == "2026-10-27T17:00:00Z"
    assert tool.token == "deed_tok_existing"


def test_token_from_environment(monkeypatch):
    monkeypatch.setenv("DEEDPAGE_TOKEN", "deed_tok_env")
    assert DeedPageTool().token == "deed_tok_env"


def test_publish_error_raises():
    cls, _ = _session_class(400, {"ok": False, "error": "slug_reserved"}, reason="Bad Request")
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        with pytest.raises(Exception, match="slug_reserved"):
            DeedPageTool().run(DeedPageToolInputSchema(files={"index.html": "x"}, slug="hello"))


def test_empty_files_rejected():
    with pytest.raises(ValueError):
        DeedPageTool().run(DeedPageToolInputSchema(files={}))


@pytest.mark.parametrize("slug", ["", "Bad_Slug", "-leading", "trailing-"])
def test_invalid_slug_rejected(slug):
    with pytest.raises(ValidationError):
        DeedPageToolInputSchema(files={"index.html": "x"}, slug=slug)


def test_delete_uses_token():
    cls, session = _session_class(200, {"ok": True})
    tool = DeedPageTool(DeedPageToolConfig(api_token="deed_tok_existing"))
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        tool.delete("quiet-fern-abcd")
    assert session.delete.call_args.args[0] == "https://deed.page/v1/sites/quiet-fern-abcd"
    assert cls.call_args.kwargs["headers"]["Authorization"] == "Bearer deed_tok_existing"


def test_delete_without_token_raises():
    with pytest.raises(ValueError):
        DeedPageTool().delete("quiet-fern-abcd")


def test_delete_error_raises():
    cls, _ = _session_class(404, {"ok": False}, reason="Not Found")
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        with pytest.raises(Exception, match="404"):
            DeedPageTool(DeedPageToolConfig(api_token="t")).delete("gone-site")


class _FakeResponse:
    def __init__(self, server, headers, body):
        self.status = 200
        self.reason = "OK"
        self._server, self._headers, self._body = server, headers, body

    async def json(self, content_type=None):
        return await self._server.handle(self._headers, self._body)


class _AsyncContext:
    def __init__(self, value):
        self.value = value

    async def __aenter__(self):
        return self.value

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    def __init__(self, server, headers):
        self._server, self._headers = server, headers

    def post(self, url, json):
        return _AsyncContext(_FakeResponse(self._server, self._headers, json))


class FakeDeedPage:
    """Offline stand-in for the publish API that tracks which token owns each site.

    A tokenless publish mints a new owner token; a publish with a bearer token
    attaches the site to that owner. The first request is held open until
    ``release_first`` is set, so a test can overlap a second publish with it.
    """

    def __init__(self):
        self.owners = {}
        self.auth_seen = []
        self.issued = 0
        self.first_in_flight = threading.Event()
        self.release_first = threading.Event()
        self._lock = threading.Lock()

    def session(self, headers=None, timeout=None):
        return _AsyncContext(_FakeSession(self, dict(headers or {})))

    async def handle(self, headers, body):
        auth = headers.get("Authorization")
        with self._lock:
            index = len(self.auth_seen)
            self.auth_seen.append(auth)
        if index == 0:
            self.first_in_flight.set()
            await asyncio.to_thread(self.release_first.wait, 5)
        new_token = None
        with self._lock:
            if auth:
                owner = auth.removeprefix("Bearer ")
            else:
                self.issued += 1
                owner = new_token = f"deed_tok_{self.issued}"
            self.owners[body["slug"]] = owner
        return {
            "ok": True,
            "url": f"https://{body['slug']}.deed.page/",
            "slug": body["slug"],
            "version": "v1",
            "token": new_token,
        }


def _assert_both_pages_manageable(server, tool, results):
    assert set(results) == {"page-a", "page-b"}
    assert set(server.owners) == {"page-a", "page-b"}
    # Both pages belong to the single token the tool holds, so it can update or delete either one.
    assert server.owners["page-a"] == server.owners["page-b"] == tool.token
    assert server.issued == 1
    assert server.auth_seen == [None, f"Bearer {tool.token}"]


def test_concurrent_first_publishes_share_one_owner_token_sync():
    server = FakeDeedPage()
    tool = DeedPageTool()
    results = {}

    def publish(slug):
        results[slug] = tool.run(DeedPageToolInputSchema(files={"index.html": slug}, slug=slug))

    with patch("tool.deed_page.aiohttp.ClientSession", server.session):
        first = threading.Thread(target=publish, args=("page-a",))
        first.start()
        assert server.first_in_flight.wait(5)
        second = threading.Thread(target=publish, args=("page-b",))
        second.start()
        time.sleep(0.2)  # let the second publish send its request if nothing holds it back
        server.release_first.set()
        first.join(10)
        second.join(10)

    _assert_both_pages_manageable(server, tool, results)


def test_concurrent_first_publishes_share_one_owner_token_async():
    server = FakeDeedPage()
    tool = DeedPageTool()

    async def main():
        first = asyncio.create_task(tool.run_async(DeedPageToolInputSchema(files={"index.html": "a"}, slug="page-a")))
        assert await asyncio.to_thread(server.first_in_flight.wait, 5)
        second = asyncio.create_task(tool.run_async(DeedPageToolInputSchema(files={"index.html": "b"}, slug="page-b")))
        await asyncio.sleep(0.2)
        server.release_first.set()
        return dict(zip(["page-a", "page-b"], await asyncio.wait_for(asyncio.gather(first, second), 10)))

    with patch("tool.deed_page.aiohttp.ClientSession", server.session):
        results = asyncio.run(main())

    _assert_both_pages_manageable(server, tool, results)


def test_failed_first_publish_releases_bootstrap():
    cls, _ = _session_class(500, {"ok": False, "error": "boom"}, reason="Server Error")
    tool = DeedPageTool()
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        with pytest.raises(Exception, match="boom"):
            tool.run(DeedPageToolInputSchema(files={"index.html": "x"}))
    cls, _ = _session_class(200, PUBLISHED)
    with patch("tool.deed_page.aiohttp.ClientSession", cls):
        tool.run(DeedPageToolInputSchema(files={"index.html": "x"}))
    assert tool.token == "deed_tok_secret"
