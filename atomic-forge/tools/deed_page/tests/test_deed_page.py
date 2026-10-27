import os
import sys
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


@pytest.mark.parametrize("slug", ["Bad_Slug", "-leading", "trailing-"])
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
