import gzip
import io
import os
import sys
from unittest.mock import patch

import pytest
import requests
from urllib3.response import HTTPResponse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.webpage_scraper import WebpageScraperTool, WebpageScraperToolConfig  # noqa: E402


class ObservedBody(io.BytesIO):
    def __init__(self, content):
        super().__init__(content)
        self.bytes_read = 0

    def read(self, size=-1):
        content = super().read(size)
        self.bytes_read += len(content)
        return content


class ObservedResponse(requests.Response):
    def __init__(self, body, encoding="utf-8", status=200):
        super().__init__()
        self.raw = body
        self.encoding = encoding
        self.status_code = status
        self.closed_by_caller = False

    def close(self):
        self.closed_by_caller = True
        super().close()


def fetch(response, limit):
    tool = WebpageScraperTool(WebpageScraperToolConfig(max_content_length=limit))
    with patch("tool.webpage_scraper.requests.get", return_value=response) as get:
        result = tool._fetch_webpage("https://example.com/article")
    assert get.call_args.kwargs["stream"] is True
    return result


@pytest.mark.parametrize("declared_length", [None, "1", "999999999"])
def test_oversize_response_stops_reading_and_closes(declared_length):
    body = ObservedBody(b"x" * (1024 * 1024))
    response = ObservedResponse(body)
    if declared_length is not None:
        response.headers["Content-Length"] = declared_length

    with pytest.raises(ValueError, match="Content length exceeds maximum of 1024 bytes"):
        fetch(response, 1024)

    assert body.bytes_read < 1024 * 1024
    assert response.closed_by_caller
    assert body.closed


@pytest.mark.parametrize(
    "content, encoding",
    [
        (("x" * 8191 + "日本語").encode("utf-8"), "utf-8"),
        ("Café déjà vu".encode("cp1252"), "cp1252"),
        ("Café déjà vu".encode("cp1252"), None),
        ("日本語".encode("utf-8"), "not-a-real-charset"),
        (b"", None),
    ],
    ids=["utf8-split-chunk", "declared-cp1252", "detected-cp1252", "invalid-charset", "empty"],
)
def test_exact_byte_limit_preserves_requests_decoding(content, encoding):
    expected = ObservedResponse(io.BytesIO(content), encoding).text
    response = ObservedResponse(ObservedBody(content), encoding)

    assert fetch(response, len(content)) == expected
    assert response.closed_by_caller


def test_limit_counts_bytes_not_unicode_characters():
    response = ObservedResponse(ObservedBody(("é" * 4096).encode("utf-8")))

    with pytest.raises(ValueError, match="Content length exceeds maximum"):
        fetch(response, 4096)

    assert response.closed_by_caller


def test_limit_counts_decompressed_response_bytes():
    compressed = gzip.compress(b"x" * 100000)
    assert len(compressed) < 1024
    raw = HTTPResponse(body=ObservedBody(compressed), headers={"Content-Encoding": "gzip"}, preload_content=False)
    response = ObservedResponse(raw)
    response.headers["Content-Length"] = str(len(compressed))

    with pytest.raises(ValueError, match="Content length exceeds maximum"):
        fetch(response, 1024)

    assert response.closed_by_caller
    assert raw.closed


def test_http_error_closes_response_without_reading_body():
    body = ObservedBody(b"error page")
    response = ObservedResponse(body, status=503)

    with pytest.raises(requests.HTTPError):
        fetch(response, 1024)

    assert body.bytes_read == 0
    assert response.closed_by_caller
    assert body.closed


def test_stream_error_closes_response():
    body = ObservedBody(b"incomplete page")
    response = ObservedResponse(body)

    def failed_read(size=-1):
        raise requests.ConnectionError("connection lost")

    body.read = failed_read
    with pytest.raises(requests.ConnectionError, match="connection lost"):
        fetch(response, 1024)

    assert response.closed_by_caller
    assert body.closed
