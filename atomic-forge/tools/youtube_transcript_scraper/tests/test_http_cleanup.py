"""Metadata requests must release their owned SDK transport on every exit."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
from pathlib import Path
import sys
from threading import Thread
from urllib.parse import parse_qs, urlsplit

from googleapiclient.discovery import build_from_document
from googleapiclient.discovery_cache import get_static_doc
from googleapiclient.errors import HttpError
import httplib2
import pytest


ROOT = Path(__file__).resolve().parents[4]
IMPLEMENTATIONS = [
    "atomic-forge/tools/youtube_transcript_scraper/tool/youtube_transcript_scraper.py",
    "atomic-examples/youtube-summarizer/youtube_summarizer/tools/youtube_transcript_scraper.py",
    "atomic-examples/youtube-to-recipe/youtube_to_recipe/tools/youtube_transcript_scraper.py",
]
VIDEO_ID = "dQw4w9WgXcQ"


class RecordingHttp(httplib2.Http):
    """Keep socket references so GC cannot make an unclosed transport look closed."""

    def __init__(self):
        super().__init__(proxy_info=None, timeout=5)
        self.sockets = []

    def request(self, *args, **kwargs):
        result = super().request(*args, **kwargs)
        self.sockets.extend(connection.sock for connection in self.connections.values() if connection.sock is not None)
        return result


@pytest.fixture
def metadata_server():
    state = {"status": 200, "payload": {}, "paths": []}

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            state["paths"].append(self.path)
            body = json.dumps(state["payload"]).encode()
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive()


@pytest.mark.parametrize("path", IMPLEMENTATIONS, ids=["forge", "summarizer", "recipe"])
@pytest.mark.parametrize("outcome", ["success", "no_items", "http_error"])
def test_metadata_closes_sdk_connections(path, outcome, metadata_server, monkeypatch):
    monkeypatch.setenv("YOUTUBE_API_KEY", "test-key")
    name = "http_cleanup_" + path.split("/")[1].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    endpoint, state = metadata_server
    state["payload"] = {
        "items": [{"snippet": {"title": "Example", "channelTitle": "Channel", "publishedAt": "2026-01-01T00:00:00Z"}}]
    }
    if outcome == "no_items":
        state["payload"] = {"items": []}
    elif outcome == "http_error":
        state["status"] = 403
        state["payload"] = {"error": {"code": 403, "message": "Access denied"}}

    document = json.loads(get_static_doc("youtube", "v3"))
    document["rootUrl"] = endpoint
    document["servicePath"] = ""
    transports = []

    def build(service, version, *, developerKey):
        assert (service, version, developerKey) == ("youtube", "v3", "test-key")
        transport = RecordingHttp()
        transports.append(transport)
        return build_from_document(document, http=transport, developerKey=developerKey)

    monkeypatch.setattr(module, "build", build)
    tool = module.YouTubeTranscriptTool(module.YouTubeTranscriptToolConfig(api_key="test-key"))
    try:
        if outcome == "success":
            for _ in range(3):
                metadata = tool.fetch_video_metadata(VIDEO_ID)
                assert (metadata.id, metadata.title, metadata.channel) == (VIDEO_ID, "Example", "Channel")
        elif outcome == "no_items":
            with pytest.raises(Exception, match="No metadata found"):
                tool.fetch_video_metadata(VIDEO_ID)
        else:
            with pytest.raises(HttpError, match="Access denied") as error:
                tool.fetch_video_metadata(VIDEO_ID)
            assert error.value.resp.status == 403

        assert len(state["paths"]) == (3 if outcome == "success" else 1)
        for path in state["paths"]:
            parsed = urlsplit(path)
            assert parsed.path == "/youtube/v3/videos"
            assert parse_qs(parsed.query)["id"] == [VIDEO_ID]
        for transport in transports:
            assert transport.sockets, "The SDK must have used a real keep-alive connection"
            assert all(sock.fileno() == -1 for sock in transport.sockets)
            assert transport.connections == {}
    finally:
        for transport in transports:
            transport.close()
