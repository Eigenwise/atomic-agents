"""Select captions through the native transcript SDK using each example's language policy."""

import importlib.util
import json
from pathlib import Path
from urllib.parse import urlparse

import pytest
import requests
from requests.adapters import BaseAdapter
from youtube_transcript_api import YouTubeTranscriptApi


class CaptionTransport(BaseAdapter):
    """Serve deterministic YouTube page, player and caption responses to Requests."""

    def __init__(self, languages):
        super().__init__()
        self.languages = languages
        self.caption_requests = []

    def send(self, request, **kwargs):
        path = urlparse(request.url).path
        if path == "/watch":
            body = '<html><script>var config = {"INNERTUBE_API_KEY":"offline_key"};</script></html>'
        elif path == "/youtubei/v1/player":
            tracks = [
                {
                    "baseUrl": f"https://www.youtube.com/captions/{language}",
                    "name": {"runs": [{"text": language}]},
                    "languageCode": language,
                }
                for language in self.languages
            ]
            body = json.dumps(
                {
                    "playabilityStatus": {"status": "OK"},
                    "captions": {"playerCaptionsTracklistRenderer": {"captionTracks": tracks}},
                }
            )
        elif path.startswith("/captions/"):
            language = path.rsplit("/", 1)[-1]
            assert language in self.languages
            self.caption_requests.append(language)
            body = f'<transcript><text start="0" dur="2">Captions in {language}</text></transcript>'
        else:
            raise AssertionError(f"Unexpected HTTP request: {request.method} {request.url}")
        response = requests.Response()
        response.status_code = 200
        response.url = request.url
        response.request = request
        response.encoding = "utf-8"
        response._content = body.encode()
        return response

    def close(self):
        pass


@pytest.fixture(params=["youtube-summarizer", "youtube-to-recipe"])
def example(request):
    name = request.param
    path = Path(__file__).resolve().parents[2] / name / name.replace("-", "_") / "tools" / "youtube_transcript_scraper.py"
    spec = importlib.util.spec_from_file_location(f"language_test_{name.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "configured,available,explicit,expected",
    [
        (["de"], ["de"], None, "de"),
        (["de", "en"], ["en", "de"], None, "de"),
        (["fr", "de"], ["de"], None, "de"),
        (None, ["en-US"], None, "en-US"),
        (["de"], ["de", "fr"], "fr", "fr"),
        (None, ["en"], None, "en"),
    ],
)
def test_configured_language_priority_and_explicit_override(example, monkeypatch, configured, available, explicit, expected):
    config = example.YouTubeTranscriptToolConfig(**({"languages": configured} if configured is not None else {}))
    original_languages = list(config.languages)
    tool = example.YouTubeTranscriptTool(config)
    transport = CaptionTransport(available)
    with requests.Session() as session:
        session.trust_env = False
        session.mount("https://", transport)
        monkeypatch.setattr(example, "YouTubeTranscriptApi", lambda: YouTubeTranscriptApi(http_client=session))
        monkeypatch.setattr(
            tool,
            "fetch_video_metadata",
            lambda video_id: example.VideoMetadata(
                id=video_id, title="Offline video", channel="Offline channel", published_at="2025-01-01T00:00:00Z"
            ),
        )
        result = tool.run(
            example.YouTubeTranscriptToolInputSchema(
                video_url="https://www.youtube.com/watch?v=abcdefghijk", language=explicit
            )
        )

    assert result.transcript == f"Captions in {expected}"
    assert result.duration == 2.0
    assert result.metadata.id == "abcdefghijk"
    assert transport.caption_requests == [expected]
    assert config.languages == original_languages
