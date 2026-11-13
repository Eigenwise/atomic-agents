"""Publication metadata must identify the same instant after serialization."""

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest


ROOT = Path(__file__).resolve().parents[4]
IMPLEMENTATIONS = [
    "atomic-forge/tools/youtube_transcript_scraper/tool/youtube_transcript_scraper.py",
    "atomic-examples/youtube-summarizer/youtube_summarizer/tools/youtube_transcript_scraper.py",
    "atomic-examples/youtube-to-recipe/youtube_to_recipe/tools/youtube_transcript_scraper.py",
]


@pytest.mark.parametrize("path", IMPLEMENTATIONS, ids=["forge", "summarizer", "recipe"])
@pytest.mark.parametrize("published_at", ["2026-01-01T03:04:05Z", "2026-01-01T03:04:05.123Z", "2026-01-01T03:04:05+05:30"])
def test_metadata_preserves_publication_instant_and_offset(path, published_at, monkeypatch):
    name = "publication_timezone_" + path.split("/")[1].replace("-", "_")
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    payload = {"items": [{"snippet": {"title": "Example", "channelTitle": "Channel", "publishedAt": published_at}}]}
    with patch.object(module, "build") as build:
        build.return_value.videos.return_value.list.return_value.execute.return_value = payload
        tool = module.YouTubeTranscriptTool(module.YouTubeTranscriptToolConfig(api_key="test-key"))
        metadata = tool.fetch_video_metadata("dQw4w9WgXcQ")

    expected = datetime.fromisoformat(published_at)
    assert metadata.published_at.utcoffset() == expected.utcoffset()
    assert metadata.published_at.timestamp() == expected.timestamp()
    assert metadata.published_at.astimezone(timezone.utc) == expected.astimezone(timezone.utc)
    serialized = json.loads(metadata.model_dump_json())["published_at"]
    assert datetime.fromisoformat(serialized) == expected
    assert module.VideoMetadata.model_validate_json(metadata.model_dump_json()).published_at == expected
    assert metadata.id == "dQw4w9WgXcQ"
    assert metadata.title == "Example"
    assert metadata.channel == "Channel"
