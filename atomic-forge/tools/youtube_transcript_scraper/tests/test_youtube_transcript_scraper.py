import os
import sys
import pytest
from unittest.mock import ANY, patch, MagicMock
from datetime import datetime

from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet, NoTranscriptFound, TranscriptsDisabled

# Add the parent directory of 'tests' to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.youtube_transcript_scraper import (  # noqa: E402
    YouTubeTranscriptTool,
    YouTubeTranscriptToolInputSchema,
    YouTubeTranscriptToolOutputSchema,
    YouTubeTranscriptToolConfig,
)

FETCH_TARGET = "tool.youtube_transcript_scraper.YouTubeTranscriptApi.fetch"


def fetched_rick_astley_transcript() -> FetchedTranscript:
    return FetchedTranscript(
        snippets=[
            FetchedTranscriptSnippet(text="Never gonna give you up", start=0.0, duration=5.0),
            FetchedTranscriptSnippet(text="Never gonna let you down", start=5.0, duration=5.0),
        ],
        video_id="dQw4w9WgXcQ",
        language="English",
        language_code="en",
        is_generated=False,
    )


@pytest.mark.parametrize(
    "mock_video_url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=share-token",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ",
        "https://www.youtube.com/embed/dQw4w9WgXcQ",
    ],
)
def test_youtube_transcript_tool(mock_video_url):
    mock_api_key = "mock_api_key"
    mock_video_id = "dQw4w9WgXcQ"
    mock_metadata = {
        "title": "Rick Astley - Never Gonna Give You Up",
        "channelTitle": "Rick Astley",
        "publishedAt": "2009-10-25T06:57:33Z",
    }

    with (
        patch(FETCH_TARGET, autospec=True, return_value=fetched_rick_astley_transcript()) as mock_fetch,
        patch("tool.youtube_transcript_scraper.build") as mock_build,
    ):
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = {"items": [{"snippet": mock_metadata}]}

        youtube_transcript_tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key=mock_api_key))
        input_schema = YouTubeTranscriptToolInputSchema(video_url=mock_video_url)
        result = youtube_transcript_tool.run(input_schema)

        mock_fetch.assert_called_once_with(ANY, mock_video_id)
        assert isinstance(result, YouTubeTranscriptToolOutputSchema)
        assert result.transcript == "Never gonna give you up Never gonna let you down"
        assert result.duration == 10.0
        assert result.comments == []
        assert result.metadata.model_dump() == {
            "id": mock_video_id,
            "title": mock_metadata["title"],
            "channel": mock_metadata["channelTitle"],
            "published_at": datetime.fromisoformat(mock_metadata["publishedAt"].replace("Z", "")),
        }


def test_youtube_transcript_tool_with_language():
    mock_api_key = "mock_api_key"
    mock_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    mock_metadata = {
        "title": "Rick Astley - Never Gonna Give You Up",
        "channelTitle": "Rick Astley",
        "publishedAt": "2009-10-25T06:57:33Z",
    }

    with (
        patch(FETCH_TARGET, autospec=True, return_value=fetched_rick_astley_transcript()) as mock_fetch,
        patch("tool.youtube_transcript_scraper.build") as mock_build,
    ):
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = {"items": [{"snippet": mock_metadata}]}

        youtube_transcript_tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key=mock_api_key))
        input_schema = YouTubeTranscriptToolInputSchema(video_url=mock_video_url, language="de")
        result = youtube_transcript_tool.run(input_schema)

        mock_fetch.assert_called_once_with(ANY, "dQw4w9WgXcQ", languages=["de"])
        assert isinstance(result, YouTubeTranscriptToolOutputSchema)
        assert result.transcript == "Never gonna give you up Never gonna let you down"


def test_youtube_transcript_tool_no_transcript():
    mock_api_key = "mock_api_key"
    mock_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch(
        FETCH_TARGET,
        autospec=True,
        side_effect=NoTranscriptFound(
            video_id="dQw4w9WgXcQ",
            requested_language_codes=["en"],
            transcript_data=None,
        ),
    ):
        youtube_transcript_tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key=mock_api_key))
        input_schema = YouTubeTranscriptToolInputSchema(video_url=mock_video_url)

        with pytest.raises(Exception) as excinfo:
            youtube_transcript_tool.run(input_schema)

        assert "Failed to fetch transcript" in str(excinfo.value)


def test_youtube_transcript_tool_transcripts_disabled():
    mock_api_key = "mock_api_key"
    mock_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

    with patch(
        FETCH_TARGET,
        autospec=True,
        side_effect=TranscriptsDisabled("dQw4w9WgXcQ"),
    ):
        youtube_transcript_tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key=mock_api_key))
        input_schema = YouTubeTranscriptToolInputSchema(video_url=mock_video_url)

        with pytest.raises(Exception) as excinfo:
            youtube_transcript_tool.run(input_schema)

        assert "Failed to fetch transcript" in str(excinfo.value)


def test_fetch_video_metadata_no_items():
    mock_api_key = "mock_api_key"

    with patch("tool.youtube_transcript_scraper.build") as mock_build:
        mock_youtube = MagicMock()
        mock_build.return_value = mock_youtube
        mock_youtube.videos().list().execute.return_value = {"items": []}

        youtube_transcript_tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key=mock_api_key))

        with pytest.raises(Exception) as excinfo:
            youtube_transcript_tool.fetch_video_metadata("dQw4w9WgXcQ")

        assert "No metadata found for video" in str(excinfo.value)


@pytest.mark.parametrize(
    "video_url",
    [
        "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "http://youtube.com/watch?feature=shared&v=dQw4w9WgXcQ&t=30#comments",
        "https://m.youtube.com/watch?v=dQw4w9WgXcQ",
        "https://WWW.YouTube.COM/watch?v=dQw4w9WgXcQ",
        "https://youtu.be/dQw4w9WgXcQ?si=share-token&t=30",
        "https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share",
        "https://www.youtube.com/embed/dQw4w9WgXcQ?start=30",
        "https://www.youtube.com/embed/dQw4w9WgXcQ/",
        "https://www.youtube.com/watch?v=aB_12-xyZ09",
        "  https://youtu.be/dQw4w9WgXcQ  ",
        "dQw4w9WgXcQ",
    ],
)
def test_extract_video_id(video_url):
    expected = "aB_12-xyZ09" if "aB_12-xyZ09" in video_url else "dQw4w9WgXcQ"
    assert YouTubeTranscriptTool.extract_video_id(video_url) == expected


@pytest.mark.parametrize(
    "video_url",
    [
        "",
        "not a URL",
        "https://example.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com.example.com/watch?v=dQw4w9WgXcQ",
        "https://example.com/?next=https://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://youtube.com@elsewhere.example/watch?v=dQw4w9WgXcQ",
        "ftp://youtube.com/watch?v=dQw4w9WgXcQ",
        "https://www.youtube.com/watch",
        "https://www.youtube.com/watch?v=",
        "https://www.youtube.com/watch?v=short",
        "https://www.youtube.com/watch?v=dQw4w9WgXcQextra",
        "https://www.youtube.com/watch?v=dQw4w9WgXc!",
        "https://www.youtube.com/watch?v=&v=dQw4w9WgXcQ",
        "https://www.youtube.com/playlist?list=dQw4w9WgXcQ",
        "https://www.youtube.com/shorts/",
        "https://youtu.be/",
        "https://youtu.be/dQw4w9WgXcQ/extra",
        "https://www.youtube.com/embed/dQw4w9WgXcQ/extra",
        "https://[invalid/watch?v=dQw4w9WgXcQ",
    ],
)
def test_extract_video_id_rejects_invalid_urls(video_url):
    with pytest.raises(ValueError, match="Invalid YouTube video URL or ID"):
        YouTubeTranscriptTool.extract_video_id(video_url)


def test_invalid_video_url_does_not_call_external_services():
    tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key="mock_api_key"))
    with patch(FETCH_TARGET) as mock_fetch, patch("tool.youtube_transcript_scraper.build") as mock_build:
        with pytest.raises(ValueError, match="Invalid YouTube video URL or ID"):
            tool.run(YouTubeTranscriptToolInputSchema(video_url="https://example.com/watch?v=dQw4w9WgXcQ"))
        mock_fetch.assert_not_called()
        mock_build.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__])
