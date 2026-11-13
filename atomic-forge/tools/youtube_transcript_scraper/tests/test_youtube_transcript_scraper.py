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


def test_youtube_transcript_tool():
    mock_api_key = "mock_api_key"
    mock_video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
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
        mock_youtube.videos().list().execute.return_value = {
            "items": [{"snippet": mock_metadata, "contentDetails": {"duration": "PT10S"}}]
        }

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
        mock_youtube.videos().list().execute.return_value = {
            "items": [{"snippet": mock_metadata, "contentDetails": {"duration": "PT10S"}}]
        }

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


def test_extract_video_id():
    youtube_transcript_tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key="mock_api_key"))
    video_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    video_id = youtube_transcript_tool.extract_video_id(video_url)
    assert video_id == "dQw4w9WgXcQ"


@pytest.mark.parametrize(
    "snippets, api_duration, expected",
    [
        ([(0, 2), (120, 2)], "PT5M", 300.0),
        ([(0, 10), (5, 10)], "PT10S", 10.0),
        ([(0, 2)], "PT40S", 40.0),
        ([], "PT2M", 120.0),
        ([(0, 2)], "P1DT2H3M4.5S", 93784.5),
        ([], "PT0S", 0.0),
        ([], "P0D", 0.0),
    ],
)
def test_duration_comes_from_video_details(snippets, api_duration, expected):
    transcript = FetchedTranscript(
        snippets=[FetchedTranscriptSnippet(text="caption", start=start, duration=duration) for start, duration in snippets],
        video_id="dQw4w9WgXcQ",
        language="English",
        language_code="en",
        is_generated=False,
    )
    with patch(FETCH_TARGET, return_value=transcript), patch("tool.youtube_transcript_scraper.build") as mock_build:
        mock_build.return_value.videos.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "snippet": {"title": "Example", "channelTitle": "Channel", "publishedAt": "2026-01-01T00:00:00Z"},
                    "contentDetails": {"duration": api_duration},
                }
            ]
        }
        tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key="test-key"))
        result = tool.run(YouTubeTranscriptToolInputSchema(video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"))

    assert result.duration == expected
    mock_build.return_value.videos.return_value.list.assert_called_once_with(part="snippet,contentDetails", id="dQw4w9WgXcQ")


@pytest.mark.parametrize("content_details", [{}, {"duration": None}, {"duration": "not-a-duration"}])
def test_missing_video_duration_is_not_replaced_with_caption_lengths(content_details):
    with (
        patch(FETCH_TARGET, return_value=fetched_rick_astley_transcript()),
        patch("tool.youtube_transcript_scraper.build") as mock_build,
    ):
        mock_build.return_value.videos.return_value.list.return_value.execute.return_value = {
            "items": [
                {
                    "snippet": {"title": "Example", "channelTitle": "Channel", "publishedAt": "2026-01-01T00:00:00Z"},
                    "contentDetails": content_details,
                }
            ]
        }
        tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key="test-key"))
        with pytest.raises(ValueError):
            tool.run(YouTubeTranscriptToolInputSchema(video_url="https://www.youtube.com/watch?v=dQw4w9WgXcQ"))


@pytest.mark.parametrize("content_details", [None, {}, {"duration": None}, {"duration": "not-a-duration"}])
def test_fetch_video_metadata_does_not_require_valid_duration(content_details):
    item = {"snippet": {"title": "Example", "channelTitle": "Channel", "publishedAt": "2026-01-01T00:00:00Z"}}
    if content_details is not None:
        item["contentDetails"] = content_details
    with patch("tool.youtube_transcript_scraper.build") as mock_build:
        mock_build.return_value.videos().list().execute.return_value = {"items": [item]}
        tool = YouTubeTranscriptTool(YouTubeTranscriptToolConfig(api_key="test-key"))
        metadata = tool.fetch_video_metadata("dQw4w9WgXcQ")
    assert metadata.id == "dQw4w9WgXcQ"
    assert metadata.title == "Example"
    assert metadata.channel == "Channel"


if __name__ == "__main__":
    pytest.main([__file__])
