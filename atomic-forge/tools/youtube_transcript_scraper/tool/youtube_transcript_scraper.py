import os
import re
from urllib.parse import parse_qs, urlsplit
from typing import List, Optional
from pydantic import Field, BaseModel
from datetime import datetime

from googleapiclient.discovery import build
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


################
# INPUT SCHEMA #
################
class YouTubeTranscriptToolInputSchema(BaseIOSchema):
    """
    Tool for fetching the transcript of a YouTube video using the YouTube Transcript API.
    Returns the transcript with text, start time, and duration.
    """

    video_url: str = Field(..., description="URL of the YouTube video to fetch the transcript for.")
    language: Optional[str] = Field(None, description="Language code for the transcript (e.g., 'en' for English).")


#################
# OUTPUT SCHEMA #
#################
class VideoMetadata(BaseModel):
    """Schema for YouTube video metadata."""

    id: str = Field(..., description="The YouTube video ID.")
    title: str = Field(..., description="The title of the YouTube video.")
    channel: str = Field(..., description="The name of the YouTube channel.")
    published_at: datetime = Field(..., description="The publication date and time of the video.")


class YouTubeTranscriptToolOutputSchema(BaseIOSchema):
    """
    Output schema for the YouTubeTranscriptTool. Contains the transcript text, duration, comments, and metadata.
    """

    transcript: str = Field(..., description="Transcript of the YouTube video.")
    duration: float = Field(..., description="Duration of the YouTube video in seconds.")
    comments: List[str] = Field(default_factory=list, description="Comments on the YouTube video.")
    metadata: VideoMetadata = Field(..., description="Metadata of the YouTube video.")


#################
# CONFIGURATION #
#################
class YouTubeTranscriptToolConfig(BaseToolConfig):
    api_key: str = Field(
        description="YouTube API key for fetching video metadata.",
        default=os.getenv("YOUTUBE_API_KEY"),
    )


#####################
# MAIN TOOL & LOGIC #
#####################
class YouTubeTranscriptTool(BaseTool[YouTubeTranscriptToolInputSchema, YouTubeTranscriptToolOutputSchema]):
    """
    Tool for fetching the transcript of a YouTube video using the YouTube Transcript API.

    Attributes:
        input_schema (YouTubeTranscriptToolInputSchema): The schema for the input data.
        output_schema (YouTubeTranscriptToolOutputSchema): The schema for the output data.
    """

    input_schema = YouTubeTranscriptToolInputSchema
    output_schema = YouTubeTranscriptToolOutputSchema

    def __init__(self, config: YouTubeTranscriptToolConfig):
        """
        Initializes the YouTubeTranscriptTool.

        Args:
            config (YouTubeTranscriptToolConfig): Configuration for the tool.
        """
        super().__init__(config)
        self.api_key = config.api_key

    def run(self, params: YouTubeTranscriptToolInputSchema) -> YouTubeTranscriptToolOutputSchema:
        """
        Runs the YouTubeTranscriptTool with the given parameters.

        Args:
            params (YouTubeTranscriptToolInputSchema): The input parameters for the tool, adhering to the input schema.

        Returns:
            YouTubeTranscriptToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            ValueError: If the video URL or ID is invalid or unsupported.
            Exception: If fetching the transcript fails.
        """
        video_id = self.extract_video_id(params.video_url)
        transcript_api = YouTubeTranscriptApi()
        try:
            if params.language:
                fetched_transcript = transcript_api.fetch(video_id, languages=[params.language])
            else:
                fetched_transcript = transcript_api.fetch(video_id)
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            raise Exception(f"Failed to fetch transcript for video '{video_id}': {str(e)}")

        transcripts = fetched_transcript.to_raw_data()
        transcript_text = " ".join([transcript["text"] for transcript in transcripts])
        total_duration = sum([transcript["duration"] for transcript in transcripts])

        metadata = self.fetch_video_metadata(video_id)

        return YouTubeTranscriptToolOutputSchema(
            transcript=transcript_text,
            duration=total_duration,
            comments=[],
            metadata=metadata,
        )

    @staticmethod
    def extract_video_id(url: str) -> str:
        """
        Extracts a video ID from a watch, share, Shorts, or embed URL, or a bare ID.

        Args:
            url (str): An HTTP(S) YouTube video URL or an 11-character video ID.

        Returns:
            str: The extracted video ID.
        """
        error_message = "Invalid YouTube video URL or ID. Use a watch, youtu.be, shorts, or embed URL, or a video ID."
        value = url.strip()
        if re.fullmatch(r"[A-Za-z0-9_-]{11}", value):
            return value

        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError(error_message) from exc

        if parsed.scheme not in {"http", "https"}:
            raise ValueError(error_message)

        video_id = ""
        path = parsed.path.rstrip("/")
        if parsed.hostname == "youtu.be":
            video_id = path.removeprefix("/")
        elif parsed.hostname in {"youtube.com", "www.youtube.com", "m.youtube.com"}:
            if path == "/watch":
                video_id = parse_qs(parsed.query, keep_blank_values=True).get("v", [""])[0]
            else:
                parts = path.split("/")
                if len(parts) == 3 and parts[1] in {"shorts", "embed"}:
                    video_id = parts[2]

        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError(error_message)
        return video_id

    def fetch_video_metadata(self, video_id: str) -> VideoMetadata:
        """
        Fetches metadata for a YouTube video.

        Args:
            video_id (str): The YouTube video ID.

        Returns:
            VideoMetadata: The metadata of the video.

        Raises:
            Exception: If no metadata is found for the video.
        """
        youtube = build("youtube", "v3", developerKey=self.api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        if not response["items"]:
            raise Exception(f"No metadata found for video '{video_id}'")

        video_info = response["items"][0]["snippet"]
        return VideoMetadata(
            id=video_id,
            title=video_info["title"],
            channel=video_info["channelTitle"],
            published_at=datetime.fromisoformat(video_info["publishedAt"].rstrip("Z")),
        )


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    from rich.console import Console
    from dotenv import load_dotenv

    load_dotenv()

    rich_console = Console()
    api_key = os.getenv("YOUTUBE_API_KEY")
    search_tool_instance = YouTubeTranscriptTool(config=YouTubeTranscriptToolConfig(api_key=api_key))

    search_input = YouTubeTranscriptTool.input_schema(video_url="https://www.youtube.com/watch?v=t1e8gqXLbsU", language="en")

    output = search_tool_instance.run(search_input)
    rich_console.print(output)
