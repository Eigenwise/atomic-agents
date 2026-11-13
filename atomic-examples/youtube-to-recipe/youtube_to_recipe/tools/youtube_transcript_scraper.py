import os
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
    """
    Configuration for the YouTubeTranscriptTool.

    Attributes:
        languages (List[str]): List of language codes to try when fetching transcripts.
    """

    languages: List[str] = ["en", "en-US", "en-GB"]


#####################
# MAIN TOOL & LOGIC #
#####################
class YouTubeTranscriptTool(BaseTool[YouTubeTranscriptToolInputSchema, YouTubeTranscriptToolOutputSchema]):
    """
    Tool for extracting transcripts from YouTube videos.

    Attributes:
        input_schema (YouTubeTranscriptToolInputSchema): The schema for the input data.
        output_schema (YouTubeTranscriptToolOutputSchema): The schema for the output data.
        languages (List[str]): List of language codes to try when fetching transcripts.
    """

    def __init__(self, config: YouTubeTranscriptToolConfig = YouTubeTranscriptToolConfig()):
        """
        Initializes the YouTubeTranscriptTool.

        Args:
            config (YouTubeTranscriptToolConfig): Configuration for the tool.
        """
        super().__init__(config)
        self.languages = config.languages

    def run(self, params: YouTubeTranscriptToolInputSchema) -> YouTubeTranscriptToolOutputSchema:
        """
        Runs the YouTubeTranscriptTool with the given parameters.

        Args:
            params (YouTubeTranscriptToolInputSchema): The input parameters for the tool, adhering to the input schema.

        Returns:
            YouTubeTranscriptToolOutputSchema: The output of the tool, adhering to the output schema.

        Raises:
            Exception: If fetching the transcript fails.
        """
        video_id = self.extract_video_id(params.video_url)
        try:
            if params.language:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id, languages=[params.language])
            else:
                transcripts = YouTubeTranscriptApi.get_transcript(video_id)
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            raise Exception(f"Failed to fetch transcript for video '{video_id}': {str(e)}")

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
        Extracts the video ID from a YouTube URL.

        Args:
            url (str): The YouTube video URL.

        Returns:
            str: The extracted video ID.
        """
        return url.split("v=")[-1].split("&")[0]

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
        api_key = os.getenv("YOUTUBE_API_KEY")
        youtube = build("youtube", "v3", developerKey=api_key)
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        if not response["items"]:
            raise Exception(f"No metadata found for video '{video_id}'")

        video_info = response["items"][0]["snippet"]
        return VideoMetadata(
            id=video_id,
            title=video_info["title"],
            channel=video_info["channelTitle"],
            published_at=datetime.fromisoformat(video_info["publishedAt"]),
        )


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    from rich.console import Console
    from dotenv import load_dotenv

    load_dotenv()

    rich_console = Console()
    search_tool_instance = YouTubeTranscriptTool(config=YouTubeTranscriptToolConfig())

    search_input = YouTubeTranscriptTool.input_schema(video_url="https://www.youtube.com/watch?v=t1e8gqXLbsU", language="en")

    output = search_tool_instance.run(search_input)
    rich_console.print(output)
