# YouTube Transcript Scraper Tool

## Overview
The **YouTube Transcript Scraper Tool** is a tool within the Atomic Agents ecosystem that allows you to fetch the transcript of a YouTube video.

## Prerequisites and Dependencies
- Python 3.9 or later
- atomic-agents (See [here](/README.md) for installation instructions)
- pydantic
- google-api-python-client
- youtube-transcript-api

## Installation
You can install the tool using any of the following options:

1. Using the CLI tool that comes with Atomic Agents. Simply run `atomic` and select the tool from the list of available tools. After doing so you will be asked for a target directory to download the tool into.
2. Good old fashioned copy/paste: Just like any other tool inside the Atomic Forge, you can copy the code from this repo directly into your own project, provided you already have atomic-agents installed according to the instructions in the main [README](/README.md).

## Configuration

### Parameters

- `api_key` (str): Your YouTube API key. Obtain this key by following the steps outlined in the [Obtaining a YouTube API Key](#obtaining-a-youtube-api-key) section.

### Example

```python
config = YouTubeTranscriptToolConfig(
    api_key="your_youtube_api_key"
)
```

### Obtaining a YouTube API Key

To use this tool, you'll need a YouTube API key. Follow these steps to obtain one:

1. **Access the Google Developers Console**
   - Visit the [Google Developers Console](https://console.developers.google.com/).
   - Sign in with your Google account. If you don't have one, you'll need to create it.

2. **Create a New Project**
   - Click on the project dropdown in the top-left corner and select "New Project."
   - Enter a project name and click "Create."

3. **Enable the YouTube Data API v3**
   - In the dashboard, click on "Enable APIs and Services."
   - Search for "YouTube Data API v3" and select it.
   - Click the "Enable" button.

4. **Generate Your API Key**
   - Navigate to "Credentials" in the left sidebar.
   - Click on "Create Credentials" and select "API Key."
   - Copy the generated API key and use it in your configuration as shown above.

## Input & Output Structure

### Input Schema
- `video_url` (str): URL of the YouTube video to fetch the transcript for.
- `language` (Optional[str]): Language code for the transcript (e.g., `'en'` for English).

### Output Schema
- `transcript` (str): Transcript of the YouTube video.
- `duration` (float): Video duration in seconds from YouTube's `contentDetails.duration`.
- `comments` (List[str]): Comments on the YouTube video.
- `metadata` (dict): Metadata of the YouTube video.

The tool fetches duration together with video metadata in one API request. It
uses the video length rather than adding caption durations, which can overlap
or omit silent sections. Missing or invalid video duration raises `ValueError` in `run()`.
Live and upcoming broadcasts report `contentDetails.duration` as `P0D`, so their output `duration` is `0.0`. The metadata-only `fetch_video_metadata()` method does not parse or validate duration.

## Usage

Here's an example of how to use the YouTube Transcript Scraper Tool:

```python
from tool.youtube_transcript_scraper import YouTubeTranscriptTool, YouTubeTranscriptToolConfig

# Initialize the tool with your API key
config = YouTubeTranscriptToolConfig(api_key="your_youtube_api_key")
transcript_tool = YouTubeTranscriptTool(config=config)

# Define input data
input_data = YouTubeTranscriptTool.input_schema(
    video_url="https://www.youtube.com/watch?v=t1e8gqXLbsU",
    language="en"
)

# Fetch the transcript
result = transcript_tool.run(input_data)
print(result)
```

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes with clear messages.
4. Open a pull request detailing your changes.

Please ensure you follow the project's coding standards and include tests for any new features or bug fixes.

## License

This project is licensed under the same license as the main Atomic Agents project. See the [LICENSE](../../../LICENSE) file in the repository root for more details.
