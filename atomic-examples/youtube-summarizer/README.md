# YouTube Summarizer

This directory contains the YouTube Summarizer example for the Atomic Agents project. This example demonstrates how to extract and summarize knowledge from YouTube videos using the Atomic Agents framework.

## Getting Started

To get started with the YouTube Summarizer:

1. **Clone the main Atomic Agents repository:**
   ```bash
   git clone https://github.com/eigenwise/atomic-agents
   ```

2. **Navigate to the YouTube Summarizer directory:**
   ```bash
   cd atomic-agents/atomic-examples/youtube-summarizer
   ```

3. **Install the dependencies using uv:**
   ```bash
   uv sync
   ```

4. **Set up environment variables:**

   Create a `.env` file in the `youtube-summarizer` directory with the following content:
   ```env
   OPENAI_API_KEY=your_openai_api_key
   YOUTUBE_API_KEY=your_youtube_api_key
   ```

   To get your YouTube API key, follow the instructions in the [YouTube Scraper README](/atomic-forge/tools/youtube_transcript_scraper/README.md).

   Replace `your_openai_api_key` and `your_youtube_api_key` with your actual API keys.

5. **Run the YouTube Summarizer:**
   ```bash
   uv run python youtube_summarizer/main.py
   ```
   or

   ```bash
   uv run python -m youtube_summarizer.main
   ```

## File Explanation

### 1. Agent (`agent.py`)

This module defines the `YouTubeKnowledgeExtractionAgent`, responsible for extracting summaries, insights, quotes, and more from YouTube video transcripts.


### 2. YouTube Transcript Scraper (`tools/youtube_transcript_scraper.py`)

This tool comes from the [Atomic Forge](/atomic-forge/README.md) and handles fetching transcripts and metadata from YouTube videos.

### 3. Main (`main.py`)

The entry point for the YouTube Summarizer application. It orchestrates fetching transcripts, processing them through the agent, and displaying the results.

## Customization

You can modify the `video_url` variable in `main.py` to analyze different YouTube videos. Additionally, you can adjust the agent's configuration in `agent.py` to tailor the summaries and insights according to your requirements.

When the transcript input omits `language`, the scraper uses the language list in
`YouTubeTranscriptToolConfig`, such as `languages=["de", "en"]`, in priority order.
An explicit input language overrides that list. The default list is `en`, `en-US`, `en-GB`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.
