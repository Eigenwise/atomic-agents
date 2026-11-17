# YouTube Recipe Extractor

This directory contains the YouTube Recipe Extractor example for the Atomic Agents project. This example demonstrates how to extract structured recipe information from cooking videos using the Atomic Agents framework.

## Getting Started

To get started with the YouTube Recipe Extractor:

1. **Clone the main Atomic Agents repository:**

   ```bash
   git clone https://github.com/eigenwise/atomic-agents
   ```

2. **Navigate to the YouTube Recipe Extractor directory:**

   ```bash
   cd atomic-agents/atomic-examples/youtube-to-recipe
   ```

3. **Install the dependencies using uv:**

   ```bash
   uv sync
   ```

4. **Set up environment variables:**

   Create a `.env` file in the `youtube-to-recipe` directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   YOUTUBE_API_KEY=your_youtube_api_key
   ```

   To get your YouTube API key, follow the instructions in the [YouTube Scraper README](/atomic-forge/tools/youtube_transcript_scraper/README.md).

   Replace `your_openai_api_key` and `your_youtube_api_key` with your actual API keys.

5. **Run the YouTube Recipe Extractor:**

   ```bash
   uv run python youtube_to_recipe/main.py
   ```

## File Explanation

### 1. Agent (`agent.py`)

This module defines the `YouTubeRecipeExtractionAgent`, responsible for extracting structured recipe information from cooking video transcripts. It extracts:

- Recipe name and description
- Ingredients with quantities and units
- Step-by-step cooking instructions
- Required equipment
- Cooking times and temperatures
- Tips and dietary information

### 2. YouTube Transcript Scraper (`tools/youtube_transcript_scraper.py`)

This tool comes from the [Atomic Forge](/atomic-forge/README.md) and handles fetching transcripts and metadata from YouTube cooking videos.

### 3. Main (`main.py`)

The entry point for the YouTube Recipe Extractor application. It orchestrates fetching transcripts, processing them through the agent, and outputting structured recipe information.

## Example Output

The agent extracts recipe information in a structured format including:

- Detailed ingredient lists with measurements
- Step-by-step cooking instructions with timing and temperature
- Required kitchen equipment
- Cooking tips and tricks
- Dietary information and cuisine type
- Preparation and cooking times

## Customization

You can modify the `video_url` variable in `main.py` to extract recipes from different cooking videos. Additionally, you can adjust the agent's configuration in `agent.py` to customize the recipe extraction format or add additional fields to capture more recipe details.

When the transcript input omits `language`, the scraper uses the language list in
`YouTubeTranscriptToolConfig`, such as `languages=["de", "en"]`, in priority order.
An explicit input language overrides that list. The default list is `en`, `en-US`, `en-GB`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.
