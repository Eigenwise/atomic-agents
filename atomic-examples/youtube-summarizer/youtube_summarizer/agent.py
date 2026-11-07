import instructor
import openai
from pydantic import Field
from typing import List, Optional

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema
from atomic_agents.context import BaseDynamicContextProvider, SystemPromptGenerator


class YtTranscriptProvider(BaseDynamicContextProvider):
    def __init__(self, title):
        super().__init__(title)
        self.transcript = None
        self.duration = None
        self.metadata = None

    def get_info(self) -> str:
        return f'VIDEO TRANSCRIPT: "{self.transcript}"\n\nDURATION: {self.duration}\n\nMETADATA: {self.metadata}'


class YouTubeKnowledgeExtractionInputSchema(BaseIOSchema):
    """This schema defines the input schema for the YouTubeKnowledgeExtractionAgent."""

    video_url: str = Field(..., description="The URL of the YouTube video to analyze")


class YouTubeKnowledgeExtractionOutputSchema(BaseIOSchema):
    """This schema defines an elaborate set of insights about the contentof the video."""

    summary: str = Field(
        ..., description="A short summary of the content, including who is presenting and the content being discussed."
    )
    insights: List[str] = Field(
        ..., min_items=5, max_items=5, description="exactly 5 of the best insights and ideas from the input."
    )
    quotes: List[str] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="exactly 5 of the most surprising, insightful, and/or interesting quotes from the input.",
    )
    habits: Optional[List[str]] = Field(
        None,
        min_items=5,
        max_items=5,
        description="exactly 5 of the most practical and useful personal habits mentioned by the speakers.",
    )
    facts: List[str] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="exactly 5 of the most surprising, insightful, and/or interesting valid facts about the greater world mentioned in the content.",
    )
    recommendations: List[str] = Field(
        ...,
        min_items=5,
        max_items=5,
        description="exactly 5 of the most surprising, insightful, and/or interesting recommendations from the content.",
    )
    references: List[str] = Field(
        ...,
        description="All mentions of writing, art, tools, projects, and other sources of inspiration mentioned by the speakers.",
    )
    one_sentence_takeaway: str = Field(
        ..., description="The most potent takeaways and recommendations condensed into a single 20-word sentence."
    )


transcript_provider = YtTranscriptProvider(title="YouTube Transcript")

youtube_knowledge_extraction_agent = AtomicAgent[
    YouTubeKnowledgeExtractionInputSchema, YouTubeKnowledgeExtractionOutputSchema
](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI()),
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "This Assistant is an expert at extracting knowledge and other insightful and interesting information from YouTube transcripts."
            ],
            steps=[
                "Analyse the YouTube transcript thoroughly to extract the most valuable insights, facts, and recommendations.",
                "Adhere strictly to the provided schema when extracting information from the input content.",
                "Ensure that the output matches the field descriptions, types and constraints exactly.",
            ],
            output_instructions=[
                "Only output Markdown-compatible strings.",
                "Ensure you follow ALL these instructions when creating your output.",
            ],
            context_providers={"yt_transcript": transcript_provider},
        ),
    )
)
