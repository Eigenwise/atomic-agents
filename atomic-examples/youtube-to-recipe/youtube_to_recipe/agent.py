import instructor
import openai
from pydantic import BaseModel, Field
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


class YouTubeRecipeExtractionInputSchema(BaseIOSchema):
    """This schema defines the input schema for the YouTubeRecipeExtractionAgent."""

    video_url: str = Field(..., description="The URL of the YouTube cooking video to analyze")


class Ingredient(BaseModel):
    """Model for recipe ingredients"""

    item: str = Field(..., description="The ingredient name")
    amount: str = Field(..., description="The quantity of the ingredient")
    unit: Optional[str] = Field(None, description="The unit of measurement, if applicable")
    notes: Optional[str] = Field(None, description="Any special notes about the ingredient")


class Step(BaseModel):
    """Model for recipe steps"""

    instruction: str = Field(..., description="The cooking instruction")
    duration: Optional[str] = Field(None, description="Time required for this step, if mentioned")
    temperature: Optional[str] = Field(None, description="Cooking temperature, if applicable")
    tips: Optional[str] = Field(None, description="Any tips or warnings for this step")


class YouTubeRecipeExtractionOutputSchema(BaseIOSchema):
    """This schema defines the structured recipe information extracted from the video."""

    recipe_name: str = Field(..., description="The name of the recipe being prepared")
    chef: Optional[str] = Field(None, description="The name of the chef/cook presenting the recipe")
    description: str = Field(..., description="A brief description of the dish and its characteristics")
    prep_time: Optional[str] = Field(None, description="Total preparation time mentioned in the video")
    cook_time: Optional[str] = Field(None, description="Total cooking time mentioned in the video")
    servings: Optional[int] = Field(None, description="Number of servings this recipe makes")
    ingredients: List[Ingredient] = Field(..., description="List of ingredients with their quantities and units")
    steps: List[Step] = Field(..., description="Detailed step-by-step cooking instructions")
    equipment: List[str] = Field(..., description="List of kitchen equipment and tools needed")
    tips: List[str] = Field(..., description="General cooking tips and tricks mentioned in the video")
    difficulty_level: Optional[str] = Field(None, description="Difficulty level of the recipe (e.g., Easy, Medium, Hard)")
    cuisine_type: Optional[str] = Field(None, description="Type of cuisine (e.g., Italian, Mexican, Japanese)")
    dietary_info: List[str] = Field(
        default_factory=list, description="Dietary information (e.g., Vegetarian, Vegan, Gluten-free)"
    )


transcript_provider = YtTranscriptProvider(title="YouTube Recipe Transcript")

youtube_recipe_extraction_agent = AtomicAgent[YouTubeRecipeExtractionInputSchema, YouTubeRecipeExtractionOutputSchema](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI()),
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "This Assistant is an expert at extracting detailed recipe information from cooking video transcripts.",
                "It understands cooking terminology, measurements, and techniques.",
            ],
            steps=[
                "Analyze the cooking video transcript thoroughly to extract recipe details.",
                "Convert approximate measurements and instructions into precise recipe format.",
                "Identify all ingredients, steps, equipment, and cooking tips mentioned.",
                "Ensure all critical recipe information is captured accurately.",
            ],
            output_instructions=[
                "Only output Markdown-compatible strings.",
                "Maintain proper units and measurements in recipe format.",
                "Include all safety warnings and important cooking notes.",
            ],
            context_providers={"yt_transcript": transcript_provider},
        ),
    )
)
