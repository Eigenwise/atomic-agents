import instructor
import openai
from pydantic import Field, HttpUrl
from typing import List
from atomic_agents import BaseIOSchema, AtomicAgent, AgentConfig
from atomic_agents.context import SystemPromptGenerator


class QuestionAnsweringAgentInputSchema(BaseIOSchema):
    """This schema defines the input schema for the QuestionAnsweringAgent."""

    question: str = Field(..., description="A question that needs to be answered based on the provided context.")


class QuestionAnsweringAgentOutputSchema(BaseIOSchema):
    """This schema defines the output schema for the QuestionAnsweringAgent."""

    markdown_output: str = Field(..., description="The answer to the question in markdown format.")
    references: List[HttpUrl] = Field(
        ..., max_items=3, description="A list of up to 3 HTTP URLs used as references for the answer."
    )
    followup_questions: List[str] = Field(
        ..., max_items=3, description="A list of up to 3 follow-up questions related to the answer."
    )


# Create the question answering agent
question_answering_agent = AtomicAgent[QuestionAnsweringAgentInputSchema, QuestionAnsweringAgentOutputSchema](
    AgentConfig(
        client=instructor.from_openai(openai.OpenAI()),
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        system_prompt_generator=SystemPromptGenerator(
            background=[
                "You are an intelligent question answering expert.",
                "Your task is to provide accurate and detailed answers to user questions based on the given context.",
            ],
            steps=[
                "You will receive a question and the context information.",
                "Provide up to 3 relevant references (HTTP URLs) used in formulating the answer.",
                "Generate up to 3 follow-up questions related to the answer.",
            ],
            output_instructions=[
                "Ensure clarity and conciseness in each answer.",
                "Ensure the answer is directly relevant to the question and context provided.",
                "Include up to 3 relevant HTTP URLs as references.",
                "Provide up to 3 follow-up questions to encourage further exploration of the topic.",
            ],
        ),
    )
)
