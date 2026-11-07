import os
import asyncio
import instructor
import openai
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from atomic_agents import BaseIOSchema, AtomicAgent, AgentConfig, BasicChatInputSchema
from atomic_agents.context import SystemPromptGenerator

# API Key setup
API_KEY = ""
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY."
    )

# Initialize a Rich Console for pretty console outputs
console = Console()

# OpenAI client setup using the Instructor library
client = instructor.from_openai(openai.AsyncOpenAI(api_key=API_KEY))


# Define a schema for the output data
class PersonSchema(BaseIOSchema):
    """Schema for person information."""

    name: str
    age: int
    pronouns: list[str]
    profession: str


# System prompt generator setup
system_prompt_generator = SystemPromptGenerator(
    background=["You parse a sentence and extract elements."],
    steps=[],
    output_instructions=[],
)

dataset = [
    "My name is Mike, I am 30 years old, my pronouns are he/him, and I am a software engineer.",
    "My name is Sarah, I am 25 years old, my pronouns are she/her, and I am a data scientist.",
    "My name is John, I am 40 years old, my pronouns are he/him, and I am a product manager.",
    "My name is Emily, I am 35 years old, my pronouns are she/her, and I am a UX designer.",
    "My name is David, I am 28 years old, my pronouns are he/him, and I am a web developer.",
    "My name is Anna, I am 32 years old, my pronouns are she/her, and I am a graphic designer.",
]

# Max concurrent requests - adjust this to see performance differences
MAX_CONCURRENT = 3
sem = asyncio.Semaphore(MAX_CONCURRENT)

# Agent setup with specified configuration
agent = AtomicAgent[BasicChatInputSchema, PersonSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        system_prompt_generator=system_prompt_generator,
    )
)


async def exec_agent(message: str, idx: int, progress_dict: dict):
    """Execute the agent with the provided message and update progress in real-time."""
    # Acquire the semaphore to limit concurrent executions
    async with sem:
        user_input = BasicChatInputSchema(chat_message=message)
        agent.reset_history()

        # Track streaming progress
        partial_data = {}
        progress_dict[idx] = {"status": "Processing", "data": partial_data, "message": message}

        partial_response = None
        # Actually demonstrate streaming by processing each partial response
        async for partial_response in agent.run_async_stream(user_input):
            if partial_response:
                # Extract any available fields from the partial response
                response_dict = partial_response.model_dump()
                for field in ["name", "age", "pronouns", "profession"]:
                    if field in response_dict and response_dict[field]:
                        partial_data[field] = response_dict[field]

                # Update progress dictionary to display changes in real-time
                progress_dict[idx]["data"] = partial_data.copy()
                # Small sleep to simulate processing and make streaming more visible
                await asyncio.sleep(0.05)

        assert partial_response
        # Final response with complete data
        response = PersonSchema(**partial_response.model_dump())
        progress_dict[idx]["status"] = "Complete"
        progress_dict[idx]["data"] = response.model_dump()

        return response


def generate_status_table(progress_dict: dict) -> Table:
    """Generate a rich table showing the current processing status."""
    table = Table(title="Asynchronous Stream Processing Demo")

    table.add_column("ID", justify="center")
    table.add_column("Status", justify="center")
    table.add_column("Input", style="cyan")
    table.add_column("Current Data", style="green")

    for idx, info in progress_dict.items():
        # Format the partial data nicely
        data_str = ""
        if info["data"]:
            for k, v in info["data"].items():
                data_str += f"{k}: {v}\n"

        status_style = "yellow" if info["status"] == "Processing" else "green"

        # Add row with current processing information
        table.add_row(
            f"{idx + 1}",
            f"[{status_style}]{info['status']}[/{status_style}]",
            Text(info["message"][:30] + "..." if len(info["message"]) > 30 else info["message"]),
            data_str or "Waiting...",
        )

    return table


async def process_all(dataset: list[str]):
    """Process all items in dataset with visual progress tracking."""
    progress_dict = {}  # Track processing status for visualization

    # Create tasks for each message processing
    tasks = []
    for idx, message in enumerate(dataset):
        # Initialize entry in progress dictionary
        progress_dict[idx] = {"status": "Waiting", "data": {}, "message": message}
        # Create task without awaiting it
        task = asyncio.create_task(exec_agent(message, idx, progress_dict))
        tasks.append(task)

    # Display live updating status while tasks run
    with Live(generate_status_table(progress_dict), refresh_per_second=10) as live:
        while not all(task.done() for task in tasks):
            # Update the live display with current progress
            live.update(generate_status_table(progress_dict))
            await asyncio.sleep(0.1)

        # Final update after all tasks complete
        live.update(generate_status_table(progress_dict))

    # Gather all results when complete
    responses = await asyncio.gather(*tasks)
    return responses


if __name__ == "__main__":
    console.print("[bold blue]Starting Asynchronous Stream Processing Demo[/bold blue]")
    console.print(f"Processing {len(dataset)} items with max {MAX_CONCURRENT} concurrent requests\n")

    responses = asyncio.run(process_all(dataset))

    # Display final results in a structured table
    results_table = Table(title="Processing Results")
    results_table.add_column("Name", style="cyan")
    results_table.add_column("Age", justify="center")
    results_table.add_column("Pronouns")
    results_table.add_column("Profession")

    for resp in responses:
        results_table.add_row(resp.name, str(resp.age), "/".join(resp.pronouns), resp.profession)

    console.print(results_table)
