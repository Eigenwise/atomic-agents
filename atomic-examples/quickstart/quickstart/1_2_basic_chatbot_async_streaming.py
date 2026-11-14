import os
import instructor
import openai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.live import Live
from atomic_agents.context import ChatHistory
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema

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

# History setup
history = ChatHistory()

# Initialize history with an initial message from the assistant
initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
history.add_message("assistant", initial_message)

# OpenAI client setup using the Instructor library for async operations
client = instructor.from_openai(openai.AsyncOpenAI(api_key=API_KEY))

# Agent setup with specified configuration
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model="gpt-5.6-terra",
        model_api_parameters={"reasoning_effort": "low"},
        history=history,
    )
)

# Generate the default system prompt for the agent
default_system_prompt = agent.system_prompt_generator.generate_prompt()
# Display the system prompt in a styled panel
console.print(Panel(default_system_prompt, width=console.width, style="bold cyan"), style="bold cyan")

# Display the initial message from the assistant
console.print(Text("Agent:", style="bold green"), end=" ")
console.print(Text(initial_message.chat_message, style="green"))


async def main():
    # Start an infinite loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input with a styled prompt
        user_input = console.input("\n[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        # Process the user's input through the agent and get the streaming response
        input_schema = BasicChatInputSchema(chat_message=user_input)
        console.print()  # Add newline before response

        # Use Live display to show streaming response
        with Live("", refresh_per_second=10, auto_refresh=True) as live:
            current_response = ""
            # Use run_async_stream instead of run_async for streaming functionality
            async for partial_response in agent.run_async_stream(input_schema):
                if hasattr(partial_response, "chat_message") and partial_response.chat_message:
                    # Only update if we have new content
                    if partial_response.chat_message != current_response:
                        current_response = partial_response.chat_message
                        # Combine the label and response in the live display
                        display_text = Text.assemble(("Agent: ", "bold green"), (current_response, "green"))
                        live.update(display_text)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
