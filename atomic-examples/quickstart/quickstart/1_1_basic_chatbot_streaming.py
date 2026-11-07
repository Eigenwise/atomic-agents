import os
import instructor
import openai
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
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

# OpenAI client setup using the Instructor library for synchronous operations
client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))

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


def main():
    """
    Main function to handle the chat loop using synchronous streaming.
    This demonstrates how to use AtomicAgent.run_stream() instead of the async version.
    """
    # Start an infinite loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input with a styled prompt
        user_input = console.input("\n[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print("Exiting chat...")
            break

        # Process the user's input through the agent
        input_schema = BasicChatInputSchema(chat_message=user_input)
        console.print()  # Add newline before response
        console.print(Text("Agent: ", style="bold green"), end="")

        # Current display string to avoid repeating output
        current_display = ""

        # Use run_stream for synchronous streaming responses
        for partial_response in agent.run_stream(input_schema):
            if hasattr(partial_response, "chat_message") and partial_response.chat_message:
                # Only output the incremental part of the message
                new_content = partial_response.chat_message
                if new_content != current_display:
                    # Only print the new part since the last update
                    if new_content.startswith(current_display):
                        incremental_text = new_content[len(current_display) :]
                        console.print(Text(incremental_text, style="green"), end="")
                        current_display = new_content
                    else:
                        # If there's a mismatch, print the full message
                        # (this should rarely happen with most LLMs)
                        console.print(Text(new_content, style="green"), end="")
                        current_display = new_content

                    # Flush to ensure output is displayed immediately
                    console.file.flush()

        console.print()  # Add a newline after the response is complete


if __name__ == "__main__":
    main()
