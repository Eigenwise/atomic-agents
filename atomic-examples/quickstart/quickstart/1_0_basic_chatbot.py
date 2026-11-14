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

# OpenAI client setup using the Instructor library
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
console.print(Text(initial_message.chat_message, style="bold green"))

# Start an infinite loop to handle user inputs and agent responses
while True:
    # Prompt the user for input with a styled prompt
    user_input = console.input("[bold blue]You:[/bold blue] ")
    # Check if the user wants to exit the chat
    if user_input.lower() in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        break
    # Check if the user wants to see token count
    if user_input.lower() == "/tokens":
        token_info = agent.get_context_token_count()
        console.print("[bold magenta]Token Usage:[/bold magenta]")
        console.print(f"  Total: {token_info.total} tokens")
        console.print(f"  System prompt: {token_info.system_prompt} tokens")
        console.print(f"  History: {token_info.history} tokens")
        if token_info.utilization:
            console.print(f"  Context utilization: {token_info.utilization:.1%}")
        continue

    # Process the user's input through the agent and get the response
    input_schema = BasicChatInputSchema(chat_message=user_input)
    response = agent.run(input_schema)

    agent_message = Text(response.chat_message, style="bold green")
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(agent_message)
