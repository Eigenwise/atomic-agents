import os
import instructor
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.context import ChatHistory
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from dotenv import load_dotenv

load_dotenv()

# Initialize a Rich Console for pretty console outputs
console = Console()

# History setup
history = ChatHistory()


# Function to set up the client based on the chosen provider
def setup_client(provider):
    console.log(f"provider: {provider}")
    if provider == "1" or provider == "openai":
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        client = instructor.from_openai(OpenAI(api_key=api_key))
        model = "gpt-5-mini"
        model_api_parameters = {"reasoning_effort": "low", "max_tokens": 2048}
        assistant_role = "assistant"
    elif provider == "2" or provider == "anthropic":
        from anthropic import Anthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        client = instructor.from_anthropic(Anthropic(api_key=api_key))
        model = "claude-3-5-haiku-20241022"
        model_api_parameters = {"max_tokens": 2048}
        assistant_role = "assistant"
    elif provider == "3" or provider == "groq":
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
        model = "mixtral-8x7b-32768"
        model_api_parameters = {"max_tokens": 2048}
        assistant_role = "assistant"
    elif provider == "4" or provider == "ollama":
        from openai import OpenAI as OllamaClient

        client = instructor.from_openai(
            OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama"), mode=instructor.Mode.JSON
        )
        model = "llama3"
        model_api_parameters = {"max_tokens": 2048}
        assistant_role = "assistant"
    elif provider == "5" or provider == "gemini":
        import google.genai

        api_key = os.getenv("GEMINI_API_KEY")
        client = instructor.from_genai(
            google.genai.Client(api_key=api_key),
            mode=instructor.Mode.GENAI_TOOLS,
        )
        model = "gemini-2.5-flash"
        model_api_parameters = {}
        assistant_role = "model"
    elif provider == "6" or provider == "openrouter":
        from openai import OpenAI as OpenRouterClient

        api_key = os.getenv("OPENROUTER_API_KEY")
        client = instructor.from_openai(OpenRouterClient(base_url="https://openrouter.ai/api/v1", api_key=api_key))
        model = "mistral/ministral-8b"
        model_api_parameters = {"max_tokens": 2048}
        assistant_role = "assistant"
    elif provider == "7" or provider == "minimax":
        from openai import OpenAI as MiniMaxClient

        api_key = os.getenv("MINIMAX_API_KEY")
        client = instructor.from_openai(
            MiniMaxClient(base_url="https://api.minimax.io/v1", api_key=api_key),
            mode=instructor.Mode.JSON,
        )
        model = "MiniMax-M3"
        model_api_parameters = {"max_tokens": 2048}
        assistant_role = "assistant"
    elif provider == "8" or provider == "edenai":
        from openai import OpenAI as EdenAIClient

        api_key = os.getenv("EDENAI_API_KEY")
        client = instructor.from_openai(EdenAIClient(base_url="https://api.edenai.run/v3", api_key=api_key))
        model = "openai/gpt-4o-mini"
        model_api_parameters = {"max_tokens": 2048}
        assistant_role = "assistant"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return client, model, model_api_parameters, assistant_role


# Prompt the user to choose a provider from one in the list below.
providers_list = ["openai", "anthropic", "groq", "ollama", "gemini", "openrouter", "minimax", "edenai"]
y = "bold yellow"
b = "bold blue"
g = "bold green"
provider_inner_str = (
    f"{' / '.join(f'[[{g}]{i + 1}[/{g}]]. [{b}]{provider}[/{b}]' for i, provider in enumerate(providers_list))}"
)
providers_str = f"[{y}]Choose a provider ({provider_inner_str}): [/{y}]"

provider = console.input(providers_str).lower()

# Set up the client and model based on the chosen provider
client, model, model_api_parameters, assistant_role = setup_client(provider)

# Initialize history with an initial message from the assistant
initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
history.add_message(assistant_role, initial_message)

# Agent setup with specified configuration
agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
    config=AgentConfig(
        client=client,
        model=model,
        history=history,
        assistant_role=assistant_role,
        model_api_parameters=model_api_parameters,
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
    # Check if the user wants to see token count (works with any provider!)
    if user_input.lower() == "/tokens":
        token_info = agent.get_context_token_count()
        console.print(f"[bold magenta]Token Usage ({model}):[/bold magenta]")
        console.print(f"  Total: {token_info.total} tokens")
        console.print(f"  System prompt: {token_info.system_prompt} tokens")
        console.print(f"  History: {token_info.history} tokens")
        if token_info.max_tokens:
            console.print(f"  Max context: {token_info.max_tokens} tokens")
        if token_info.utilization:
            console.print(f"  Context utilization: {token_info.utilization:.1%}")
        continue

    # Process the user's input through the agent and get the response
    input_schema = BasicChatInputSchema(chat_message=user_input)
    response = agent.run(input_schema)

    agent_message = Text(response.chat_message, style="bold green")
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(agent_message)
