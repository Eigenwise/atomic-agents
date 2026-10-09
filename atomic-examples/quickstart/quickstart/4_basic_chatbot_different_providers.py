import os
import instructor
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from atomic_agents.context import ChatHistory
from atomic_agents import AtomicAgent, AgentConfig, BasicChatInputSchema, BasicChatOutputSchema
from dotenv import load_dotenv

# Initialize a Rich Console for pretty console outputs
console = Console()

CHEAPER_INFERENCE_BASE_URL = "https://api.cheaperinference.com/v1"


def _require_api_key(env_var):
    """Return a nonempty API key from the environment, or raise before any client exists."""
    api_key = (os.getenv(env_var) or "").strip()
    if not api_key:
        raise ValueError(f"{env_var} is not set. Set it to a nonempty API key to use this provider.")
    return api_key


# Each setup function returns (client, model, model_api_parameters, assistant_role)
def _setup_openai():
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    client = instructor.from_openai(OpenAI(api_key=api_key))
    return client, "gpt-5-mini", {"reasoning_effort": "low", "max_tokens": 2048}, "assistant"


def _setup_anthropic():
    from anthropic import Anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY")
    client = instructor.from_anthropic(Anthropic(api_key=api_key))
    return client, "claude-3-5-haiku-20241022", {"max_tokens": 2048}, "assistant"


def _setup_groq():
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    client = instructor.from_groq(Groq(api_key=api_key), mode=instructor.Mode.JSON)
    return client, "mixtral-8x7b-32768", {"max_tokens": 2048}, "assistant"


def _setup_ollama():
    from openai import OpenAI as OllamaClient

    client = instructor.from_openai(
        OllamaClient(base_url="http://localhost:11434/v1", api_key="ollama"), mode=instructor.Mode.JSON
    )
    return client, "llama3", {"max_tokens": 2048}, "assistant"


def _setup_gemini():
    import google.genai

    api_key = os.getenv("GEMINI_API_KEY")
    client = instructor.from_genai(
        google.genai.Client(api_key=api_key),
        mode=instructor.Mode.GENAI_TOOLS,
    )
    return client, "gemini-2.5-flash", {}, "model"


def _setup_openrouter():
    from openai import OpenAI as OpenRouterClient

    api_key = os.getenv("OPENROUTER_API_KEY")
    client = instructor.from_openai(OpenRouterClient(base_url="https://openrouter.ai/api/v1", api_key=api_key))
    return client, "mistral/ministral-8b", {"max_tokens": 2048}, "assistant"


def _setup_minimax():
    from openai import OpenAI as MiniMaxClient

    api_key = os.getenv("MINIMAX_API_KEY")
    client = instructor.from_openai(
        MiniMaxClient(base_url="https://api.minimax.io/v1", api_key=api_key),
        mode=instructor.Mode.JSON,
    )
    return client, "MiniMax-M3", {"max_tokens": 2048}, "assistant"


def _setup_edenai():
    from openai import OpenAI as EdenAIClient

    api_key = os.getenv("EDENAI_API_KEY")
    client = instructor.from_openai(EdenAIClient(base_url="https://api.edenai.run/v3", api_key=api_key))
    return client, "openai/gpt-4o-mini", {"max_tokens": 2048}, "assistant"


def _setup_cheaperinference():
    # Validate first: with api_key=None the OpenAI SDK falls back to OPENAI_API_KEY.
    api_key = _require_api_key("CHEAPER_INFERENCE_API_KEY")

    from openai import OpenAI as CheaperInferenceClient

    client = instructor.from_openai(
        CheaperInferenceClient(base_url=CHEAPER_INFERENCE_BASE_URL, api_key=api_key),
        mode=instructor.Mode.TOOLS,
    )
    return client, "gpt-5.4-mini", {"max_tokens": 2048}, "assistant"


# Provider name -> setup function. The order defines the menu numbers (1, 2, ...).
PROVIDER_SETUPS = {
    "openai": _setup_openai,
    "anthropic": _setup_anthropic,
    "groq": _setup_groq,
    "ollama": _setup_ollama,
    "gemini": _setup_gemini,
    "openrouter": _setup_openrouter,
    "minimax": _setup_minimax,
    "edenai": _setup_edenai,
    "cheaperinference": _setup_cheaperinference,
}
providers_list = list(PROVIDER_SETUPS)
PROVIDER_NUMBERS = {str(i): name for i, name in enumerate(providers_list, start=1)}


# Function to set up the client based on the chosen provider (menu number or name)
def setup_client(provider):
    console.log(f"provider: {provider}")
    setup = PROVIDER_SETUPS.get(PROVIDER_NUMBERS.get(provider, provider))
    if setup is None:
        raise ValueError(f"Unsupported provider: {provider}")
    return setup()


def _provider_prompt():
    """Build the styled provider menu, e.g. "[1]. openai / [2]. anthropic / ..."."""
    y = "bold yellow"
    b = "bold blue"
    g = "bold green"
    provider_inner_str = (
        f"{' / '.join(f'[[{g}]{i + 1}[/{g}]]. [{b}]{provider}[/{b}]' for i, provider in enumerate(providers_list))}"
    )
    return f"[{y}]Choose a provider ({provider_inner_str}): [/{y}]"


def _choose_provider():
    """Prompt the user to choose a provider by menu number or name."""
    return console.input(_provider_prompt()).lower()


def _print_agent_message(message):
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(Text(message, style="bold green"))


def _build_agent(client, model, model_api_parameters, assistant_role):
    """Create the agent with a history that starts with the assistant greeting."""
    history = ChatHistory()
    initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
    history.add_message(assistant_role, initial_message)

    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model=model,
            history=history,
            assistant_role=assistant_role,
            model_api_parameters=model_api_parameters,
        )
    )
    return agent, initial_message


def _print_token_usage(agent, model):
    """Show the context token count (works with any provider)."""
    token_info = agent.get_context_token_count()
    console.print(f"[bold magenta]Token Usage ({model}):[/bold magenta]")
    console.print(f"  Total: {token_info.total} tokens")
    console.print(f"  System prompt: {token_info.system_prompt} tokens")
    console.print(f"  History: {token_info.history} tokens")
    if token_info.max_tokens:
        console.print(f"  Max context: {token_info.max_tokens} tokens")
    if token_info.utilization:
        console.print(f"  Context utilization: {token_info.utilization:.1%}")


def _handle_turn(agent, model, user_input):
    """Handle one user input. Return False when the user wants to exit."""
    command = user_input.lower()
    if command in ["/exit", "/quit"]:
        console.print("Exiting chat...")
        return False
    if command == "/tokens":
        _print_token_usage(agent, model)
        return True

    response = agent.run(BasicChatInputSchema(chat_message=user_input))
    _print_agent_message(response.chat_message)
    return True


def _chat_loop(agent, model):
    """Read user inputs and print agent responses until the user exits."""
    while True:
        user_input = console.input("[bold blue]You:[/bold blue] ")
        if not _handle_turn(agent, model, user_input):
            break


def main():
    load_dotenv()

    provider = _choose_provider()
    client, model, model_api_parameters, assistant_role = setup_client(provider)
    agent, initial_message = _build_agent(client, model, model_api_parameters, assistant_role)

    # Display the default system prompt in a styled panel, then the greeting
    default_system_prompt = agent.system_prompt_generator.generate_prompt()
    console.print(Panel(default_system_prompt, width=console.width, style="bold cyan"), style="bold cyan")
    _print_agent_message(initial_message.chat_message)

    _chat_loop(agent, model)


if __name__ == "__main__":
    main()
