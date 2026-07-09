"""
Persistent Memory Example
==========================

Demonstrates a custom, persistent ``ChatHistory`` backend built with nothing but the
Python standard library's ``sqlite3`` module (zero extra dependencies). It follows the
recommended extension pattern from the Memory guide's "Writing a Custom Memory Backend"
section: subclass ``ChatHistory`` to inherit turn handling and serialization, and
override ``add_message`` to persist on every write.

Run this script twice: the second run rehydrates the conversation from the first.
"""

import contextlib
import os
import sqlite3
from typing import Optional

import instructor
import openai
from dotenv import load_dotenv
from rich.console import Console
from rich.text import Text

from atomic_agents import AtomicAgent, AgentConfig, BaseIOSchema, BasicChatInputSchema, BasicChatOutputSchema
from atomic_agents.context import ChatHistory

load_dotenv()

# API Key setup
API_KEY = ""
if not API_KEY:
    API_KEY = os.getenv("OPENAI_API_KEY")

if not API_KEY:
    raise ValueError(
        "API key is not set. Please set the API key as a static variable or in the environment variable OPENAI_API_KEY."
    )


class SQLiteChatHistory(ChatHistory):
    """A ChatHistory that persists to a local SQLite database, keyed by session_id.

    Follows the recommended pattern from the Memory guide: subclass ChatHistory to inherit
    turn handling and serialization, then override add_message to persist. copy() is also
    overridden so the agent's initial_history snapshot and reset_history() keep this
    persistent backend instead of silently degrading to a plain in-memory ChatHistory.

    A short-lived connection is opened per operation (via contextlib.closing) rather than
    held open for the object's lifetime, so no handle is leaked and the backend is safe to
    use from more than one thread.
    """

    def __init__(self, session_id: str, db_path: str = "chat_history.db", max_messages: Optional[int] = None):
        super().__init__(max_messages=max_messages)
        self.session_id = session_id
        self.db_path = db_path

        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS sessions (session_id TEXT PRIMARY KEY, data TEXT)")
            conn.commit()
            row = conn.execute("SELECT data FROM sessions WHERE session_id = ?", (self.session_id,)).fetchone()

        if row is not None:
            self.load(row[0])

    def add_message(self, role: str, content: BaseIOSchema) -> None:
        super().add_message(role, content)
        self._save()

    def _save(self) -> None:
        with contextlib.closing(sqlite3.connect(self.db_path)) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO sessions (session_id, data) VALUES (?, ?)",
                (self.session_id, self.dump()),
            )
            conn.commit()

    def copy(self) -> "SQLiteChatHistory":
        new_history = SQLiteChatHistory(self.session_id, db_path=self.db_path, max_messages=self.max_messages)
        new_history.load(self.dump())
        new_history.current_turn_id = self.current_turn_id
        return new_history


def main() -> None:
    # Initialize a Rich Console for pretty console outputs
    console = Console()

    # History setup - restores prior turns for this session_id if chat_history.db exists
    history = SQLiteChatHistory(session_id="demo")
    console.print(f"[bold yellow]Restored {history.get_message_count()} message(s) from {history.db_path}[/bold yellow]")

    if history.get_message_count() == 0:
        initial_message = BasicChatOutputSchema(chat_message="Hello! How can I assist you today?")
        history.add_message("assistant", initial_message)

    # OpenAI client setup using the Instructor library
    client = instructor.from_openai(openai.OpenAI(api_key=API_KEY))

    # Agent setup with specified configuration
    agent = AtomicAgent[BasicChatInputSchema, BasicChatOutputSchema](
        config=AgentConfig(
            client=client,
            model="gpt-5-mini",
            model_api_parameters={"reasoning_effort": "low"},
            history=history,
        )
    )

    # Display the last message so the user sees where the conversation left off
    last_message = history.history[-1].content
    console.print(Text("Agent:", style="bold green"), end=" ")
    console.print(Text(last_message.chat_message, style="bold green"))

    # Start an infinite loop to handle user inputs and agent responses
    while True:
        # Prompt the user for input with a styled prompt
        user_input = console.input("[bold blue]You:[/bold blue] ")
        # Check if the user wants to exit the chat
        if user_input.lower() in ["/exit", "/quit"]:
            console.print(
                f"Exiting chat... your conversation is saved in {history.db_path}. Run again to pick up where you left off."
            )
            break

        # Process the user's input through the agent and get the response
        input_schema = BasicChatInputSchema(chat_message=user_input)
        response = agent.run(input_schema)

        agent_message = Text(response.chat_message, style="bold green")
        console.print(Text("Agent:", style="bold green"), end=" ")
        console.print(agent_message)


if __name__ == "__main__":
    main()
