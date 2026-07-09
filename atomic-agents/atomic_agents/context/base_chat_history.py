from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, Optional

from atomic_agents.base.base_io_schema import BaseIOSchema

if TYPE_CHECKING:
    from atomic_agents.context.chat_history import Message


class BaseChatHistory(ABC):
    """
    Declares the public contract of a chat history.

    This is the full method surface of the built-in ``ChatHistory``, of which ``AtomicAgent``
    itself uses a subset (``add_message``, ``get_history``, ``initialize_turn``,
    ``delete_turn_id``, ``copy``, plus direct reads of the ``history`` / ``current_turn_id``
    attributes). The remaining methods (``dump`` / ``load`` / ``get_message_count`` /
    ``get_current_turn_id``) round out the standard history surface that tooling and callers
    rely on.

    This is an interface-only abstract base class: it defines no state and no behavior,
    only the methods a chat history implementation must provide. ``ChatHistory`` is the
    framework's built-in, in-memory implementation. Implement this interface directly (or
    subclass ``ChatHistory``) to plug in a custom persistent or backend-specific memory store.

    Every implementation must maintain the following two attributes, since callers
    (including ``AtomicAgent._trim_context``) read them directly:

    Attributes:
        history (List[Message]): The in-memory working list of messages that make up the
            chat history. ``AtomicAgent._trim_context`` reads this list directly (via
            ``msg.turn_id``) to decide which turns to trim when a context-length limit
            is exceeded.
        current_turn_id (Optional[str]): The ID of the current turn, or ``None`` if no
            turn has been started yet. Set by ``initialize_turn()``.
    """

    history: List["Message"]
    current_turn_id: Optional[str]

    @abstractmethod
    def initialize_turn(self) -> None:
        """
        Starts a new turn, generating and storing a new turn ID.

        Implementations must set ``self.current_turn_id`` to a new, unique value each
        time this is called.
        """
        raise NotImplementedError

    @abstractmethod
    def add_message(self, role: str, content: BaseIOSchema) -> None:
        """
        Adds a message to the chat history, tagging it with the current turn ID.

        If no turn has been started yet (``current_turn_id`` is ``None``), implementations
        should start one (e.g. by calling ``initialize_turn()``) before recording the message.

        Args:
            role (str): The role of the message sender (e.g. 'user', 'system', 'assistant').
            content (BaseIOSchema): The content of the message.
        """
        raise NotImplementedError

    @abstractmethod
    def get_history(self) -> List[Dict]:
        """
        Retrieves the chat history in the wire format expected by the LLM client.

        Returns:
            List[Dict]: The list of messages as dictionaries with 'role' and 'content' keys,
                suitable for passing to a chat completion call.
        """
        raise NotImplementedError

    @abstractmethod
    def get_current_turn_id(self) -> Optional[str]:
        """
        Returns the current turn ID.

        Returns:
            Optional[str]: The current turn ID, or ``None`` if no turn has been started.
        """
        raise NotImplementedError

    @abstractmethod
    def delete_turn_id(self, turn_id: str):
        """
        Deletes all messages belonging to the given turn ID.

        Args:
            turn_id (str): The turn ID whose messages should be removed.

        Raises:
            ValueError: If the specified turn ID is not found in the history.
        """
        raise NotImplementedError

    @abstractmethod
    def get_message_count(self) -> int:
        """
        Returns the number of messages currently in the chat history.

        Returns:
            int: The number of messages.
        """
        raise NotImplementedError

    @abstractmethod
    def dump(self) -> str:
        """
        Serializes the entire chat history to a JSON string.

        The returned string must be a valid argument to ``load()``, such that
        ``instance.load(instance.dump())`` round-trips the full history state.

        Returns:
            str: A JSON string representation of the chat history.
        """
        raise NotImplementedError

    @abstractmethod
    def load(self, serialized_data: str) -> None:
        """
        Deserializes a JSON string produced by ``dump()`` and loads it into this instance,
        replacing any existing history state.

        Args:
            serialized_data (str): A JSON string representation of the chat history,
                as produced by ``dump()``.

        Raises:
            ValueError: If the serialized data is invalid or cannot be deserialized.
        """
        raise NotImplementedError

    @abstractmethod
    def copy(self) -> "BaseChatHistory":
        """
        Creates a copy of the chat history, independent of the original.

        ``AtomicAgent`` calls ``copy()`` to snapshot ``initial_history`` at construction and
        to restore it in ``reset_history()``. The built-in ``ChatHistory.copy()`` returns a
        plain ``ChatHistory``; a subclass that carries extra state (a database handle, a
        ``session_id``, a store reference) MUST override ``copy()`` to return its own type,
        otherwise ``reset_history()`` silently replaces the backend with a plain in-memory
        history and later writes stop reaching the store.

        Returns:
            BaseChatHistory: A copy of the chat history.
        """
        raise NotImplementedError
