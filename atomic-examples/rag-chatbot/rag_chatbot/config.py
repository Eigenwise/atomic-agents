import os
from dataclasses import dataclass
from enum import Enum


class VectorDBType(Enum):
    CHROMA = "chroma"
    QDRANT = "qdrant"


def get_api_key() -> str:
    """Retrieve API key from environment or raise error"""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key not found. Please set the OPENAI_API_KEY environment variable.")
    return api_key


def get_vector_db_type() -> VectorDBType:
    """Get the vector database type from environment variable"""
    db_type = os.getenv("VECTOR_DB_TYPE", "chroma").lower()
    try:
        return VectorDBType(db_type)
    except ValueError:
        raise ValueError(f"Invalid VECTOR_DB_TYPE: {db_type}. Must be 'chroma' or 'qdrant'")


@dataclass
class ChatConfig:
    """Configuration for the chat application"""

    api_key: str = get_api_key()
    model: str = "gpt-5-mini"
    reasoning_effort: str = "low"
    exit_commands: set[str] = frozenset({"/exit", "exit", "quit", "/quit"})

    def __init__(self):
        # Prevent instantiation
        raise TypeError("ChatConfig is not meant to be instantiated")


# Model Configuration
EMBEDDING_MODEL = "text-embedding-3-small"  # OpenAI's latest embedding model
CHUNK_SIZE = 1000  # Maximum characters per chunk, including paragraph separators
CHUNK_OVERLAP = 200  # Up to this many trailing characters; must be less than CHUNK_SIZE

# Vector Search Configuration
NUM_CHUNKS_TO_RETRIEVE = 3
SIMILARITY_METRIC = "cosine"

# Vector Database Configuration
VECTOR_DB_TYPE = get_vector_db_type()

# ChromaDB Configuration
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")

# Qdrant Configuration
QDRANT_PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qdrant_db")

# History Configuration
HISTORY_SIZE = 10  # Number of messages to keep in conversation history
MAX_CONTEXT_LENGTH = 4000  # Maximum length of combined context to send to the model
