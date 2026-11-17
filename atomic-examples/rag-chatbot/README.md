# RAG Chatbot

This directory contains the RAG (Retrieval-Augmented Generation) Chatbot example for the Atomic Agents project. This example demonstrates how to build an intelligent chatbot that uses document retrieval to provide context-aware responses using the Atomic Agents framework.

## Features

1. Document Chunking: Automatically splits documents into manageable chunks with configurable overlap
2. Vector Storage: Supports both [ChromaDB](https://www.trychroma.com/) and [Qdrant](https://qdrant.tech/) for efficient storage and retrieval of document chunks
3. Semantic Search: Generates and executes semantic search queries to find relevant context
4. Context-Aware Responses: Provides detailed answers based on retrieved document chunks
5. Interactive UI: Rich console interface with progress indicators and formatted output

## Getting Started

To get started with the RAG Chatbot:

1. **Clone the main Atomic Agents repository:**

   ```bash
   git clone https://github.com/eigenwise/atomic-agents
   ```

2. **Navigate to the RAG Chatbot directory:**

   ```bash
   cd atomic-agents/atomic-examples/rag-chatbot
   ```

3. **Install the dependencies using uv:**

   ```bash
   uv sync
   ```

4. **Set up environment variables:**
   Create a `.env` file in the `rag-chatbot` directory with the following content:

   ```env
   OPENAI_API_KEY=your_openai_api_key
   VECTOR_DB_TYPE=chroma  # or 'qdrant'
   ```

   Replace `your_openai_api_key` with your actual OpenAI API key.

5. **Run the RAG Chatbot:**

   ```bash
   uv run python rag_chatbot/main.py
   ```

## Vector Database Configuration

The RAG Chatbot supports two vector databases:

### ChromaDB (Default)

- **Local storage**: Data is stored locally in the `chroma_db/` directory
- **Configuration**: Set `VECTOR_DB_TYPE=chroma` in your `.env` file

### Qdrant

- **Local storage**: Data is stored locally in the `qdrant_db/` directory
- **Configuration**: Set `VECTOR_DB_TYPE=qdrant` in your `.env` file

## Usage

### Using ChromaDB (Default)

```bash
export VECTOR_DB_TYPE=chroma
uv run python rag_chatbot/main.py
```

### Using Qdrant (Local)

```bash
export VECTOR_DB_TYPE=qdrant
uv run python rag_chatbot/main.py
```

## Components

### 1. Query Agent (`agents/query_agent.py`)

Generates semantic search queries based on user questions to find relevant document chunks.

### 2. QA Agent (`agents/qa_agent.py`)

Analyzes retrieved chunks and generates comprehensive answers to user questions.

### 3. Vector Database Services (`services/`)

- **Base Service** (`services/base.py`): Abstract interface for vector database operations
- **ChromaDB Service** (`services/chroma_db.py`): ChromaDB implementation
- **Qdrant Service** (`services/qdrant_db.py`): Qdrant implementation
- **Factory** (`services/factory.py`): Creates the appropriate service based on configuration

### 4. Context Provider (`context_providers.py`)

Provides retrieved document chunks as context to the agents.

### 5. Main Script (`main.py`)

Orchestrates the entire process, from document processing to user interaction.

## How It Works

1. The system initializes by:
   - Downloading a sample document (State of the Union address)
   - Splitting it into chunks with configurable overlap
   - Storing chunks in the selected vector database with vector embeddings

2. For each user question:
   - The Query Agent generates an optimized semantic search query
   - Relevant chunks are retrieved from the vector database
   - The QA Agent analyzes the chunks and generates a detailed answer
   - The system displays the thought process and final answer

## Customization

You can customize the RAG Chatbot by:

- Modifying chunk size and overlap in `config.py`
- Adjusting the number of chunks to retrieve for each query
- Using different documents as the knowledge base
- Customizing the system prompts for both agents
- Switching between ChromaDB and Qdrant by changing the `VECTOR_DB_TYPE` environment variable

`CHUNK_SIZE` and `CHUNK_OVERLAP` are measured in characters. Chunks stay within
`CHUNK_SIZE`, including separators, and carry up to `CHUNK_OVERLAP` trailing
characters into the next chunk. The splitter prefers paragraph, line and word
boundaries, and splits long paragraphs or words when necessary. Boundary
whitespace is trimmed. Use a positive chunk size and `0 <= CHUNK_OVERLAP < CHUNK_SIZE`.

## Example Usage

The chatbot can answer questions about the loaded document, such as:

- "What were the main points about the economy?"
- "What did the president say about healthcare?"
- "How did he address foreign policy?"

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.
