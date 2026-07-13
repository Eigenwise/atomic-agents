# Web Search Agent

This project demonstrates an intelligent web search agent built using the Atomic Agents framework. The agent can perform web searches, generate relevant queries, and provide detailed answers to user questions based on the search results.

## Features

1. Query Generation: Automatically generates relevant search queries based on user input.
2. Web Search: Utilizes SearXNG to perform web searches across multiple search engines.
3. Question Answering: Provides detailed answers to user questions based on search results.
4. Follow-up Questions: Suggests related questions to encourage further exploration of the topic.

## Components

The Web Search Agent consists of several key components:

1. Query Agent (`query_agent.py`): Generates diverse and relevant search queries based on user input.
2. SearXNG Search Tool (`searxng_search.py`): Performs web searches using the SearXNG meta-search engine.
3. Question Answering Agent (`question_answering_agent.py`): Analyzes search results and provides detailed answers to user questions.
4. Main Script (`main.py`): Orchestrates the entire process, from query generation to final answer presentation.

## Getting Started

To run the Web Search Agent:

1. Setting up SearXNG server if you haven't:

Make sure to add these lines to `settings.tml`:

   ```yaml
   search:
   formats:
      - html
      - json
   ```

1. Clone the Atomic Agents repository:

   ```bash
   git clone https://github.com/eigenwise/atomic-agents
   ```

1. Navigate to the web-search-agent directory:

   ```bash
   cd atomic-agents/atomic-examples/web-search-agent
   ```

1. Install dependencies using uv:

   ```bash
   uv sync
   ```

1. Set up environment variables:
   Create a `.env` file in the `web-search-agent` directory with the following content:

   ```bash
   OPENAI_API_KEY=your_openai_api_key
   SEARXNG_BASE_URL=your_searxng_instance_url
   ```

   Replace `your_openai_api_key` with your actual OpenAI API key and `your_searxng_instance_url` with the URL of your SearXNG instance.
   If you do not have a SearxNG instance, see the instructions below to set up one locally with docker.

2. Run the Web Search Agent:

   ```bash
   uv run python web_search_agent/main.py
   ```

## How It Works

1. The user provides an initial question or topic for research.
2. The Query Agent generates multiple relevant search queries based on the user's input.
3. The SearXNG Search Tool performs web searches using the generated queries.
4. The Question Answering Agent analyzes the search results and formulates a detailed answer.
5. The main script presents the answer, along with references and follow-up questions.

## SearxNG Setup with docker

From the [official instructions](https://docs.searxng.org/admin/installation-docker.html):

```shell
mkdir my-instance
cd my-instance
export PORT=8080
docker pull searxng/searxng
docker run --rm \
           -d -p ${PORT}:8080 \
           -v "${PWD}/searxng:/etc/searxng" \
           -e "BASE_URL=http://localhost:$PORT/" \
           -e "INSTANCE_NAME=my-instance" \
           searxng/searxng
```

Set the `SEARXNG_BASE_URL` environment variable to `http://localhost:8080/` in your `.env` file.


Note: for the agent to communicate with SearxNG, the instance must enable the JSON engine, which is disabled by default.
Edit `/etc/searxng/settings.yml` and add `- json` in the `search.formats` section, then restart the container.


## Customization

You can customize the Web Search Agent by modifying the following:

- Adjust the number of generated queries in `main.py`.
- Modify the search categories or parameters in `searxng_search.py`.
- Customize the system prompts for the Query Agent and Question Answering Agent in their respective files.

## Contributing

Contributions to the Web Search Agent project are welcome! Please fork the repository and submit a pull request with your enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](../../LICENSE) file for details.

