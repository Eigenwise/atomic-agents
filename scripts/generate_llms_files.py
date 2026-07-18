#!/usr/bin/env python3
"""
Generate separate and combined llms.txt files:
- llms-docs.txt: Documentation only
- llms-source.txt: Source code only  
- llms-examples.txt: Examples only
- llms-full.txt: Everything combined
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# Define paths
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
ATOMIC_AGENTS_DIR = PROJECT_ROOT / "atomic-agents"
ATOMIC_EXAMPLES_DIR = PROJECT_ROOT / "atomic-examples"
SINGLEHTML_DIR = DOCS_DIR / "_build" / "singlehtml"

# Output files
OUTPUT_INDEX = DOCS_DIR / "llms.txt"
OUTPUT_DOCS = DOCS_DIR / "llms-docs.txt"
OUTPUT_SOURCE = DOCS_DIR / "llms-source.txt"
OUTPUT_EXAMPLES = DOCS_DIR / "llms-examples.txt"
OUTPUT_FULL = DOCS_DIR / "llms-full.txt"

SITE_URL = "https://eigenwise.github.io/atomic-agents"

# File extensions to include in source code
SOURCE_CODE_EXTENSIONS = {'.py', '.md', '.txt', '.yml', '.yaml', '.toml', '.cfg', '.ini', '.json'}

# Directories to exclude
EXCLUDE_DIRS = {'__pycache__', '.git', '.venv', 'venv', 'env', '.env', 'node_modules', 
                'build', 'dist', '.egg-info', '.pytest_cache', '.mypy_cache', '.ruff_cache'}

def create_section_divider(title: str, level: int = 1) -> str:
    """Create a clear section divider."""
    border = "=" * 80 if level == 1 else "-" * 80
    return f"\n\n{border}\n{title}\n{border}\n\n"

def get_files_to_include(directory: Path, extensions: set) -> List[Tuple[Path, str]]:
    """
    Get all files to include from a directory, respecting exclusions.
    Returns list of (file_path, relative_path_string) tuples.
    """
    files_to_include = []
    
    for root, dirs, files in os.walk(directory):
        # Remove excluded directories from dirs to prevent walking into them
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        
        root_path = Path(root)
        for file in files:
            file_path = root_path / file
            if file_path.suffix.lower() in extensions:
                relative_path = file_path.relative_to(directory)
                files_to_include.append((file_path, str(relative_path)))
    
    return sorted(files_to_include, key=lambda x: x[1])

def extract_documentation_from_html() -> str:
    """Extract documentation from built HTML."""
    html_file = SINGLEHTML_DIR / "index.html"
    
    if not html_file.exists():
        print("Warning: singlehtml output not found. Build with 'make singlehtml' first.")
        return "Documentation not yet built. Please run 'make singlehtml' in the docs directory.\n"
    
    with open(html_file, encoding="utf-8") as f:
        html = f.read()
    
    soup = BeautifulSoup(html, "html.parser")
    doc_main = soup.find("div", {"role": "main"}) or soup.find("div", class_="document")
    content_html = str(doc_main) if doc_main else html
    
    # Convert to Markdown
    md_text = md(content_html)
    return md_text

def process_source_file(file_path: Path, relative_path: str) -> str:
    """Process a single source file for inclusion."""
    output = f"\n### File: {relative_path}\n\n"
    
    try:
        with open(file_path, encoding="utf-8", errors='ignore') as f:
            content = f.read()
        
        # Add code fence for syntax highlighting
        if file_path.suffix == '.py':
            output += "```python\n"
        elif file_path.suffix in ['.yml', '.yaml']:
            output += "```yaml\n"
        elif file_path.suffix == '.json':
            output += "```json\n"
        elif file_path.suffix == '.toml':
            output += "```toml\n"
        elif file_path.suffix == '.md':
            output += ""  # Markdown doesn't need code fence
        else:
            output += "```\n"
        
        output += content
        
        if file_path.suffix != '.md':
            output += "\n```\n"
        
    except Exception as e:
        output += f"Error reading file: {e}\n"
    
    return output

def generate_documentation_only() -> str:
    """Generate documentation-only content."""
    content = create_section_divider("ATOMIC AGENTS DOCUMENTATION", 1)
    content += "This file contains the complete documentation for the Atomic Agents framework.\n"
    content += "Generated for use with Large Language Models and AI assistants.\n"
    content += f"Project Repository: https://github.com/eigenwise/atomic-agents\n"
    
    content += create_section_divider("DOCUMENTATION", 1)
    content += extract_documentation_from_html()
    
    content += create_section_divider("END OF DOCUMENT", 1)
    return content

def generate_source_code_only() -> str:
    """Generate source code-only content."""
    content = create_section_divider("ATOMIC AGENTS SOURCE CODE", 1)
    content += "This file contains the complete source code for the Atomic Agents framework.\n"
    content += "Generated for use with Large Language Models and AI assistants.\n"
    content += f"Project Repository: https://github.com/eigenwise/atomic-agents\n"
    
    files = get_files_to_include(ATOMIC_AGENTS_DIR, SOURCE_CODE_EXTENSIONS)
    
    for file_path, relative_path in files:
        content += process_source_file(file_path, f"atomic-agents/{relative_path}")
    
    content += create_section_divider("END OF DOCUMENT", 1)
    return content

def generate_examples_only() -> str:
    """Generate examples-only content."""
    content = create_section_divider("ATOMIC AGENTS EXAMPLES", 1)
    content += "This file contains all example implementations using the Atomic Agents framework.\n"
    content += "Each example includes its README documentation and complete source code.\n"
    content += f"Project Repository: https://github.com/eigenwise/atomic-agents\n"
    
    # Get all example directories
    example_dirs = [d for d in ATOMIC_EXAMPLES_DIR.iterdir() if d.is_dir() and d.name not in EXCLUDE_DIRS]
    example_dirs.sort()
    
    for example_dir in example_dirs:
        content += create_section_divider(f"Example: {example_dir.name}", 2)
        
        # Add GitHub link
        github_url = f"https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/{example_dir.name}"
        content += f"**View on GitHub:** {github_url}\n\n"
        
        # Process README first if it exists
        readme_files = list(example_dir.glob("README.*"))
        if readme_files:
            readme_file = readme_files[0]
            content += "## Documentation\n\n"
            try:
                with open(readme_file, encoding="utf-8", errors='ignore') as f:
                    content += f.read() + "\n\n"
            except Exception as e:
                content += f"Error reading README: {e}\n\n"
        
        # Process all source files in the example
        content += "## Source Code\n\n"
        files = get_files_to_include(example_dir, SOURCE_CODE_EXTENSIONS)
        
        for file_path, relative_path in files:
            # Skip README files as we've already processed them
            if not file_path.name.startswith("README"):
                content += process_source_file(file_path, f"atomic-examples/{example_dir.name}/{relative_path}")
    
    content += create_section_divider("END OF DOCUMENT", 1)
    return content

def generate_full_content() -> str:
    """Generate the comprehensive llms-full.txt content."""
    content = create_section_divider("ATOMIC AGENTS - COMPREHENSIVE DOCUMENTATION, SOURCE CODE, AND EXAMPLES", 1)
    content += "This file contains the complete documentation, source code, and examples for the Atomic Agents framework.\n"
    content += "Generated for use with Large Language Models and AI assistants.\n"
    content += f"Project Repository: https://github.com/eigenwise/atomic-agents\n"
    content += f"\nTable of Contents:\n"
    content += "1. Documentation\n"
    content += "2. Atomic Agents Source Code\n"
    content += "3. Atomic Examples\n"
    
    # Section 1: Documentation
    content += create_section_divider("DOCUMENTATION", 1)
    content += "This section contains the full documentation built from the docs folder.\n\n"
    content += extract_documentation_from_html()
    
    # Section 2: Source Code
    content += create_section_divider("ATOMIC AGENTS SOURCE CODE", 1)
    content += "This section contains the complete source code for the Atomic Agents framework.\n\n"
    
    files = get_files_to_include(ATOMIC_AGENTS_DIR, SOURCE_CODE_EXTENSIONS)
    for file_path, relative_path in files:
        content += process_source_file(file_path, f"atomic-agents/{relative_path}")
    
    # Section 3: Examples
    content += create_section_divider("ATOMIC EXAMPLES", 1)
    content += "This section contains all example implementations using the Atomic Agents framework.\n"
    content += "Each example includes its README documentation and complete source code.\n\n"
    
    example_dirs = [d for d in ATOMIC_EXAMPLES_DIR.iterdir() if d.is_dir() and d.name not in EXCLUDE_DIRS]
    example_dirs.sort()
    
    for example_dir in example_dirs:
        content += create_section_divider(f"Example: {example_dir.name}", 2)
        
        github_url = f"https://github.com/eigenwise/atomic-agents/tree/main/atomic-examples/{example_dir.name}"
        content += f"**View on GitHub:** {github_url}\n\n"
        
        readme_files = list(example_dir.glob("README.*"))
        if readme_files:
            readme_file = readme_files[0]
            content += "## Documentation\n\n"
            try:
                with open(readme_file, encoding="utf-8", errors='ignore') as f:
                    content += f.read() + "\n\n"
            except Exception as e:
                content += f"Error reading README: {e}\n\n"
        
        content += "## Source Code\n\n"
        files = get_files_to_include(example_dir, SOURCE_CODE_EXTENSIONS)
        
        for file_path, relative_path in files:
            if not file_path.name.startswith("README"):
                content += process_source_file(file_path, f"atomic-examples/{example_dir.name}/{relative_path}")
    
    content += create_section_divider("END OF DOCUMENT", 1)
    content += "This comprehensive documentation was generated for use with AI assistants and LLMs.\n"
    content += "For the latest version, please visit: https://github.com/eigenwise/atomic-agents\n"
    
    return content

def generate_index() -> str:
    """Generate the llms.txt index per the llmstxt.org spec: a small Markdown
    navigation file at the site root pointing to the full bundles and key pages."""
    return f"""# Atomic Agents

> Atomic Agents is a lightweight, modular Python framework for building agentic AI applications
> as composable, schema-driven building blocks. Built on Instructor and Pydantic: every agent is
> a typed transformer from an input schema to an output schema, with full developer control and
> no hidden abstractions. Provider-agnostic (OpenAI, Anthropic, Groq, Gemini, Ollama, and more).

Key API facts: import from the top-level `atomic_agents` package (`AtomicAgent`, `AgentConfig`,
`BaseIOSchema`, `BaseTool`) and from `atomic_agents.context` (`ChatHistory`,
`SystemPromptGenerator`, `BaseDynamicContextProvider`). Agents are constructed as
`AtomicAgent[InputSchema, OutputSchema](config=AgentConfig(...))` with an Instructor-wrapped
client. Every `BaseIOSchema` subclass requires a docstring. Python 3.12+.

## Documentation

- [Quickstart guide]({SITE_URL}/guides/quickstart.html): first agent, providers, streaming
- [Tools guide]({SITE_URL}/guides/tools.html): BaseTool, Atomic Forge, MCP interop
- [Orchestration guide]({SITE_URL}/guides/orchestration.html): multi-agent patterns
- [Memory guide]({SITE_URL}/guides/memory.html): ChatHistory, persistence, custom backends
- [Hooks guide]({SITE_URL}/guides/hooks.html): telemetry, retries, error handling
- [API reference]({SITE_URL}/api/index.html): full API documentation

## Bundles

- [llms-full.txt]({SITE_URL}/llms-full.txt): complete documentation, framework source code, and all examples in one file
- [llms-docs.txt]({SITE_URL}/llms-docs.txt): documentation only
- [llms-source.txt]({SITE_URL}/llms-source.txt): framework source code only
- [llms-examples.txt]({SITE_URL}/llms-examples.txt): runnable example projects only

## Optional

- [GitHub repository](https://github.com/eigenwise/atomic-agents): source, issues, discussions
- [Agent skills](https://github.com/eigenwise/atomic-agents/tree/main/claude-plugin/atomic-agents): installable skills for coding assistants (`npx skills add eigenwise/atomic-agents`)
- [PyPI package](https://pypi.org/project/atomic-agents/): `pip install atomic-agents`
"""


def write_to_file_and_copy(output_file: Path, content: str):
    """Write content to file and copy to HTML build directory if it exists."""
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Generated {output_file.name} ({len(content):,} characters)")
    
    # Also copy to the HTML build directory if it exists
    html_output = DOCS_DIR / "_build" / "html" / output_file.name
    if html_output.parent.exists():
        with open(html_output, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Copied to HTML build directory")

def main():
    """Generate all llms.txt files."""
    print("Generating llms.txt files...")
    print("-" * 40)
    
    # Generate documentation only
    print("Generating llms-docs.txt...")
    docs_content = generate_documentation_only()
    write_to_file_and_copy(OUTPUT_DOCS, docs_content)
    
    # Generate source code only
    print("\nGenerating llms-source.txt...")
    source_content = generate_source_code_only()
    write_to_file_and_copy(OUTPUT_SOURCE, source_content)
    
    # Generate examples only
    print("\nGenerating llms-examples.txt...")
    examples_content = generate_examples_only()
    write_to_file_and_copy(OUTPUT_EXAMPLES, examples_content)
    
    # Generate full combined file
    print("\nGenerating llms-full.txt...")
    full_content = generate_full_content()
    write_to_file_and_copy(OUTPUT_FULL, full_content)

    # Generate the spec-compliant index
    print("\nGenerating llms.txt index...")
    write_to_file_and_copy(OUTPUT_INDEX, generate_index())
    
    print("\n" + "=" * 40)
    print("Successfully generated all llms.txt files!")

if __name__ == "__main__":
    main()