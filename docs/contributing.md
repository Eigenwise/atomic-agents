# Contributing Guide

Thank you for your interest in contributing to Atomic Agents! This guide will help you get started with contributing to the project.

## Ways to Contribute

There are many ways to contribute to Atomic Agents:

1. **Report Bugs**: Submit bug reports on our [Issue Tracker](https://github.com/eigenwise/atomic-agents/issues)
2. **Suggest Features**: Share your ideas for new features or improvements
3. **Improve Documentation**: Help us make the documentation clearer and more comprehensive
4. **Submit Code**: Fix bugs, add features, or create new tools
5. **Share Examples**: Create example projects that showcase different use cases
6. **Write Tests**: Help improve our test coverage and reliability

## Development Setup

1. Fork and clone the repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/atomic-agents.git
   cd atomic-agents
   ```

2. Install dependencies with uv:
   ```bash
   uv sync
   ```

   To install all workspace packages (examples and tools):
   ```bash
   uv sync --all-packages
   ```

3. Set up pre-commit hooks:
   ```bash
   pre-commit install
   ```

4. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Code Style

We follow these coding standards:

- Use [Black](https://black.readthedocs.io/) for code formatting
- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Write docstrings in [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Add type hints to function signatures
- Keep functions focused and modular
- Write clear commit messages

## Creating Tools

When creating new tools:

1. Create a Python module manually, for example `my_tool.py`. The `atomic` command
   opens the tool browser and downloader; it does not scaffold new tools.

2. Define input and output schemas with `BaseIOSchema` and pass them as type
   parameters to `BaseTool`. Give each schema a non-empty docstring and describe
   its fields. Use `BaseToolConfig` to override the tool name or description;
   otherwise they come from the input schema's title and docstring.

   ```python
   from pydantic import Field

   from atomic_agents import BaseIOSchema, BaseTool


   class MyToolInput(BaseIOSchema):
       """An integer to double."""

       value: int = Field(..., description="Integer to double.")


   class MyToolOutput(BaseIOSchema):
       """The doubled integer."""

       doubled: int = Field(..., description="Twice the input value.")


   class MyTool(BaseTool[MyToolInput, MyToolOutput]):
       def run(self, params: MyToolInput) -> MyToolOutput:
           return MyToolOutput(doubled=params.value * 2)
   ```

3. Add tests for behavior and the public tool metadata, for example in
   `test_my_tool.py` alongside `my_tool.py`:

   ```python
   from atomic_agents import BaseToolConfig

   from my_tool import MyTool, MyToolInput, MyToolOutput


   def test_my_tool():
       tool = MyTool(config=BaseToolConfig(title="my_tool", description="Double an integer."))
       result = tool.run(MyToolInput(value=3))

       assert isinstance(result, MyToolOutput)
       assert result.doubled == 6
       assert tool.input_schema is MyToolInput
       assert tool.output_schema is MyToolOutput
       assert tool.tool_name == "my_tool"
       assert tool.tool_description == "Double an integer."
   ```

   Run the example test with `uv run pytest test_my_tool.py` and add cases for
   your tool's edge cases and errors.

4. Document your tool:
   - Add a README.md with usage examples
   - Include configuration instructions
   - Document any dependencies
   - Explain error handling

## Testing

Run tests with pytest:

```bash
uv run pytest
```

Include tests for:
- Normal operation
- Edge cases
- Error conditions
- Async functionality
- Integration with other components

## Documentation

When adding documentation:

1. Follow the existing structure
2. Include code examples
3. Add type hints and docstrings
4. Update relevant guides
5. Build and verify locally:
   ```bash
   cd docs
   uv run sphinx-build -b html . _build/html
   ```

## Submitting Changes

1. Commit your changes:
   ```bash
   git add .
   git commit -m "feat: add new feature"
   ```

2. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

3. Create a Pull Request:
   - Describe your changes
   - Reference any related issues
   - Include test results
   - Add documentation updates

## Getting Help

If you need help:

- Join our [Reddit community](https://www.reddit.com/r/AtomicAgents/)
- Check the [documentation](https://atomic-agents.readthedocs.io/)
- Ask questions on [GitHub Discussions](https://github.com/eigenwise/atomic-agents/discussions)

## Code of Conduct

Please note that this project is released with a Code of Conduct. By participating in this project you agree to abide by its terms. You can find the full text in our [GitHub repository](https://github.com/eigenwise/atomic-agents/blob/main/CODE_OF_CONDUCT.md).
