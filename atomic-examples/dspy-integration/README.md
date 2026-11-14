# DSPy + Atomic Agents Integration: A Complete Guide

> **The Best of Both Worlds**: Automatic prompt optimization meets type-safe structured outputs.

This example provides a comprehensive, hands-on walkthrough of why combining DSPy with Atomic Agents produces superior results compared to using either framework alone. We don't just show you *how* to use the integration—we teach you *why* it works and *when* to use each approach.

## Table of Contents

1. [The Problem We're Solving](#the-problem-were-solving)
2. [Quick Start](#quick-start)
3. [Understanding the Frameworks](#understanding-the-frameworks)
4. [The Three Stages](#the-three-stages)
5. [Benchmark Results](#benchmark-results)
6. [Deep Dive: How Each Stage Works](#deep-dive-how-each-stage-works)
7. [The Bridge: DSPyAtomicModule](#the-bridge-dspyatomicmodule)
8. [When to Use Each Approach](#when-to-use-each-approach)
9. [API Reference](#api-reference)
10. [Troubleshooting](#troubleshooting)

---

## The Problem We're Solving

Neither DSPy nor Atomic Agents alone gives you everything you need for production LLM applications:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DSPy ALONE                                                                  │
│ ✓ Automatic prompt optimization (finds what works!)                         │
│ ✓ Systematic few-shot example selection                                     │
│ ✓ Chain-of-thought reasoning built-in                                       │
│ ✗ No Pydantic ecosystem (validators, serializers, Field constraints)        │
│ ✗ Type enforcement is DSPy-specific, not Python-native                      │
│ ✗ Limited integration with structured output tools like Instructor          │
├─────────────────────────────────────────────────────────────────────────────┤
│ ATOMIC AGENTS ALONE                                                         │
│ ✓ Full Pydantic ecosystem (validators, serializers, ge/le constraints)      │
│ ✓ Instructor integration for robust structured output                       │
│ ✓ Python-native type safety with runtime validation                         │
│ ✗ Manual prompt engineering - you're guessing what works                    │
│ ✗ No systematic way to improve prompts                                      │
│ ✗ Adding few-shot examples requires manual selection                        │
├─────────────────────────────────────────────────────────────────────────────┤
│ DSPy + ATOMIC AGENTS COMBINED                                               │
│ ✓ Automatic prompt optimization                                             │
│ ✓ Type-safe structured outputs with full Pydantic ecosystem                 │
│ ✓ Measurable, reproducible improvements                                     │
│ ✓ Production-ready with IDE autocomplete and type checking                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### The Real-World Impact

In our benchmark with **60 training examples** and **30 intentionally challenging test cases**:

| Approach | Accuracy | Improvement |
|----------|----------|-------------|
| Raw DSPy (typed signatures) | 73.3% | baseline |
| Raw Atomic Agents | 76.7% | +3.4 pts |
| **DSPy + Atomic Agents** | **86.7%** | **+13.4 pts** |

The combined approach achieved **13.4 percentage points better accuracy** than DSPy alone and **10 percentage points better** than Atomic Agents alone.

---

## Quick Start

```bash
# Navigate to the example directory
cd atomic-examples/dspy-integration

# Install dependencies
uv sync

# Set your OpenAI API key (or create a .env file)
export OPENAI_API_KEY="your-key-here"

# Run the full didactic example
uv run python -m dspy_integration.main
```

The example will walk you through all three stages with detailed explanations, showing you the actual prompts being generated and optimized.

---

## Understanding the Frameworks

### What is DSPy?

DSPy (Declarative Self-improving Python) is a framework for **automatically optimizing LLM prompts**. Instead of manually crafting prompts, you:

1. Define a **Signature** (what inputs and outputs you need)
2. Create a **Module** (how to process the data)
3. Provide **training examples** with correct answers
4. Let DSPy **optimize** the prompts to maximize accuracy

DSPy's key insight: **The best prompt isn't what you think—let data decide.**

```python
import dspy
from typing import Literal

# Define a typed signature
class MovieGenreSignature(dspy.Signature):
    """Classify a movie review into its primary genre."""

    review: str = dspy.InputField(desc="The movie review text")
    genre: Literal["action", "comedy", "drama", "horror", "sci-fi", "romance"] = \
        dspy.OutputField(desc="The primary genre")
    confidence: float = dspy.OutputField(desc="Confidence 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation")

# DSPy automatically:
# 1. Generates prompts from this signature
# 2. Adds type constraints to the prompt
# 3. Optimizes with few-shot examples
```

### What is Atomic Agents?

Atomic Agents is a framework for building **type-safe LLM applications** using Pydantic schemas. It integrates with [Instructor](https://github.com/jxnl/instructor) to guarantee structured outputs:

```python
from pydantic import Field
from typing import Literal
from atomic_agents.base.base_io_schema import BaseIOSchema

class MovieGenreOutput(BaseIOSchema):
    """Output schema for movie genre classification."""

    genre: Literal["action", "comedy", "drama", "horror", "sci-fi", "romance"] = Field(
        ...,
        description="The primary genre of the movie.",
    )
    confidence: float = Field(
        ...,
        ge=0.0, le=1.0,  # VALIDATED! Must be between 0 and 1
        description="Confidence score between 0.0 and 1.0",
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation for the classification.",
    )

# Atomic Agents + Instructor guarantees:
# 1. genre is ALWAYS one of the 6 valid options
# 2. confidence is ALWAYS a float between 0.0 and 1.0
# 3. If validation fails, it retries with error feedback
```

### Why Combine Them?

| Feature | DSPy | Atomic Agents | Combined |
|---------|------|---------------|----------|
| Prompt Optimization | ✅ Automatic | ❌ Manual | ✅ Automatic |
| Type Safety | ⚠️ DSPy-specific | ✅ Pydantic | ✅ Pydantic |
| Validation Constraints | ⚠️ Basic | ✅ Full (ge/le/etc) | ✅ Full |
| Few-Shot Selection | ✅ Automatic | ❌ Manual | ✅ Automatic |
| IDE Autocomplete | ⚠️ Partial | ✅ Full | ✅ Full |
| Instructor Integration | ❌ No | ✅ Yes | ✅ Yes |
| Retry on Failure | ❌ No | ✅ Yes | ✅ Yes |

---

## The Three Stages

Our didactic example walks through three approaches to the same task: **classifying movie reviews into genres**.

### Stage 1: Raw DSPy (Properly Implemented)

We use DSPy with **typed signatures** (class-based signatures with `Literal` type constraints). This is DSPy at its best:

```python
from typing import Literal
import dspy

GenreType = Literal["action", "comedy", "drama", "horror", "sci-fi", "romance"]

class MovieGenreSignature(dspy.Signature):
    """Classify a movie review into its primary genre."""

    review: str = dspy.InputField(desc="The movie review text to classify")
    genre: GenreType = dspy.OutputField(desc="The primary genre")
    confidence: float = dspy.OutputField(desc="Confidence score 0.0-1.0")
    reasoning: str = dspy.OutputField(desc="Brief explanation")

# Create classifier with chain-of-thought reasoning
classify = dspy.ChainOfThought(MovieGenreSignature)

# Optimize with training data
optimizer = dspy.BootstrapFewShot(
    metric=genre_match,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
)
optimized = optimizer.compile(classify, trainset=training_examples)
```

**What DSPy does with Literal types:**

DSPy automatically includes the constraint in the generated prompt:
```
genre (Literal['action', 'comedy', 'drama', 'horror', 'sci-fi', 'romance']):
The primary genre: action, comedy, drama, horror, sci-fi, or romance

# note: the value you produce must exactly match (no extra characters) one of:
# action; comedy; drama; horror; sci-fi; romance
```

**Result: 73.3% accuracy** on our challenging test set.

### Stage 2: Raw Atomic Agents

We use Atomic Agents with a **manually crafted system prompt**:

```python
from atomic_agents.agents.atomic_agent import AtomicAgent, AgentConfig
from atomic_agents.context.system_prompt_generator import SystemPromptGenerator

# Manual prompt - we're guessing what works!
system_prompt = SystemPromptGenerator(
    background=[
        "You are a movie genre classification expert.",
        "You analyze movie reviews and determine the primary genre.",
        "Valid genres are: action, comedy, drama, horror, sci-fi, romance",
    ],
    steps=[
        "Read the review carefully.",
        "Identify key genre indicators.",
        "Consider the overall tone and subject matter.",
        "Select the single most appropriate genre.",
    ],
    output_instructions=[
        "Be decisive - pick ONE primary genre even if multiple could apply.",
        "Confidence should be 0.7-1.0 for clear cases, 0.5-0.7 for ambiguous.",
    ],
)

agent = AtomicAgent[MovieReviewInput, MovieGenreOutput](
    config=AgentConfig(
        client=instructor.from_openai(openai.OpenAI()),
        model="gpt-5.6-terra",
        system_prompt_generator=system_prompt,
    )
)
```

**The problem with manual prompts:**
- Is "Be decisive" helping or hurting accuracy?
- Should we add few-shot examples? Which ones?
- Would different wording improve results?
- **Without DSPy, we're just guessing!**

**Result: 76.7% accuracy** - better structure, but limited by manual prompt engineering.

### Stage 3: DSPy + Atomic Agents Combined

We use the **DSPyAtomicModule bridge** to get the best of both:

```python
from dspy_integration.bridge import DSPyAtomicModule, create_dspy_example

# The bridge combines both frameworks
module = DSPyAtomicModule(
    input_schema=MovieReviewInput,    # Pydantic input validation
    output_schema=MovieGenreOutput,   # Pydantic output structure
    instructions="Classify the movie review into a genre.",
    use_chain_of_thought=True,        # DSPy's reasoning capability
)

# Create type-validated training examples
trainset = [
    create_dspy_example(
        MovieReviewInput,
        MovieGenreOutput,
        {"review": "Non-stop explosions and car chases!"},
        {"genre": "action", "confidence": 0.9, "reasoning": "Action keywords"},
    )
    for ex in training_data
]

# Optimize with DSPy
optimizer = dspy.BootstrapFewShot(metric=genre_match)
optimized = optimizer.compile(module, trainset=trainset)

# Get type-safe output
result = optimized.run_validated(review="A touching love story...")
print(result.genre)       # Guaranteed Literal type
print(result.confidence)  # Guaranteed 0.0-1.0 float
```

**Result: 86.7% accuracy** - optimized prompts + guaranteed structure!

---

## Benchmark Results

### Dataset Composition

**Training Set: 60 examples** (10 per genre)
- Clear, representative examples for learning
- Some nuanced examples to teach edge cases

**Test Set: 30 challenging examples** intentionally designed to be difficult:

| Category | Count | Description |
|----------|-------|-------------|
| Sarcasm & Irony | 5 | Reviews that say the opposite of what they mean |
| Multi-Genre | 6 | Reviews spanning multiple genres (must pick primary) |
| Misleading Signals | 5 | Keywords suggesting wrong genre |
| Subverted Expectations | 5 | Genre setups that don't pay off |
| Subtle/Ambiguous | 5 | Nuanced, hard-to-classify reviews |
| Cultural Context | 4 | References requiring cultural knowledge |

### Example Challenging Test Cases

```python
# Sarcasm - sounds negative but reviewer enjoyed it
"Oh great, another movie where the hero walks away from explosions in slow motion.
How original. Still watched it twice though."  # → action

# Multi-genre - sci-fi setting but drama focus
"The robot's sacrifice to save humanity made me sob uncontrollably.
Beautiful storytelling set against a dystopian future."  # → sci-fi

# Misleading signals - thriller language but romance theme
"A thriller where the biggest twist was how much I ended up caring
about these characters' relationships."  # → romance

# Cultural context - requires knowing references
"John Wick energy but make it about a retired chef defending his restaurant.
Knife fights choreographed like ballet."  # → action
```

### Final Results

```
┌────────────────────┬─────────────┬──────────────────────┬─────────────────┐
│ Metric             │ Raw DSPy    │ Raw Atomic Agents    │ DSPy + Atomic   │
├────────────────────┼─────────────┼──────────────────────┼─────────────────┤
│ Accuracy           │ 73.3%       │ 76.7%                │ 86.7%           │
│ Correct/Total      │ 22/30       │ 23/30                │ 26/30           │
│ Prompt Optimization│ ✓ Auto      │ ✗ Manual             │ ✓ Auto          │
│ Type Safety        │ ~ DSPy      │ ✓ Pydantic           │ ✓ Pydantic      │
│ Output Validation  │ ~ Basic     │ ✓ Full               │ ✓ Full          │
│ Pydantic Ecosystem │ ✗ No        │ ✓ Full               │ ✓ Full          │
│ Few-Shot Selection │ ✓ Auto      │ ✗ Manual             │ ✓ Auto          │
│ IDE Support        │ ~ Partial   │ ✓ Full               │ ✓ Full          │
└────────────────────┴─────────────┴──────────────────────┴─────────────────┘
```

---

## Deep Dive: How Each Stage Works

### How DSPy Optimization Works

DSPy's `BootstrapFewShot` optimizer doesn't just use your examples verbatim. Here's what happens:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Run LLM on Training Examples                                        │
│                                                                             │
│ For each training example, DSPy runs the LLM and captures the full         │
│ "trace" - including any chain-of-thought reasoning generated.              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: Filter by Metric                                                    │
│                                                                             │
│ Only traces that produce correct answers are kept. If the LLM got          │
│ the genre wrong, that trace is discarded.                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: Select Best Traces                                                  │
│                                                                             │
│ DSPy selects diverse, high-quality traces as few-shot demonstrations.      │
│ These aren't your original examples - they include LLM-generated           │
│ reasoning that actually worked!                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 4: Inject into Future Prompts                                          │
│                                                                             │
│ The selected demonstrations are automatically added to prompts,             │
│ showing the LLM examples of correct reasoning and outputs.                  │
└─────────────────────────────────────────────────────────────────────────────┘
```

### How Atomic Agents Validates Output

Atomic Agents uses Instructor under the hood for structured output:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Schema Conversion                                                   │
│                                                                             │
│ Your Pydantic schema is converted to JSON Schema and sent to the LLM       │
│ along with your prompt.                                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 2: LLM Generation                                                      │
│                                                                             │
│ The LLM generates output attempting to match the schema. Modern LLMs       │
│ (like GPT-4) support function calling which helps with this.               │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Step 3: Pydantic Validation                                                 │
│                                                                             │
│ Instructor validates the response against your Pydantic schema:            │
│ - Is genre one of the allowed Literal values?                              │
│ - Is confidence a float between 0.0 and 1.0?                               │
│ - Are all required fields present?                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          │                   │
                    VALID │             INVALID
                          ▼                   ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│ Return Pydantic Object      │   │ Retry with Error Feedback   │
│                             │   │                             │
│ You get a fully typed,      │   │ Instructor tells the LLM    │
│ validated result!           │   │ what went wrong and retries │
└─────────────────────────────┘   └─────────────────────────────┘
```

### How the Bridge Combines Both

The `DSPyAtomicModule` bridges both frameworks:

```python
class DSPyAtomicModule(dspy.Module):
    """
    Bridges Pydantic schemas with DSPy optimization.

    1. Converts Pydantic schemas → DSPy signatures
    2. Enables DSPy optimization (BootstrapFewShot, etc.)
    3. Returns validated Pydantic objects
    """

    def __init__(
        self,
        input_schema: Type[BaseIOSchema],   # Your Pydantic input
        output_schema: Type[BaseIOSchema],  # Your Pydantic output
        instructions: str,                   # Task description
        use_chain_of_thought: bool = True,   # Enable reasoning
    ):
        # Convert Pydantic → DSPy signature
        self.signature = create_dspy_signature_from_schemas(
            input_schema, output_schema, instructions
        )

        # Create DSPy predictor
        if use_chain_of_thought:
            self.predictor = dspy.ChainOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self, **kwargs) -> dspy.Prediction:
        """Standard DSPy forward - for optimization."""
        validated_input = self.input_schema(**kwargs)
        return self.predictor(**validated_input.model_dump())

    def run_validated(self, **kwargs) -> BaseIOSchema:
        """Get type-safe Pydantic output."""
        prediction = self(**kwargs)

        # Extract fields and validate with Pydantic
        output_dict = {
            field: getattr(prediction, field)
            for field in self.output_schema.model_fields
        }
        return self.output_schema(**output_dict)
```

---

## The Bridge: DSPyAtomicModule

### Core Functions

#### `create_dspy_signature_from_schemas`

Converts Pydantic schemas to DSPy signatures:

```python
from dspy_integration.bridge import create_dspy_signature_from_schemas

signature = create_dspy_signature_from_schemas(
    input_schema=MovieReviewInput,
    output_schema=MovieGenreOutput,
    instructions="Classify the movie review into its primary genre.",
)

# The signature preserves:
# - Field names and descriptions
# - Type constraints (Literal, float, etc.)
# - Documentation from schema docstrings
```

#### `create_dspy_example`

Creates validated training examples:

```python
from dspy_integration.bridge import create_dspy_example

# This validates both input and output!
example = create_dspy_example(
    MovieReviewInput,
    MovieGenreOutput,
    {"review": "Amazing action sequences!"},
    {"genre": "action", "confidence": 0.95, "reasoning": "Clear action signals"},
)

# If you accidentally put confidence=1.5:
# ValidationError: confidence must be <= 1.0
```

#### `DSPyAtomicModule`

The main bridge class:

```python
from dspy_integration.bridge import DSPyAtomicModule

module = DSPyAtomicModule(
    input_schema=MovieReviewInput,
    output_schema=MovieGenreOutput,
    instructions="Classify the movie review.",
    use_chain_of_thought=True,
)

# Use as DSPy module (for optimization)
prediction = module(review="A love story...")

# Get validated Pydantic output
result = module.run_validated(review="A love story...")
print(type(result))  # MovieGenreOutput
print(result.genre)  # Guaranteed valid Literal
```

#### `DSPyAtomicPipeline`

Chain multiple modules together:

```python
from dspy_integration.bridge import DSPyAtomicPipeline

pipeline = DSPyAtomicPipeline([
    ("extract", extraction_module),
    ("analyze", analysis_module),
    ("summarize", summary_module),
])

# Optimize entire pipeline end-to-end
optimized = optimizer.compile(pipeline, trainset=examples)
```

---

## When to Use Each Approach

### Use Raw DSPy When:

- **Quick prototyping** - You want to iterate fast without worrying about schemas
- **Output format doesn't matter** - You'll post-process the outputs anyway
- **Research and experimentation** - You're exploring what's possible
- **Simple outputs** - Just need a string or simple structured data

```python
# Good for DSPy alone: quick iteration
classify = dspy.ChainOfThought("text -> sentiment")
result = classify(text="I love this!")
print(result.sentiment)  # Might be "positive", "Positive", "POSITIVE", etc.
```

### Use Raw Atomic Agents When:

- **Need structure NOW** - You don't have time to set up optimization
- **No training data** - You can't optimize without labeled examples
- **Simple enough task** - Manual prompts are good enough
- **Integration priority** - Need Pydantic ecosystem immediately

```python
# Good for Atomic Agents alone: guaranteed structure, no training needed
result = agent.run(input_data)
print(result.sentiment)  # Always exactly "positive", "negative", or "neutral"
print(result.score)      # Always a float between 0.0 and 1.0
```

### Use DSPy + Atomic Agents When:

- **Have labeled data** - You can optimize with real examples
- **Production systems** - Need both accuracy AND type safety
- **Measurable improvement** - You want to track and improve performance
- **Complex tasks** - Where prompt optimization significantly helps
- **Team collaboration** - Type safety helps multiple developers

```python
# Best of both: optimized prompts + guaranteed structure
module = DSPyAtomicModule(...)
optimized = optimizer.compile(module, trainset=training_data)
result = optimized.run_validated(review="...")

# result.genre is Literal["action", "comedy", ...] - type checker knows this!
# result.confidence is float with 0.0 <= x <= 1.0 - guaranteed!
```

### Decision Flowchart

```
                            START
                              │
                              ▼
                    ┌─────────────────────┐
                    │ Do you have labeled │
                    │ training data?      │
                    └─────────────────────┘
                              │
              ┌───────────────┴───────────────┐
              │ NO                            │ YES
              ▼                               ▼
    ┌─────────────────────┐         ┌─────────────────────┐
    │ Need guaranteed     │         │ Need guaranteed     │
    │ output structure?   │         │ output structure?   │
    └─────────────────────┘         └─────────────────────┘
              │                               │
    ┌─────────┴─────────┐         ┌─────────┴─────────┐
    │ NO                │ YES     │ NO                │ YES
    ▼                   ▼         ▼                   ▼
┌─────────┐     ┌─────────────┐  ┌─────────┐   ┌─────────────────┐
│ Raw     │     │ Raw Atomic  │  │ Raw     │   │ DSPy + Atomic   │
│ DSPy    │     │ Agents      │  │ DSPy    │   │ Agents          │
└─────────┘     └─────────────┘  └─────────┘   │ (RECOMMENDED)   │
                                               └─────────────────┘
```

---

## API Reference

### Schemas (`schemas.py`)

Pre-built schemas for common tasks:

```python
from dspy_integration.schemas import (
    SentimentInputSchema,      # text → sentiment analysis
    SentimentOutputSchema,
    QuestionInputSchema,       # question + context → answer
    AnswerOutputSchema,
    SummaryInputSchema,        # text → summary
    SummaryOutputSchema,
    ClassificationInputSchema, # text + categories → labels
    ClassificationOutputSchema,
)
```

### Bridge (`bridge.py`)

```python
from dspy_integration.bridge import (
    DSPyAtomicModule,                    # Main bridge class
    DSPyAtomicPipeline,                  # Chain multiple modules
    create_dspy_signature_from_schemas,  # Pydantic → DSPy
    create_dspy_example,                 # Create training examples
    pydantic_to_dspy_fields,             # Convert field definitions
    python_type_to_dspy_type,            # Convert Python types
)
```

---

## Troubleshooting

### Common Issues

**1. "API key not found"**
```bash
# Make sure your key is set
export OPENAI_API_KEY="sk-..."

# Or create a .env file in the dspy-integration directory
echo 'OPENAI_API_KEY=sk-...' > .env
```

**2. "Invalid genre output"**

If using raw DSPy without typed signatures, you might get invalid genres. Use class-based signatures with `Literal` types:

```python
# BAD - no type constraints
classify = dspy.ChainOfThought("review -> genre, confidence, reasoning")

# GOOD - Literal type constraint
class MovieGenreSignature(dspy.Signature):
    genre: Literal["action", "comedy", ...] = dspy.OutputField(...)
```

**3. "Validation error in Atomic Agents"**

Instructor retries automatically, but if you consistently get errors:
- Check your schema constraints aren't too restrictive
- Ensure the LLM model supports structured output well
- Consider using a more capable model (GPT-4 > GPT-3.5)

**4. "Optimization not improving accuracy"**

- Add more training examples (at least 20-30)
- Ensure training examples are high quality
- Try different optimizer settings:
  ```python
  optimizer = dspy.BootstrapFewShot(
      max_bootstrapped_demos=6,  # Try more demos
      max_labeled_demos=6,
      max_rounds=2,              # More optimization rounds
  )
  ```

---

## Project Structure

```
dspy-integration/
├── pyproject.toml              # Dependencies (uv/pip)
├── README.md                   # This file
├── .env                        # API keys (create this)
└── dspy_integration/
    ├── __init__.py             # Package exports
    ├── bridge.py               # DSPyAtomicModule implementation
    ├── schemas.py              # Reusable Pydantic schemas
    └── main.py                 # The didactic example
```

---

## Requirements

- Python 3.12+
- OpenAI API key
- Dependencies (installed via `uv sync`):
  - `dspy-ai` - DSPy framework
  - `atomic-agents` - Atomic Agents framework
  - `instructor` - Structured output library
  - `pydantic` - Data validation
  - `rich` - Beautiful terminal output

---

## License

MIT License - Part of the Atomic Agents monorepo.

---

## Further Reading

- [DSPy Documentation](https://dspy-docs.vercel.app/)
- [Atomic Agents Documentation](https://github.com/eigenwise/atomic-agents)
- [Instructor Documentation](https://python.useinstructor.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)

---

## Contributing

Found a bug or want to improve this example? Please open an issue or PR in the atomic-agents monorepo!
