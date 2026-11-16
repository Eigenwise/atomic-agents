"""
Bridge module connecting DSPy's optimization framework with Atomic Agents' structured outputs.

This module provides the core integration that allows:
1. Using Pydantic schemas as DSPy signatures
2. Wrapping Atomic Agents as DSPy modules for optimization
3. Applying DSPy optimizers (BootstrapFewShot, MIPROv2, etc.) to improve agent performance
"""

from typing import Any, Dict, List, Literal, Optional, Type, get_args, get_origin

import dspy
from pydantic import BaseModel

from atomic_agents.base.base_io_schema import BaseIOSchema


def python_type_to_dspy_type(python_type: Any) -> Any:
    """
    Convert Python/Pydantic types to DSPy-compatible type annotations.

    Args:
        python_type: The Python type to convert

    Returns:
        A DSPy-compatible type annotation
    """
    origin = get_origin(python_type)

    # Handle Literal types
    if origin is Literal:
        return python_type

    # Handle List types
    if origin is list:
        args = get_args(python_type)
        if args:
            return list[python_type_to_dspy_type(args[0])]
        return list

    # Handle Optional types
    if origin is type(None) or (hasattr(origin, "__origin__") and origin.__origin__ is type(None)):
        return python_type

    # Handle Union types (including Optional)
    if hasattr(origin, "__name__") and origin.__name__ == "UnionType":
        args = get_args(python_type)
        # Filter out NoneType for Optional handling
        non_none_args = [a for a in args if a is not type(None)]
        if len(non_none_args) == 1:
            return python_type_to_dspy_type(non_none_args[0])
        return python_type

    # Basic types pass through
    if python_type in (str, int, float, bool, list, dict):
        return python_type

    return str  # Default to string for complex types


def pydantic_to_dspy_fields(schema: Type[BaseModel], field_type: str = "input") -> Dict[str, tuple]:
    """
    Convert Pydantic schema fields to DSPy field definitions.

    Args:
        schema: A Pydantic BaseModel class
        field_type: Either "input" or "output" to determine DSPy field type

    Returns:
        Dictionary mapping field names to (DSPyField, type) tuples
    """
    fields = {}

    for field_name, field_info in schema.model_fields.items():
        description = field_info.description or f"{field_name} field"

        # Get the field's Python type
        field_annotation = field_info.annotation
        dspy_type = python_type_to_dspy_type(field_annotation)

        # Create DSPy field
        if field_type == "input":
            dspy_field = dspy.InputField(desc=description)
        else:
            dspy_field = dspy.OutputField(desc=description)

        fields[field_name] = (dspy_field, dspy_type)

    return fields


def create_dspy_signature_from_schemas(
    input_schema: Type[BaseIOSchema],
    output_schema: Type[BaseIOSchema],
    instructions: Optional[str] = None,
) -> Type[dspy.Signature]:
    """
    Create a DSPy Signature class from Pydantic input/output schemas.

    This bridges Atomic Agents' schema-first design with DSPy's signature system,
    enabling optimization of prompts while maintaining type safety.

    Args:
        input_schema: Pydantic schema for inputs
        output_schema: Pydantic schema for outputs
        instructions: Optional task instructions for the signature

    Returns:
        A DSPy Signature class that can be used with DSPy modules
    """
    # Build field definitions
    field_definitions = {}

    # Add input fields
    input_fields = pydantic_to_dspy_fields(input_schema, "input")
    for name, (field, field_type) in input_fields.items():
        field_definitions[name] = (field_type, field)

    # Add output fields
    output_fields = pydantic_to_dspy_fields(output_schema, "output")
    for name, (field, field_type) in output_fields.items():
        field_definitions[name] = (field_type, field)

    # Generate instructions from schema docstrings if not provided
    if instructions is None:
        input_desc = input_schema.__doc__ or "Process the input"
        output_desc = output_schema.__doc__ or "Generate the output"
        instructions = f"{input_desc.strip()} {output_desc.strip()}"

    # Create the signature class dynamically
    signature_class = dspy.Signature(field_definitions, instructions)

    return signature_class


class DSPyAtomicModule(dspy.Module):
    """
    A DSPy module that bridges Atomic Agents schemas with DSPy's optimization framework.

    This module allows you to:
    1. Define tasks using Pydantic schemas (Atomic Agents style)
    2. Optimize prompts using DSPy optimizers (BootstrapFewShot, MIPROv2, etc.)
    3. Get type-safe structured outputs validated by Pydantic

    Example:
        ```python
        module = DSPyAtomicModule(
            input_schema=SentimentInputSchema,
            output_schema=SentimentOutputSchema,
            use_chain_of_thought=True
        )

        # Use directly
        result = module(text="I love this product!")

        # Or optimize with DSPy
        optimizer = dspy.BootstrapFewShot(metric=my_metric)
        optimized = optimizer.compile(module, trainset=examples)
        ```
    """

    def __init__(
        self,
        input_schema: Type[BaseIOSchema],
        output_schema: Type[BaseIOSchema],
        instructions: Optional[str] = None,
        use_chain_of_thought: bool = True,
    ):
        """
        Initialize the DSPy-Atomic bridge module.

        Args:
            input_schema: Pydantic schema class for input validation
            output_schema: Pydantic schema class for output structure
            instructions: Optional custom instructions for the task
            use_chain_of_thought: Whether to use ChainOfThought (recommended for complex tasks)
        """
        super().__init__()

        self.input_schema = input_schema
        self.output_schema = output_schema

        # Create DSPy signature from schemas
        self.signature = create_dspy_signature_from_schemas(input_schema, output_schema, instructions)

        # Create the predictor
        if use_chain_of_thought:
            self.predictor = dspy.ChainOfThought(self.signature)
        else:
            self.predictor = dspy.Predict(self.signature)

    def forward(self, **kwargs) -> dspy.Prediction:
        """
        Execute the module with given inputs.

        Args:
            **kwargs: Input fields matching the input_schema

        Returns:
            DSPy Prediction object with validated outputs
        """
        # Validate inputs using Pydantic schema
        try:
            validated_input = self.input_schema(**kwargs)
            # Convert back to dict for DSPy
            input_dict = validated_input.model_dump()
        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")

        # Run prediction
        prediction = self.predictor(**input_dict)

        return prediction

    def run_validated(self, **kwargs) -> BaseIOSchema:
        """
        Execute and return a validated Pydantic output schema instance.

        This provides the full type-safety of Atomic Agents while leveraging
        DSPy's optimization capabilities.

        Args:
            **kwargs: Input fields matching the input_schema

        Returns:
            Validated output schema instance
        """
        # Call self() which invokes __call__ -> forward properly
        prediction = self(**kwargs)

        # Extract output fields from prediction
        prediction_fields = dict(prediction.items())
        output_dict = {name: prediction_fields[name] for name in self.output_schema.model_fields if name in prediction_fields}

        # Validate and return as Pydantic model
        return self.output_schema(**output_dict)


class DSPyAtomicPipeline(dspy.Module):
    """
    A pipeline module that chains multiple DSPyAtomicModules together.

    This enables building complex multi-step workflows that can be
    optimized end-to-end by DSPy.

    Example:
        ```python
        pipeline = DSPyAtomicPipeline([
            ("extract", extraction_module),
            ("analyze", analysis_module),
            ("summarize", summary_module),
        ])

        # Optimize entire pipeline
        optimized = optimizer.compile(pipeline, trainset=examples)
        ```
    """

    def __init__(self, steps: List[tuple]):
        """
        Initialize the pipeline with named steps.

        Args:
            steps: List of (name, DSPyAtomicModule) tuples
        """
        super().__init__()
        self.step_names = []

        for name, module in steps:
            self.step_names.append(name)
            setattr(self, name, module)

    def forward(self, **kwargs) -> Dict[str, Any]:
        """
        Execute all pipeline steps in sequence.

        Args:
            **kwargs: Initial inputs for the first step

        Returns:
            Dictionary with results from each step
        """
        results = {}
        current_input = kwargs

        for name in self.step_names:
            module = getattr(self, name)
            prediction = module(**current_input)
            results[name] = prediction

            # Prepare input for next step (using all prediction fields)
            current_input = dict(prediction.items())

        return results


def create_dspy_example(
    input_schema: Type[BaseIOSchema],
    output_schema: Type[BaseIOSchema],
    input_data: Dict[str, Any],
    output_data: Dict[str, Any],
) -> dspy.Example:
    """
    Create a DSPy Example from Pydantic schema instances.

    This is useful for creating training sets for optimization.

    Args:
        input_schema: Input schema class for validation
        output_schema: Output schema class for validation
        input_data: Dictionary of input values
        output_data: Dictionary of expected output values

    Returns:
        A DSPy Example that can be used for training
    """
    # Validate data
    validated_input = input_schema(**input_data)
    validated_output = output_schema(**output_data)

    # Combine into single dict
    example_data = {
        **validated_input.model_dump(),
        **validated_output.model_dump(),
    }

    # Create DSPy example with input fields marked
    example = dspy.Example(**example_data).with_inputs(*list(input_schema.model_fields.keys()))

    return example
