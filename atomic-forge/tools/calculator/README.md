# Calculator Tool

## Overview
The Calculator Tool is a utility within the Atomic Agents ecosystem designed for performing a variety of mathematical calculations. It's essentially a wrapper around the sympy library to allow for expression-based calculations.

## Prerequisites and Dependencies
- Python 3.9 or later
- atomic-agents (See [here](/README.md) for installation instructions)
- pydantic
- sympy

## Installation
You can install the tool using any of the following options:

1. Using the CLI tool that comes with Atomic Agents. Simply run `atomic` and select the tool from the list of available tools. After doing so you will be asked for a target directory to download the tool into.
2. Good old fashioned copy/paste: Just like any other tool inside the Atomic Forge, you can copy the code from this repo directly into your own project, provided you already have atomic-agents installed according to the instructions in the main [README](/README.md).

## Input & Output Structure

### Input Schema
- `expression` (str): Mathematical expression to evaluate. For example, '2 + 2'.

### Output Schema
- `result` (str): Result of the calculation.

## Usage

Here's an example of how to use the Calculator Tool:

```python
from tool.calculator import CalculatorTool, CalculatorToolConfig

# Initialize the tool
calculator = CalculatorTool(config=CalculatorToolConfig())

# Define input data
input_data = CalculatorTool.input_schema(
    expression="sin(pi/2) + cos(pi/4)"
)

# Perform the calculation
result = calculator.run(input_data)
print(result)  # Expected output: {"result":"1.70710678118655"}
```

## Expression configuration

`safe_mode=True` is the default. It accepts numeric literals, symbolic variables,
parentheses, arithmetic (`+`, `-`, `*`, `/`, `//`, `%`, `**`, and `^`), and named
math functions: `sin`, `cos`, `tan`, `sec`, `csc`, `cot`, `asin`, `acos`, `atan`,
`atan2`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`, `exp`, `log`, `ln`,
`sqrt`, `cbrt`, `root`, `Abs`, `abs`, `sign`, `floor`, `ceiling`, `factorial`,
`Min`, `Max`, `Mod`, `gamma`, `binomial`, `gcd`, `lcm`, `erf`, `Rational`,
`integrate`, and `diff`. Calls use positional math arguments, for example
`Rational(1, 3)`, `integrate(x**2, x)`, or `diff(x**3, x)`. Write `factorial(n)`;
postfix `n!` is not supported. The input schema includes this syntax reference
so it is also available to models using the tool.

Constants are `pi`, `E`, `I`, `oo`, `nan`, `GoldenRatio`, `EulerGamma`, and
`Catalan`. Expressions are built directly from their
syntax tree; Python attribute access, indexing, comprehensions, strings, keyword
arguments, and arbitrary function calls raise `ValueError`.

Applications can explicitly supply trusted functions through configuration:

```python
calculator = CalculatorTool(
    CalculatorToolConfig(allowed_functions={"double": lambda value: 2 * value})
)
result = calculator.run(CalculatorTool.input_schema(expression="double(3) + 1"))
```

Configured functions can override built-in math functions. They receive SymPy
expressions and should return a SymPy expression or a Python number. They run
application code, so only the application should define this configuration.

Setting `safe_mode=False` restores the full SymPy string parser, including its
Python evaluation behavior. Use that mode only with trusted expressions. Safe
mode restricts expression syntax; it does not impose time or memory limits on
expensive mathematical calculations.

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new feature branch.
3. Commit your changes with clear messages.
4. Open a pull request detailing your changes.

Please ensure you follow the project's coding standards and include tests for any new features or bug fixes.

## License

This project is licensed under the same license as the main Atomic Agents project. See the [LICENSE](../../../LICENSE) file in the repository root for more details.
