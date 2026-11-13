import os
import sys
import builtins

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tool.calculator import (  # noqa: E402
    CalculatorTool,
    CalculatorToolConfig,
    CalculatorToolInputSchema,
    CalculatorToolOutputSchema,
)


def test_calculator_tool():
    calculator_tool = CalculatorTool()
    input_schema = CalculatorToolInputSchema(expression="2 + 2")
    result = calculator_tool.run(input_schema)
    assert result == CalculatorToolOutputSchema(result="4.00000000000000")


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("sin(pi/2) + cos(pi/4)", "1.70710678118655"),
        ("sqrt(16) + log(E)", "5.00000000000000"),
        ("-2**2 + 3 * (4 - 1) / 2", "0.500000000000000"),
        ("2^3 + 1", "9.00000000000000"),
        ("7 // 2 + 7 % 2", "4.00000000000000"),
        ("1/3 + 2/3", "1.00000000000000"),
        ("1e-3 + .001", "0.00200000000000000"),
        ("Abs(-3) + factorial(4)", "27.0000000000000"),
        ("x**2 + 2*x + 1", "x**2 + 2.0*x + 1.0"),
        ("sqrt(-1)", "1.0*I"),
        ("1e400", "1.00000000000000e+400"),
        ("1e-400", "1.00000000000000e-400"),
        ("0.12345678901234567890 - 0.12345678901234567891", "-9.95263713023803e-21"),
        ("2.5j + 1", "1.0 + 2.5*I"),
        ("sec(0)", "1.00000000000000"),
        ("csc(pi/2)", "1.00000000000000"),
        ("cot(pi/4)", "1.00000000000000"),
        ("Mod(7, 3)", "1.00000000000000"),
        ("gamma(5)", "24.0000000000000"),
        ("binomial(5, 2)", "10.0000000000000"),
        ("gcd(12, 8)", "4.00000000000000"),
        ("lcm(12, 8)", "24.0000000000000"),
        ("erf(1)", "0.842700792949715"),
        ("Rational(1, 3)", "0.333333333333333"),
        ("integrate(x**2, x)", "0.333333333333333*x**3"),
        ("diff(x**3, x)", "3.0*x**2"),
        ("factorial(5)", "120.000000000000"),
    ],
)
def test_safe_calculator_math(expression, expected):
    result = CalculatorTool().run(CalculatorToolInputSchema(expression=expression))
    assert result.result == expected


@pytest.mark.parametrize(
    "expression",
    [
        "pi.evalf()",
        "[1, 2][0]",
        "sum([1, 2])",
        "Symbol('x')",
        "eval('2 + 2')",
        "(lambda: 2)()",
        "2 if True else 3",
        "[x for x in (1, 2)][0]",
        "sin(*(pi,))",
        "sqrt(4, evaluate=False)",
        "5!",
        "Rational('1/3')",
        "diff(x.__class__, x)",
    ],
)
def test_safe_calculator_rejects_non_math_operations(expression):
    with pytest.raises(ValueError, match="safe mode"):
        CalculatorTool().run(CalculatorToolInputSchema(expression=expression))


def test_safe_calculator_does_not_execute_python_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(builtins, "calculator_test_probe", lambda: calls.append(True) or 4, raising=False)

    with pytest.raises(ValueError, match="safe mode"):
        CalculatorTool().run(CalculatorToolInputSchema(expression="calculator_test_probe()"))

    assert calls == []


@pytest.mark.parametrize("safe_mode", [True, False])
def test_calculator_uses_configured_functions(safe_mode):
    config = CalculatorToolConfig(safe_mode=safe_mode, allowed_functions={"double": lambda value: value * 2})
    result = CalculatorTool(config).run(CalculatorToolInputSchema(expression="double(3) + 1"))
    assert result.result == "7.00000000000000"


def test_calculator_can_override_math_functions():
    config = CalculatorToolConfig(allowed_functions={"sqrt": lambda value: value})
    result = CalculatorTool(config).run(CalculatorToolInputSchema(expression="sqrt(9)"))
    assert result.result == "9.00000000000000"


def test_explicit_unsafe_mode_preserves_sympy_expression_support():
    tool = CalculatorTool(CalculatorToolConfig(safe_mode=False))
    result = tool.run(CalculatorToolInputSchema(expression="Symbol('x') + 2"))
    assert result.result == "x + 2.0"


if __name__ == "__main__":
    test_calculator_tool()
