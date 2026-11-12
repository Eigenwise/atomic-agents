import ast
import operator
from typing import Dict, Any

from pydantic import Field
import sympy

from atomic_agents import BaseIOSchema, BaseTool, BaseToolConfig


_MATH_FUNCTIONS = {
    name: getattr(sympy, name)
    for name in (
        "sin",
        "cos",
        "tan",
        "asin",
        "acos",
        "atan",
        "atan2",
        "sinh",
        "cosh",
        "tanh",
        "asinh",
        "acosh",
        "atanh",
        "exp",
        "log",
        "sqrt",
        "cbrt",
        "root",
        "Abs",
        "sign",
        "floor",
        "ceiling",
        "factorial",
        "Min",
        "Max",
    )
}
_MATH_FUNCTIONS.update(abs=sympy.Abs, ln=sympy.log)
_MATH_CONSTANTS = {
    name: getattr(sympy, name) for name in ("pi", "E", "I", "oo", "nan", "GoldenRatio", "EulerGamma", "Catalan")
}
_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


################
# INPUT SCHEMA #
################
class CalculatorToolInputSchema(BaseIOSchema):
    """
    Tool for performing calculations. Supports basic arithmetic operations
    like addition, subtraction, multiplication, and division, as well as more
    complex operations like exponentiation and trigonometric functions.
    Use this tool to evaluate mathematical expressions.
    """

    expression: str = Field(..., description="Mathematical expression to evaluate. For example, '2 + 2'.")


#################
# OUTPUT SCHEMA #
#################
class CalculatorToolOutputSchema(BaseIOSchema):
    """
    Schema for the output of the CalculatorTool.
    """

    result: str = Field(..., description="Result of the calculation.")


#################
# CONFIGURATION #
#################
class CalculatorToolConfig(BaseToolConfig):
    """
    Configuration for the CalculatorTool.

    Attributes:
        title (Optional[str]): Overrides the default title of the tool.
        description (Optional[str]): Overrides the default description of the tool.
        safe_mode (bool): Whether to run in safe mode with restricted operations.
        allowed_functions (Dict[str, Any]): Functions to make available in the calculator.
    """

    safe_mode: bool = True
    allowed_functions: Dict[str, Any] = {}


#####################
# MAIN TOOL & LOGIC #
#####################
class CalculatorTool(BaseTool[CalculatorToolInputSchema, CalculatorToolOutputSchema]):
    """
    Tool for evaluating mathematical expressions.

    Attributes:
        input_schema (CalculatorToolInputSchema): Schema defining the input data.
        output_schema (CalculatorToolOutputSchema): Schema defining the output data.
        safe_mode (bool): Whether to run in safe mode with restricted operations.
        allowed_functions (Dict[str, Any]): Functions to make available in the calculator.
    """

    def __init__(self, config: CalculatorToolConfig = CalculatorToolConfig()):
        """
        Initializes the CalculatorTool.

        Args:
            config (CalculatorToolConfig): Configuration for the tool.
        """
        super().__init__(config)
        self.safe_mode = config.safe_mode
        self.allowed_functions = config.allowed_functions

    def _evaluate_node(self, node: ast.AST, expression: str) -> sympy.Expr:
        """Build a mathematical expression without evaluating Python source."""
        if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
            if type(node.value) is int:
                return sympy.Integer(node.value)
            literal = ast.get_source_segment(expression, node).replace("_", "")
            if type(node.value) is complex:
                return sympy.Float(literal[:-1]) * sympy.I
            return sympy.Float(literal)
        if isinstance(node, ast.Name):
            return _MATH_CONSTANTS.get(node.id, sympy.Symbol(node.id))
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](
                self._evaluate_node(node.left, expression), self._evaluate_node(node.right, expression)
            )
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](self._evaluate_node(node.operand, expression))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            function = self.allowed_functions.get(node.func.id, _MATH_FUNCTIONS.get(node.func.id))
            if callable(function):
                result = function(*(self._evaluate_node(arg, expression) for arg in node.args))
                if type(result) in (int, float, complex):
                    return sympy.sympify(result)
                if isinstance(result, sympy.Expr):
                    return result
        raise ValueError("Operation is not supported in calculator safe mode")

    def _parse_expression(self, expression: str) -> sympy.Expr:
        if not self.safe_mode:
            return sympy.sympify(expression, locals=self.allowed_functions)
        try:
            # SymPy accepts ^ as exponentiation; normalize before Python parses
            # it so it has power precedence rather than bitwise-XOR precedence.
            expression = expression.strip().replace("^", "**")
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Invalid expression in calculator safe mode") from exc
        return self._evaluate_node(tree.body, expression)

    def run(self, params: CalculatorToolInputSchema) -> CalculatorToolOutputSchema:
        """
        Executes the CalculatorTool with the given parameters.

        Args:
            params (CalculatorToolInputSchema): The input parameters for the tool.

        Returns:
            CalculatorToolOutputSchema: The result of the calculation.
        """
        parsed_expr = self._parse_expression(params.expression)

        # Evaluate the expression numerically
        result = parsed_expr.evalf()
        return CalculatorToolOutputSchema(result=str(result))


#################
# EXAMPLE USAGE #
#################
if __name__ == "__main__":
    calculator = CalculatorTool()
    result = calculator.run(CalculatorToolInputSchema(expression="sin(pi/2) + cos(pi/4)"))
    print(result)  # Expected output: {"result":"1.70710678118655"}
