import ast
import subprocess
from typing import Tuple

from verl.workers.agent.tools.base_tool import BaseTool


class PythonTool(BaseTool):
    """Python code execution tool, using local conda environment"""

    # Packages auto-imported into every executed snippet. Each call runs as a
    # fresh `python -c`, so these must be prepended on every execution. Ordering
    # matters: stdlib modules that `from sympy import *` shadows (notably `re`
    # and `product`) are re-bound AFTER the star import. scipy/pandas are
    # intentionally omitted — they roughly double per-call import latency.
    PREAMBLE = (
        "import math\n"
        "import cmath\n"
        "import statistics\n"
        "import datetime\n"
        "import itertools\n"
        "import functools\n"
        "import heapq\n"
        "import bisect\n"
        "import decimal\n"
        "import numpy as np\n"
        "import numpy\n"
        "import sympy\n"
        "import sympy as sp\n"
        "from sympy import *\n"
        "import re\n"
        "import random\n"
        "from math import comb, perm, log2, radians, degrees, hypot, isclose\n"
        "from itertools import permutations, combinations, combinations_with_replacement, product, chain\n"
        "from functools import reduce, lru_cache, cache\n"
        "from heapq import heappush, heappop, heapify\n"
        "from bisect import bisect_left, bisect_right, insort\n"
        "from fractions import Fraction\n"
        "from decimal import Decimal\n"
        "from collections import Counter, defaultdict, deque\n"
    )

    def __init__(self, conda_path: str, conda_env: str):
        """
        Initialize Python tool
        
        Args:
            conda_path: conda installation path
            conda_env: conda environment name
        """
        self.conda_path = conda_path
        self.conda_env = conda_env
        self.python_path = f"{conda_path}/envs/{conda_env}/bin/python"

    @property
    def name(self) -> str:
        return "python_interpreter"
    
    @property
    def trigger_tag(self) -> str:
        return "python"
    
    def execute(self, code: str, timeout: int = 120) -> str:
        """Execute Python code and return result"""
        result, report = self._run_code(code, timeout)
        
        if report == "Done":
            return result
        else:
            return report
    
    def _run_code(self, code: str, timeout: int) -> Tuple[str, str]:
        """Run Python code in conda environment and return result and status"""
        # 处理交互式代码
        code = self._preprocess_code(code)
        # Auto-import common packages so model snippets can use them without importing
        code = self.PREAMBLE + code
        
        try:
            # Use subprocess.run to execute the command synchronously
            process = subprocess.run(
                [self.python_path, '-c', code],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False 
            )

            if process.returncode == 0:
                return process.stdout.strip(), "Done"
            else:
                return "", process.stderr.strip()

        except subprocess.TimeoutExpired:
            return "", f"Execution timeout (超过 {timeout} 秒）"
        except Exception as e:
            return "", f"Execution exception: {str(e)}"
    
    def _preprocess_code(self, code: str) -> str:
        """
        Preprocess Python code, process interactive code
        Convert the last expression to a print statement (if not print)
        """
        try:
            tree = ast.parse(code)
            if tree.body:
                last_expr = tree.body[-1]
                if isinstance(last_expr, ast.Expr):
                    # Only convert when the last expression is not a print call
                    if not (isinstance(last_expr.value, ast.Call) 
                            and isinstance(last_expr.value.func, ast.Name) 
                            and last_expr.value.func.id == 'print'):
                        print_call = ast.Expr(
                            value=ast.Call(
                                func=ast.Name(id='print', ctx=ast.Load()),
                                args=[last_expr.value],
                                keywords=[]
                            )
                        )
                        tree.body[-1] = print_call
                        code = ast.unparse(tree)
        except:
            pass  # Keep the original code unchanged
        
        return code

def _test():
    batch_code = [
        """
# Create symbolic variables
x = sympy.symbols('x')
y = sympy.symbols('y')

# Create an expression
expr = x**2 + 2*x*y + y**2

print(f"Expression: {expr}")

# Derivative
derivative = sympy.diff(expr, x)
print(f"Derivative with respect to x: {derivative}")

# Substitute specific values
result = expr.subs([(x, 1), (y, 2)])
print(f"Value at x=1, y=2: {result}")
        """,
        """
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        np.array([1, 2, 3])
        print(np.array([1, 2, 3]))
        """
    ]
    
    async def run_test():
        # Create Python tool instance
        python_tool = PythonTool(
            conda_path="<your_conda_path>",  # Please modify according to the actual conda installation path
            conda_env="verl",              # Please modify according to the actual environment name
            max_concurrent=64
        )
        
        # Execute each code snippet
        for i, code in enumerate(batch_code):
            print(f"\n--- execute code snippet {i+1} ---")
            result = await python_tool.execute(code)
            print(f"Result:\n{result}")
    
    # Run test
    asyncio.run(run_test())

if __name__ == "__main__":
    _test()


