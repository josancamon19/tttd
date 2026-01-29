"""Erdős code executor.

Exact port of TTT-Discover's execution pattern from tasks/erdos_min_overlap/task.py.
Runs generated code in a subprocess with injected verifier functions and initial_h_values.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import pickle
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from tttd.erdos.verifier import evaluate_erdos_solution, verify_c5_solution

logger = logging.getLogger(__name__)

# Module-level executor for reuse
_executor: ProcessPoolExecutor | None = None


def get_executor(max_workers: int = 4) -> ProcessPoolExecutor:
    """Get or create the process pool executor."""
    global _executor
    if _executor is None:
        _executor = ProcessPoolExecutor(max_workers=max_workers)
    return _executor


def shutdown_executor():
    """Shutdown the executor."""
    global _executor
    if _executor is not None:
        _executor.shutdown(wait=False)
        _executor = None


def extract_code_block(text: str) -> str | None:
    """Extract Python code block from markdown text."""
    # Try ```python first
    match = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fall back to generic code block
    match = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()

    return None


def preprocess_generation(
    generation: str,
    initial_h_values: list[float] | np.ndarray | None = None,
) -> str:
    """Preprocess generated code by injecting verifier functions and initial_h_values.

    Exact port from TTT-Discover's task.py:preprocess_generation.
    """
    numpy_import = "import numpy as np"

    # Get source code of verifier functions
    verifier_src = inspect.getsource(evaluate_erdos_solution)
    verify_src = inspect.getsource(verify_c5_solution)

    base = (
        numpy_import + "\n\n" +
        verify_src + "\n\n" +
        verifier_src + "\n\n"
    )

    # Inject initial_h_values if available
    if initial_h_values is not None:
        if isinstance(initial_h_values, np.ndarray):
            initial_h_values = initial_h_values.tolist()
        initial_h_values_code = f"initial_h_values = np.array({initial_h_values!r})"
        base += initial_h_values_code + "\n\n"

    return base + generation


def _run_code_in_subprocess(
    code: str,
    timeout: int,
    initial_h_values: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Run code in a subprocess and return results.

    This function preprocesses the code to inject verifier functions
    and initial_h_values, exactly like the original TTT-Discover.
    """
    # Preprocess code to inject verifier functions and initial_h_values
    preprocessed_code = preprocess_generation(code, initial_h_values)

    # Create a wrapper script that executes the code and saves results
    wrapper_script = f'''
import sys
import pickle
import numpy as np
import io
from contextlib import redirect_stdout, redirect_stderr

# Capture stdout
stdout_capture = io.StringIO()
stderr_capture = io.StringIO()

# User code (preprocessed with verifier functions and initial_h_values)
{preprocessed_code}

# Execute and save results
try:
    with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
        result = run()
    stdout_str = stdout_capture.getvalue()
    with open(sys.argv[1], 'wb') as f:
        pickle.dump({{"success": True, "result": result, "stdout": stdout_str}}, f)
except Exception as e:
    import traceback
    stdout_str = stdout_capture.getvalue()
    stderr_str = stderr_capture.getvalue()
    with open(sys.argv[1], 'wb') as f:
        pickle.dump({{
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stdout": stdout_str,
            "stderr": stderr_str,
        }}, f)
'''

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as code_file:
        code_file.write(wrapper_script)
        code_path = code_file.name

    results_path = code_path + ".pkl"

    try:
        # Run in subprocess
        process = subprocess.run(
            [sys.executable, code_path, results_path],
            capture_output=True,
            timeout=timeout,
            text=True,
        )

        # Load results
        if Path(results_path).exists():
            with open(results_path, "rb") as f:
                result = pickle.load(f)
                # Add any additional stdout/stderr from process
                if process.stdout:
                    result["stdout"] = result.get("stdout", "") + process.stdout
                if process.stderr:
                    result["stderr"] = result.get("stderr", "") + process.stderr
                return result
        else:
            return {
                "success": False,
                "error": f"No results file. stdout: {process.stdout[:500]}, stderr: {process.stderr[:500]}",
                "stdout": process.stdout,
                "stderr": process.stderr,
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s", "stdout": "", "stderr": ""}
    except Exception as e:
        return {"success": False, "error": str(e), "stdout": "", "stderr": ""}
    finally:
        # Cleanup
        Path(code_path).unlink(missing_ok=True)
        Path(results_path).unlink(missing_ok=True)


def run_erdos_eval(
    code: str,
    timeout: int = 60,
    initial_h_values: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Evaluate Erdős solution code.

    Args:
        code: Python code defining a run() function
        timeout: Execution timeout in seconds
        initial_h_values: Optional initial h_values to inject into execution context

    Returns:
        Dict with keys: success, score, c5_bound, msg, h_values, stdout
    """
    # Run code in subprocess (with injected verifier functions and initial_h_values)
    result = _run_code_in_subprocess(code, timeout, initial_h_values)
    stdout = result.get("stdout", "")

    if not result.get("success"):
        error_msg = result.get("error", "Unknown error")
        if result.get("traceback"):
            error_msg += "\n" + result.get("traceback", "")[:500]
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": error_msg,
            "h_values": None,
            "stdout": stdout,
        }

    # Unpack result
    try:
        h_values, c5_bound, n_points = result["result"]
        h_values = np.asarray(h_values)
    except Exception as e:
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": f"Invalid return value: {e}",
            "h_values": None,
            "stdout": stdout,
        }

    # Verify and evaluate using the same verifier that was injected
    try:
        actual_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)
    except ValueError as e:
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": f"Verification failed: {e}",
            "h_values": None,
            "stdout": stdout,
        }

    # Score = 1 / bound (higher is better for lower bounds)
    # Matching original: return float(1.0 / (1e-8 + c5_bound))
    score = 1.0 / (1e-8 + actual_bound)

    return {
        "success": True,
        "score": score,
        "c5_bound": actual_bound,
        "msg": f"OK. C5 bound: {actual_bound:.6f}",
        "h_values": h_values,
        "stdout": stdout,
    }


async def run_erdos_eval_async(
    code: str,
    timeout: int = 60,
    initial_h_values: list[float] | np.ndarray | None = None,
) -> dict[str, Any]:
    """Async wrapper for run_erdos_eval using ProcessPoolExecutor."""
    loop = asyncio.get_event_loop()
    executor = get_executor()

    try:
        # Run in thread pool (ProcessPoolExecutor submission happens in thread)
        result = await asyncio.wait_for(
            loop.run_in_executor(
                executor,
                run_erdos_eval,
                code,
                timeout,
                initial_h_values,
            ),
            timeout=timeout + 10,  # Extra buffer for executor overhead
        )
        return result
    except asyncio.TimeoutError:
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": f"Async timeout after {timeout}s",
            "h_values": None,
            "stdout": "",
        }
    except Exception as e:
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": f"Executor error: {e}",
            "h_values": None,
            "stdout": "",
        }
