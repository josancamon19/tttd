"""Local subprocess executor for Erdős task.

Runs generated Python code in a subprocess with timeout.
No Ray, no SLURM, no Daytona - just ProcessPoolExecutor.
"""

from __future__ import annotations

import logging
import pickle
import re
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

import numpy as np

from tttd.erdos.verifier import evaluate_erdos_solution

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


def _run_code_in_subprocess(code: str, timeout: int) -> dict[str, Any]:
    """Run code in a subprocess and return results.

    This function is designed to be called in a separate process.
    """
    # Create a wrapper script that executes the code and saves results
    wrapper_script = f"""
import sys
import pickle
import numpy as np

# User code
{code}

# Execute and save results
try:
    result = run()
    with open(sys.argv[1], 'wb') as f:
        pickle.dump({{"success": True, "result": result}}, f)
except Exception as e:
    import traceback
    with open(sys.argv[1], 'wb') as f:
        pickle.dump({{"success": False, "error": str(e), "traceback": traceback.format_exc()}}, f)
"""

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
                return pickle.load(f)
        else:
            return {
                "success": False,
                "error": f"No results file. stdout: {process.stdout[:500]}, stderr: {process.stderr[:500]}",
            }

    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Timeout after {timeout}s"}
    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        # Cleanup
        Path(code_path).unlink(missing_ok=True)
        Path(results_path).unlink(missing_ok=True)


def run_erdos_eval(code: str, timeout: int = 60) -> dict[str, Any]:
    """Evaluate Erdős solution code.

    Args:
        code: Python code defining a run() function
        timeout: Execution timeout in seconds

    Returns:
        Dict with keys: success, score, c5_bound, msg, h_values (if successful)
    """
    # Run code in subprocess
    result = _run_code_in_subprocess(code, timeout)

    if not result.get("success"):
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": result.get("error", "Unknown error"),
            "h_values": None,
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
        }

    # Verify and evaluate
    actual_bound = evaluate_erdos_solution(h_values, c5_bound, n_points)

    if actual_bound == float("inf"):
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": "Verification failed",
            "h_values": None,
        }

    # Score = 1 / bound (higher is better for lower bounds)
    score = 1.0 / (1e-8 + actual_bound)

    return {
        "success": True,
        "score": score,
        "c5_bound": actual_bound,
        "msg": f"OK. C5 bound: {actual_bound:.6f}",
        "h_values": h_values,
    }


async def run_erdos_eval_async(code: str, timeout: int = 60) -> dict[str, Any]:
    """Async wrapper for run_erdos_eval using ProcessPoolExecutor."""
    import asyncio

    loop = asyncio.get_event_loop()
    executor = get_executor()

    try:
        # Run in thread pool (ProcessPoolExecutor submission happens in thread)
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, run_erdos_eval, code, timeout),
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
        }
    except Exception as e:
        return {
            "success": False,
            "score": 0.0,
            "c5_bound": float("inf"),
            "msg": f"Executor error: {e}",
            "h_values": None,
        }
