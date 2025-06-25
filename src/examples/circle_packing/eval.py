# Task:
# Choose circle (c, r) packing within a unit square that maximises the sum
# of the radii of the circles, while ensuring that no two circles overlap

import builtins
import json
import os
import signal
import time
from enum import Enum, auto
from typing import Callable, TypeVar

import numpy as np
import scipy

T = TypeVar("T")


class Reason(Enum):
    INVALID_CODE = auto()
    TIMEOUT = auto()
    INVALID_TYPE = auto()
    INVALID_LENGTH = auto()
    INVALID_CIRCLE = auto()
    OUT_OF_BOUNDS = auto()
    OVERLAP = auto()
    VALID = auto()


def is_valid(packing: np.ndarray, tol: float = 1e-9) -> tuple[bool, Reason]:
    # type checks
    if not isinstance(packing, np.ndarray):
        return False, Reason.INVALID_TYPE
    if packing.ndim != 2 or packing.shape[0] != 26 or packing.shape[1] != 3:
        return False, Reason.INVALID_LENGTH
    if not np.all(packing > 0):
        return False, Reason.INVALID_CIRCLE
    if not _is_in_unit_square(packing, tol):
        return False, Reason.OUT_OF_BOUNDS
    if _is_overlapping(packing, tol):
        return False, Reason.OVERLAP
    return True, Reason.VALID


def _is_in_unit_square(packing: np.ndarray, tol: float) -> bool:
    x, y, r = packing.T
    return bool(
        ((tol < x - r) & (x + r < 1 - tol) & (tol < y - r) & (y + r < 1 - tol)).all()
    )


def _is_overlapping(packing: np.ndarray, tol: float) -> bool:
    # Sort by x-coordinate for efficient early termination
    n = len(packing)
    if n < 2:
        return False
    sorted_indices = np.argsort(packing[:, 0])
    sorted_circles = packing[sorted_indices]
    for i in range(n):
        x1, y1, r1 = sorted_circles[i]
        for j in range(i + 1, n):
            x2, y2, r2 = sorted_circles[j]
            radius_sum = r1 + r2 + tol
            if x2 - x1 > radius_sum:
                break
            distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if distance_squared < radius_sum**2:
                return True
    return False


def score_packing(packing: np.ndarray) -> float:
    return packing[:, 2].sum()


def print_score(score: float, time_remaining_frac: float, reason: Reason):
    # Cascade evaluation: if the program is invalid for an earlier reason, all subsequent checks are scored as 0.
    failed = False
    score_dict = {}
    for r in Reason:
        if r == reason and r != Reason.VALID:
            failed = True
        score_dict[f"{r.name}_CHECK"] = 0.0 if failed else 1.0
    score_dict["TIME_REMAINING_FRAC"] = round(time_remaining_frac, 4)
    score_dict["SCORE"] = score
    score_str = json.dumps(score_dict, indent=None)
    print(f"<SCORE>{score_str}</SCORE>")


def print_artifacts(artifact_dict):
    artifact_str = json.dumps(artifact_dict, indent=None)
    print(f"<ARTIFACT>{artifact_str}</ARTIFACT>")


def run_with_timeout(func: Callable[[], T], timeout) -> tuple[T, float]:
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution time exceeded the 5-minute limit.")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        start_time = time.time()
        result = func()
        elapsed_time = time.time() - start_time
        time_remaining_frac = (timeout - elapsed_time) / timeout
        return result, time_remaining_frac
    except TimeoutError:
        raise
    finally:
        signal.alarm(0)  # Disable the alarm after execution


def _exec_safe(program_content: str):
    unsafe_builtins = [
        "open",  # file access
        "input",  # can hang execution
        "eval",  # arbitrary code execution
        "exec",  # executes arbitrary code
        "compile",  # compiles code into executable
        "globals",  # exposes global namespace
        "locals",  # can be used to inspect locals
        "__import__",  # access arbitrary modules (e.g., os, sys)
        "vars",  # access variable mappings
        "dir",  # introspection
        "getattr",  # dynamic access to object properties
        "setattr",  # can mutate state
        "delattr",  # can remove properties
        "super",  # class introspection
        "help",  # introspection
        "property",  # can run arbitrary code
        "classmethod",  # can define dynamic class behavior
        "staticmethod",
        "memoryview",  # buffer access
        "bytearray",  # mutable binary data
        "bytes",  # binary data
        "globals",  # access global scope
        "object",  # base class, access to dunder methods
        "type",  # can dynamically create classes
        "print",
    ]
    safe_builtins = {
        name: getattr(builtins, name)
        for name in dir(builtins)
        if name not in unsafe_builtins and not name.startswith("__")
    }
    safe_globals = {
        "__builtins__": safe_builtins,
        "np": np,
        "scipy": scipy,
    }
    try:
        exec(program_content, safe_globals)
        return safe_globals
    except Exception as e:
        raise RuntimeError(f"Error executing program: {e}")


def main():
    timeout = int(os.environ.get("TIME_LIMIT", 300))
    program_content = os.environ.get("PROGRAM_CONTENT", "")

    score = 0.0
    time_remaining_frac = 1.0
    reason = Reason.INVALID_CODE
    artifact_dict = {}

    try:
        safe_globals = _exec_safe(program_content)
        pack_26 = safe_globals.get("pack_26")
        packing, time_remaining_frac = run_with_timeout(pack_26, timeout)
        artifact_dict["packing"] = packing.tolist()
        valid, reason = is_valid(packing)
        if valid:
            score = score_packing(packing)
    except TimeoutError:
        reason = Reason.TIMEOUT
    except Exception as e:
        artifact_dict["error"] = str(e)
        artifact_dict["program_content"] = program_content
        reason = Reason.INVALID_CODE
    finally:
        print_score(score, time_remaining_frac, reason)
        print_artifacts(artifact_dict)


if __name__ == "__main__":
    main()
