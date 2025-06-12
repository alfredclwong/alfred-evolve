# Task:
# Choose circle (c, r) packing within a unit square that maximises the sum
# of the radii of the circles, while ensuring that no two circles overlap

import argparse
import signal
from enum import Enum, auto
from pathlib import Path

import numpy as np


class Reason(Enum):
    VALID = auto()
    INVALID_TYPE = auto()
    INVALID_LENGTH = auto()
    INVALID_CIRCLE = auto()
    OUT_OF_BOUNDS = auto()
    OVERLAP = auto()
    TIMEOUT = auto()
    INVALID_CODE = auto()


def is_valid(packing: np.ndarray, tol: float = 1e-9) -> tuple[bool, Reason]:
    # type checks
    if not isinstance(packing, np.ndarray):
        print("Packing must be a numpy array.")
        return False, Reason.INVALID_TYPE
    if packing.ndim != 2 or packing.shape[0] != 26 or packing.shape[1] != 3:
        print("Packing must contain exactly 26 circles.")
        return False, Reason.INVALID_LENGTH
    if not np.all(packing > 0):
        print("Circle coordinates and radius must be positive numbers.")
        return False, Reason.INVALID_CIRCLE
    if not _is_in_unit_square(packing, tol):
        print("Circles must be within the unit square.")
        return False, Reason.OUT_OF_BOUNDS
    if _is_overlapping(packing, tol):
        print("Circles must not overlap.")
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


def init_parser():
    parser = argparse.ArgumentParser(description="Evaluate circle packing solutions.")
    parser.add_argument("-p", "--program", type=str, required=True, help="Program path")
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output"
    )
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        default=300,
        help="Timeout in seconds for the packing function (default: 300 seconds)",
    )
    return parser


def print_score(score: float, reason: Reason):
    score_dict = {
        f"{r.name}_CHECK": 0.0 if r == reason else 1.0
        for r in Reason
        if r != Reason.VALID
    }
    score_dict["SCORE"] = score
    score_str = ", ".join(f"{key}: {value}" for key, value in score_dict.items())
    print(f"<SCORE>{score_str}</SCORE>")


def run_with_timeout(func, timeout):
    def timeout_handler(signum, frame):
        raise TimeoutError("Execution time exceeded the 5-minute limit.")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        return func()
    except TimeoutError:
        raise
    finally:
        signal.alarm(0)  # Disable the alarm after execution


def main():
    parser = init_parser()
    args = parser.parse_args()
    program_path = Path(args.program)
    verbose = args.verbose
    timeout = args.timeout

    with open(program_path, "r") as f:
        program_content = f.read()

    score = 0.0

    try:
        local_vars = {}
        exec(program_content, globals(), local_vars)
        packing = run_with_timeout(local_vars["pack_26"], timeout)
        if verbose:
            print(f"<PACKING>{packing}</PACKING>")
        valid, reason = is_valid(packing)
        if valid:
            score = score_packing(packing)
    except TimeoutError:
        reason = Reason.TIMEOUT
    except Exception as e:
        print(e)
        reason = Reason.INVALID_CODE

    print_score(score, reason)


if __name__ == "__main__":
    main()
