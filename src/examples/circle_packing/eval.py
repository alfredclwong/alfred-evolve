# Task:
# Choose circle (c, r) packing within a unit square that maximises the sum
# of the radii of the circles, while ensuring that no two circles overlap

import argparse
import signal
from enum import Enum, auto
from pathlib import Path

import numpy as np
import sys


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


def init_parser():
    parser = argparse.ArgumentParser(description="Evaluate circle packing solutions.")
    parser.add_argument("-p", "--program", type=str, required=True, help="Program path")
    parser.add_argument(
        "-t",
        "--timeout",
        type=int,
        required=True,
        help="Timeout in seconds for the packing function",
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


def print_artifacts(artifact_dict):
    print(f"<ARTIFACT>{artifact_dict}</ARTIFACT>")


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
    timeout = args.timeout

    score = 0.0
    reason = Reason.INVALID_CODE
    artifact_dict = {}

    # Import the pack_26 function from the provided program
    sys.path.insert(0, str(program_path.parent))
    module_name = program_path.stem
    try:
        module = __import__(module_name)
        packing = run_with_timeout(module.pack_26, timeout)
        artifact_dict["packing"] = packing.tolist()
        valid, reason = is_valid(packing)
        if valid:
            score = score_packing(packing)
    except TimeoutError:
        reason = Reason.TIMEOUT
    except Exception as e:
        artifact_dict["error"] = str(e)
        artifact_dict["program_content"] = program_path.read_text()
        reason = Reason.INVALID_CODE
    finally:
        print_score(score, reason)
        print_artifacts(artifact_dict)

if __name__ == "__main__":
    main()
