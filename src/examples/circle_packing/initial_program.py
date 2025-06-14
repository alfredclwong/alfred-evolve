from enum import Enum, auto

import numpy as np


def pack_26() -> np.ndarray:
    raise NotImplementedError


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
