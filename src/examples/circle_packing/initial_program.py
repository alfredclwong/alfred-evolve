import numpy as np

def check_packing(circles: np.ndarray, tol: float = 1e-9) -> bool:
    if len(circles) != 26:
        return False
    for i in range(len(circles)):
        x1, y1, r1 = circles[i]
        if not (tol <= x1 <= 1 - tol and tol <= y1 <= 1 - tol and r1 > tol):
            return False
        for j in range(i + 1, len(circles)):
            x2, y2, r2 = circles[j]
            distance_squared = (x1 - x2) ** 2 + (y1 - y2) ** 2
            if distance_squared < (r1 + r2 + tol) ** 2:
                return False
    return True

def pack_26() -> np.ndarray:
    raise NotImplementedError
