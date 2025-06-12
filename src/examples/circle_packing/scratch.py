# %%
import numpy as np
import difflib

# %%
diff_content = """\
@@ -4,6 +4,34 @@
import numpy as np
 
 
def pack_26() -> np.ndarray:
+    circles = np.array([
+        (0.107, 0.107, 0.066),
+        (0.107, 0.893, 0.066),
+        (0.893, 0.107, 0.066),
+        (0.893, 0.893, 0.066),
+        (0.333, 0.333, 0.05),
+        (0.333, 0.667, 0.05),
+        (0.667, 0.333, 0.05),
+        (0.667, 0.667, 0.05),
+        (0.166, 0.5, 0.042),
+        (0.834, 0.5, 0.042),
+        (0.5, 0.166, 0.042),
+        (0.5, 0.834, 0.042),
+        (0.25, 0.25, 0.033),
+        (0.25, 0.75, 0.033),
+        (0.75, 0.25, 0.033),
+        (0.75, 0.75, 0.033),
+        (0.417, 0.417, 0.025),
+        (0.417, 0.583, 0.025),
+        (0.583, 0.417, 0.025),
+        (0.583, 0.583, 0.025),
+        (0.375, 0.5, 0.018),
+        (0.625, 0.5, 0.018),
+        (0.5, 0.375, 0.018),
+        (0.5, 0.625, 0.018),
+        (0.5, 0.5, 0.012),
+    ])
     return np.zeros((26, 3), dtype=np.float64)
"""
program_content = """\
import numpy as np


def pack_26() -> np.ndarray:
    return np.zeros((26, 3), dtype=np.float64)

"""

from diff_match_patch import diff_match_patch

dmp = diff_match_patch()
patches = dmp.patch_fromText(diff_content)
patched_content, success = dmp.patch_apply(patches, program_content)
print(patches)
print(patched_content)
print(success)

# %%
packing = np.array(
    [
        (0.5, 0.5, 0.058319027604622216),
        (0.8, 0.5, 0.05513527493330955),
        (0.7121320343559643, 0.7121320343559643, 0.10466903285363564),
        (0.5, 0.8, 0.08482428765987904),
        (0.2878679656440358, 0.7121320343559643, 0.09642007408553413),
        (0.2, 0.5, 0.08407916958746124),
        (0.2878679656440357, 0.2878679656440358, 0.09670729085271067),
        (0.49999999999999994, 0.2, 0.0992779475419393),
        (0.7121320343559642, 0.2878679656440357, 0.13033211187711455),
        (0.999, 0.5, 0.0010000000000000009),
        (0.999, 0.7678784026555628, 0.0010000000000000009),
        (0.9949747468305833, 0.9949747468305832, 0.00502525316941671),
        (0.7678784026555628, 0.999, 0.0010000000000000009),
        (0.5, 0.999, 0.0010000000000000009),
        (0.2321215973444372, 0.999, 0.0010000000000000009),
        (0.005025253169416821, 0.9949747468305833, 0.00502525316941671),
        (0.001, 0.767878402655563, 0.001),
        (0.001, 0.5000000000000001, 0.001),
        (0.001, 0.23212159734443727, 0.001),
        (0.005025253169416655, 0.005025253169416821, 0.004747782821924281),
        (0.23212159734443677, 0.001, 0.001),
        (0.4999999999999999, 0.001, 0.001),
        (0.767878402655563, 0.001, 0.001),
        (0.9949747468305832, 0.005025253169416655, 0.005025253169416655),
        (0.999, 0.23212159734443677, 0.0010000000000000009),
        (0.001, 0.001, 0.0009447848022501555),
    ]
)


# %%
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


# %%
tol = 1e-9
_is_in_unit_square(packing, tol=tol), _is_overlapping(packing, tol=tol)

# %%
