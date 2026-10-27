"""Gradient demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np


def num_grad(f, p, h=1e-6):
    """Numeric gradient: nudge one input at a time, freeze the others."""
    p = np.asarray(p, dtype=float)
    g = np.zeros_like(p)
    for i in range(len(p)):
        step = np.zeros_like(p)
        step[i] = h
        g[i] = (f(p + step) - f(p - step)) / (2 * h)
    return g


f1 = lambda p: p[0] ** 2 + 3 * p[1]          # f(x, y) = x^2 + 3y
f2 = lambda p: p[0] ** 2 + p[1] ** 2         # f(x, y) = x^2 + y^2 (a bowl)
p1, p2 = np.array([1.0, 2.0]), np.array([3.0, 4.0])
g1_num, g1_exact = num_grad(f1, p1), np.array([2 * p1[0], 3.0])
g2_num, g2_exact = num_grad(f2, p2), 2 * p2
h = np.array([0.1, 0.1])
lin_guess = f1(p1) + g1_exact @ h            # f(x + h) ~ f(x) + grad . h
lin_true = f1(p1 + h)
h2 = np.array([0.1, -0.1])
lin2_guess, lin2_true = f2(p2) + g2_exact @ h2, f2(p2 + h2)
angles = np.array([0, 45, 90, 180])          # direction u relative to the gradient
rates = np.linalg.norm(g2_exact) * np.cos(np.radians(angles))
step = p2 - 0.1 * g2_exact                   # one step against the gradient
f_before, f_after = f2(p2), f2(step)

if __name__ == "__main__":
    print("f = x^2 + 3y at (1, 2): numeric", g1_num.round(4), " exact", g1_exact)
    print("f = x^2 + y^2 at (3, 4): numeric", g2_num.round(4), " exact", g2_exact)
    print(f"linear guess f(1.1, 2.1) = {lin_guess:.2f}, true {lin_true:.2f}")
    print(f"linear guess f(3.1, 3.9) = {lin2_guess:.2f}, true {lin2_true:.2f}")
    for a, r in zip(angles, rates):
        print(f"angle {a:>3} deg from gradient: slope {r:+.2f}")
    print("step against gradient:", p2, "->", step.round(2))
    print(f"f before {f_before:.2f}, after {f_after:.2f}")
