"""Derivative demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np


def f(x):
    return x ** 2


def ratio(fun, x, h):
    """Average slope between x and x + h (the difference quotient)."""
    return (fun(x + h) - fun(x)) / h


hs = np.array([1.0, 0.1, 0.01, 0.001])
ratios = np.array([ratio(f, 3.0, h) for h in hs])
exact = 2 * 3.0                          # f'(x) = 2x at x = 3
approx_31 = f(3.0) + exact * 0.1         # local linear approximation of f(3.1)
true_31 = f(3.1)
cube = lambda x: x ** 3                  # second example: x cubed at x = 2
cube_num = ratio(cube, 2.0, 0.01)
cube_exact = 3 * 2.0 ** 2
cube_approx = cube(2.0) + cube_exact * 0.1
cube_true = cube(2.1)
loss = lambda w: (2 * w - 6) ** 2        # toy loss of one weight
loss_slope = ratio(loss, 1.0, 1e-6)      # exact: 8w - 24 = -16 at w = 1
loss_change = loss(1.01) - loss(1.0)

if __name__ == "__main__":
    for h, r in zip(hs, ratios):
        print(f"h = {h:<6}  (f(3+h) - f(3)) / h = {r:.4f}")
    print("exact f'(3) = 2*3 =", exact)
    print(f"f(3.1): linear guess {approx_31:.2f}, true {true_31:.4f}")
    print(f"x^3 at 2: numeric {cube_num:.4f}, exact {cube_exact}")
    print(f"(2.1)^3: linear guess {cube_approx:.2f}, true {cube_true:.4f}")
    print(f"toy loss slope at w=1: {loss_slope:.4f}")
    print(f"nudge w by 0.01: loss changes by {loss_change:.4f}")
