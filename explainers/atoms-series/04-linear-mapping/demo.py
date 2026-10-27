"""Linear mapping demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

fe1, fe2 = np.array([2, 1]), np.array([-1, 1])   # where the map sends e1 and e2
A = np.column_stack([fe1, fe2])                  # its matrix: those are the columns
x = np.array([3, 2])
Ax = A @ x
u, v = np.array([1, 2]), np.array([3, -1])
lhs = A @ (2 * u + 3 * v)                        # f(2u + 3v)
rhs = 2 * (A @ u) + 3 * (A @ v)                  # 2 f(u) + 3 f(v)
shift = lambda t: t + 1                          # noqa: E731  (not linear)
square = lambda t: t ** 2                        # noqa: E731  (not linear)
S = np.array([[1, 1], [0, 1]])                   # shear
P = np.array([[1, 0], [0, 0]])                   # projection onto the x-axis
PS = P @ S
y = np.array([1, 2])
W1 = np.array([[1, 2], [0, 1], [1, 0]])          # toy layer 1: 2 -> 3
W2 = np.array([[1, 0, 1], [2, 1, 0]])            # toy layer 2: 3 -> 2

if __name__ == "__main__":
    print("A =", A.tolist())
    print("A @ [3 2] =", Ax)
    print("f(2u+3v) =", lhs, "  2f(u)+3f(v) =", rhs)
    print("shift:  f(2+3) =", shift(2 + 3), "  f(2)+f(3) =", shift(2) + shift(3))
    print("square: f(1+1) =", square(1 + 1), "  f(1)+f(1) =", square(1) + square(1))
    print("P @ S =", PS.tolist())
    print("S @ y =", S @ y, "  P @ (S @ y) =", P @ (S @ y), "  (P @ S) @ y =", PS @ y)
    print("P @ [0 5] =", P @ np.array([0, 5]), "  (kernel)")
    print("W2 @ W1 =", (W2 @ W1).tolist())
    print("same map?", bool((W2 @ (W1 @ y) == (W2 @ W1) @ y).all()))
