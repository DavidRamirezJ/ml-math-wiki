"""Activation function demo — real NumPy output shown in the video. Toy numbers only."""
import math
import numpy as np


def relu(z):
    return np.maximum(0, z)


W1 = np.array([[1, 2], [0, 1]])            # toy layer 1
W2 = np.array([[2, 0], [1, 1]])            # toy layer 2
x = np.array([3, -1])
h = W1 @ x                                 # layer 1 output
two = W2 @ h                               # layer 2 after layer 1
W = W2 @ W1                                # the single equivalent matrix
one = W @ x
bent = W2 @ relu(h)                        # with ReLU in between

z = np.array([-2, -0.5, 0, 1.5, 3])
rz = relu(z)


def relu_interp(knots, f):
    """Sum of shifted ReLUs that matches f at the knots (piecewise-linear)."""
    s = np.diff(f(knots)) / np.diff(knots)                 # slope of each piece
    a = np.concatenate([[s[0]], np.diff(s)])               # weight of each ReLU
    g = lambda t: f(knots[0]) + sum(ak * relu(t - k) for ak, k in zip(a, knots[:-1]))
    return a, g


sq = lambda t: t ** 2
t = np.linspace(-2, 2, 4001)
a4, g4 = relu_interp(np.linspace(-2, 2, 5), sq)            # 4 pieces
a8, g8 = relu_interp(np.linspace(-2, 2, 9), sq)            # 8 pieces
err4 = np.max(np.abs(g4(t) - sq(t)))
err8 = np.max(np.abs(g8(t) - sq(t)))

eps = 1e-6                                                  # numeric slopes of ReLU
slopes = [(relu(v + eps) - relu(v - eps)) / (2 * eps) for v in (-2.0, 3.0)]
xs = np.linspace(0, 5, 11)                                  # a "dead" unit: w = 1, b = -10
pre = 1 * xs - 10
alive = int(np.sum(pre > 0))

Phi = lambda v: 0.5 * (1 + math.erf(v / math.sqrt(2)))
gelu = {v: v * Phi(v) for v in (-1.0, 1.0)}
sig0, tanh0 = 1 / (1 + math.exp(0)), math.tanh(0)

if __name__ == "__main__":
    print("W2 @ (W1 @ x)   =", two)
    print("(W2 @ W1) @ x   =", one, "  W2 @ W1 =", W.tolist())
    print("W2 @ relu(W1 x) =", bent)
    print("relu", z.tolist(), "=", rz.tolist())
    print("4 ReLU weights  =", a4.tolist(), " max error:", round(err4, 4))
    print("8 pieces        : max error", round(err8, 4))
    print("ReLU slope at -2 and 3 =", [round(float(s), 3) for s in slopes])
    print("dead unit: inputs with slope 1 =", alive, "of", len(xs))
    print("GELU(-1), GELU(1) =", round(gelu[-1.0], 4), round(gelu[1.0], 4))
    print("sigmoid(0), tanh(0) =", sig0, tanh0)
