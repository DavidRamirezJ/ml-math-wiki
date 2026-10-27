"""Chain rule demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

H = 1e-6


def slope(fun, x, h=H):
    return (fun(x + h) - fun(x - h)) / (2 * h)       # numeric derivative


f = lambda x: 3 * x + 1          # inner function
g = lambda u: u ** 2             # outer function
h = lambda x: g(f(x))            # composition (3x + 1)^2
x0 = 1.0
inner = 3.0                      # f'(x) = 3
outer = 2 * f(x0)                # g'(u) = 2u, at u = f(1) = 4
chain = outer * inner            # 8 * 3 = 24
num = slope(h, x0)

chain3 = lambda x: 5 * (2 * x) ** 2                  # a = 2x, b = a^2, c = 5b
local = np.array([2.0, 2 * (2 * x0), 5.0])           # da/dx, db/da, dc/db at x = 1
back = np.cumprod(local[::-1])                       # multiply from the end backward
num3 = slope(chain3, x0)

F = lambda p: np.array([p[0] ** 2, p[0] * p[1]])     # R^2 -> R^2
G = lambda u: u[0] + u[1] ** 2                       # R^2 -> R
p0 = np.array([1.0, 2.0])
J_F = np.array([[2 * p0[0], 0.0], [p0[1], p0[0]]])   # Jacobian of F at (1, 2)
u0 = F(p0)
J_G = np.array([[1.0, 2 * u0[1]]])                   # Jacobian of G at F(1, 2)
J_chain = J_G @ J_F                                  # chained Jacobians multiply
J_num = np.array([[slope(lambda t: G(F(p0 + t * e)), 0.0) for e in np.eye(2)]])

if __name__ == "__main__":
    print(f"(3x+1)^2 at x=1: outer {outer:g} * inner {inner:g} = {chain:g}")
    print(f"numeric slope: {num:.4f}")
    print("3-step chain, local slopes:", local)
    print("multiplied from the end:", back, " numeric:", round(num3, 4))
    print("J_F at (1, 2) =", J_F.tolist())
    print("J_G at F(1, 2) =", J_G.tolist())
    print("J_G @ J_F =", J_chain.tolist(), " numeric:", J_num.round(4).tolist())
