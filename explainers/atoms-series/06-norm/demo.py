"""Norm demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

x = np.array([3.0, 4.0])
l1 = np.linalg.norm(x, ord=1)          # |3| + |4|
l2 = np.linalg.norm(x)                 # sqrt(3^2 + 4^2), the default
linf = np.linalg.norm(x, ord=np.inf)   # max(|3|, |4|)
x3 = np.array([2.0, 3.0, 6.0])
n3 = np.linalg.norm(x3)                # sqrt(4 + 9 + 36)
unit = x / np.linalg.norm(x)           # normalize: divide by the norm
a = np.array([1.0, 2.0, 3.0])
b = np.array([-1.0, -1.0, 0.0])
dist = np.linalg.norm(a - b)           # distance = norm of the difference
u, v = np.array([3.0, 4.0]), np.array([4.0, -3.0])
tri_left = np.linalg.norm(u + v)       # triangle inequality check
tri_right = np.linalg.norm(u) + np.linalg.norm(v)
g = np.array([6.0, 8.0])               # toy gradient
max_norm = 5.0
g_clipped = g * min(1.0, max_norm / np.linalg.norm(g))
query = np.array([2.0, 2.0])
points = {"A": np.array([1.0, 5.0]), "B": np.array([4.0, 3.0]), "C": np.array([0.0, 0.0])}
dists = {k: np.linalg.norm(p - query) for k, p in points.items()}
nearest = min(dists, key=dists.get)
sim_x, sim_y = np.array([4.0, 3.0]), np.array([4.2, 3.4])   # two similar toy vectors
sim_dot = float(sim_x @ sim_y)                     # large inner product
sim_dist = float(np.linalg.norm(sim_x - sim_y))    # small distance

if __name__ == "__main__":
    print("x = [3 4]:  L1 =", l1, " L2 =", l2, " Linf =", linf)
    print("norm of [2 3 6] =", n3)
    print("unit vector x/||x|| =", unit, " length:", np.linalg.norm(unit))
    print("distance [1 2 3] to [-1 -1 0] =", float(round(dist, 4)))
    print("||u+v|| =", float(round(tri_left, 4)), "<= ||u||+||v|| =", tri_right)
    print("gradient [6 8] clipped to norm 5 ->", g_clipped)
    print("distances to query:", {k: float(round(d, 2)) for k, d in dists.items()})
    print("nearest neighbour:", nearest)
