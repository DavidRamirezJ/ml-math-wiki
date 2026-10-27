"""Vector demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

u = np.array([3, 1])
v = np.array([1, 2])
add = u + v                      # component by component
u2 = np.array([2, 1])
v2 = np.array([1, 2])
scaled = 2 * u2
comb = 2 * u2 + 1 * v2           # a linear combination
# find a, b with a*u2 + b*v2 = (5, 4): put u2, v2 as columns and solve
ab = np.linalg.solve(np.column_stack([u2, v2]), np.array([5, 4]))
img1 = np.array([1.0, 0.0, 0.0, 1.0])   # toy 2x2 image, pixels in a row
img2 = np.array([0.0, 1.0, 0.5, 0.5])
blend = 0.5 * img1 + 0.5 * img2

if __name__ == "__main__":
    print("u + v        =", add)
    print("2 * u        =", scaled, "  (u = [2 1])")
    print("-1 * u       =", -1 * u2)
    print("2*u + 1*v    =", comb)
    print("a, b for [5 4] =", ab)
    print("img1         =", img1)
    print("img2         =", img2)
    print("0.5*img1 + 0.5*img2 =", blend)
