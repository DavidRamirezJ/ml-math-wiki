"""Basis & rank demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

a, b, c = np.array([2, 1]), np.array([1, 2]), np.array([3, 3])
abc = np.column_stack([a, b, c])            # vectors as columns
zero = a + b - c                            # weights 1, 1, -1 give zero -> dependent
B = np.column_stack([[1, 1], [-1, 1]])      # new basis b1 = (1, 1), b2 = (-1, 1)
p = np.array([1, 3])
coords = np.linalg.solve(B, p)              # c1*b1 + c2*b2 = p
R1 = np.outer([1, 2, 0], [1, -1, 3])        # columns (1,2,0), (-1,-2,0), (3,6,0)
R2 = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 2]])   # column 3 = column 1 + column 2
I3 = np.eye(3)
meters = np.array([1.2, 0.8, 2.0])          # toy table: length m, length cm, weight kg
table = np.column_stack([meters, 100 * meters, [5, 3, 9]])
rk = np.linalg.matrix_rank

if __name__ == "__main__":
    print("a + b - c =", zero)
    print("rank [a b c] =", rk(abc))
    print("coordinates of p in basis b1, b2 =", coords)
    print("R1 =", R1.tolist())
    print("rank R1 =", rk(R1))
    print("rank R2 =", rk(R2))
    print("rank I3 =", rk(I3))
    print("toy table columns: m, cm, kg")
    print(table)
    print("rank of toy table =", rk(table))
