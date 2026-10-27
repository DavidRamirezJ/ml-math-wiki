"""Matrix demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

X = np.array([[120, 3, 15],      # toy houses: area m², bedrooms, age in years
              [80, 2, 30],
              [150, 4, 5]])
A = np.array([[2, 1], [1, 3]])
x = np.array([1, 2])
Ax = A @ x                                   # matrix times vector
colview = x[0] * A[:, 0] + x[1] * A[:, 1]    # linear combination of the columns
S = np.array([[2, 0], [0, 1]])               # stretch x by 2
R = np.array([[0, -1], [1, 0]])              # rotate 90 degrees counterclockwise
p = np.array([1, 2])
RS, SR = R @ S, S @ R                        # S first then R, and R first then S
P = np.array([[1, 2], [3, 4]])
Q = np.array([[5, 6], [7, 8]])
PQ, QP = P @ Q, Q @ P
M = np.array([[1, 2, 3], [4, 5, 6]])
I = np.eye(2, dtype=int)
w = np.array([2, 10, -1])                    # toy weights
Xw = X @ w

if __name__ == "__main__":
    print("X.shape =", X.shape)
    print("A @ x =", Ax)
    print("1*col1 + 2*col2 =", colview)
    print("S @ p =", S @ p)
    print("R @ p =", R @ p)
    print("R @ S =", RS.tolist())
    print("S @ R =", SR.tolist())
    print("P @ Q =", PQ.tolist())
    print("Q @ P =", QP.tolist())
    print("M.T =", M.T.tolist())
    print("I @ P == P:", bool((I @ P == P).all()))
    print("X @ w =", Xw)
