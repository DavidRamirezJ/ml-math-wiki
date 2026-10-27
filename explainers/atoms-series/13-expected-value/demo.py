"""Expected value demo — real NumPy output shown in the video. Seeded, toy numbers only."""
import numpy as np

faces = np.arange(1, 7)
p = np.full(6, 1 / 6)
e_die = np.sum(faces * p)                          # E[X] = sum of x * p(x) = 3.5
win = np.array([8.0, -2.0])                        # toy lottery: net win
p_win = np.array([0.1, 0.9])
e_lottery = np.sum(win * p_win)                    # -1.0
sums = np.add.outer(faces, faces).ravel()          # 36 equally likely outcomes
e_two = sums.mean()                                # E[X + Y] = 7
rng = np.random.default_rng(3)
rolls = rng.integers(1, 7, size=10000)
running = np.cumsum(rolls) / np.arange(1, 10001)   # running average of the rolls
checks = [10, 100, 1000, 10000]
data = np.array([2.0, 4.0, 4.0, 5.0, 10.0])        # toy data set

if __name__ == "__main__":
    print("E[die]          =", e_die)
    print("E[lottery win]  =", round(e_lottery, 2))
    print("E[two dice]     =", e_two)
    print("first 10 rolls  :", rolls[:10])
    for n in checks:
        print(f"average of {n:>5} rolls = {running[n - 1]:.4f}")
    print("mean of data    =", data.mean())
