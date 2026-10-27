"""Variance & covariance demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

A = np.array([4, 5, 5, 5, 6])          # toy quiz scores, class A
B = np.array([2, 4, 5, 6, 8])          # toy quiz scores, class B (same mean)
mu = B.mean()
dev = B - mu                           # deviations from the mean
var_B = np.mean(dev ** 2)              # E[(X - mu)^2]
var_A = np.var(A)
sd_B = np.std(B)
shortcut = np.mean(B ** 2) - mu ** 2   # E[X^2] - mu^2

x = np.array([1, 3, 5, 7])             # toy: hours studied
y = np.array([2, 3, 6, 5])             # toy: quiz score
cov_xy = np.mean((x - x.mean()) * (y - y.mean()))
C_n = np.cov(x, y, bias=True)          # divide by n
C_n1 = np.cov(x, y)                    # default: divide by n - 1

rng = np.random.default_rng(0)         # three toy scatter plots
t = rng.normal(size=40)
noise = rng.normal(size=40)
y_pos, y_neg, y_none = t + 0.5 * noise, -t + 0.5 * noise, rng.normal(size=40)
covs = [float(np.cov(t, yy, bias=True)[0, 1]) for yy in (y_pos, y_neg, y_none)]

z = (B - mu) / sd_B                    # standardization (z-score)

if __name__ == "__main__":
    print("mean A, mean B   =", A.mean(), mu)
    print("deviations of B  =", dev, " average:", dev.mean())
    print("squared dev.     =", dev ** 2)
    print("Var A, Var B     =", round(var_A, 2), var_B)
    print("std of B         =", sd_B)
    print("E[X^2] - mu^2    =", shortcut)
    print("Cov(x, y)        =", cov_xy)
    print("cov matrix (1/n) =", C_n.tolist())
    print("cov matrix (n-1) =", np.round(C_n1, 3).tolist())
    print("scatter covs     =", [round(c, 2) for c in covs])
    print("z-scores of B    =", z, " var:", z.var())
