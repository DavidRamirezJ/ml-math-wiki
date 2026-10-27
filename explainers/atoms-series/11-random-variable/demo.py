"""Random variable demo — real NumPy output shown in the video. Seeded, so it repeats exactly."""
import numpy as np

rng = np.random.default_rng(7)                    # seeded random number generator
dice = rng.integers(1, 7, size=(10000, 2))        # 10,000 rolls of two dice
X = dice.sum(axis=1)                              # random variable X = sum of the two dice
counts = np.bincount(X, minlength=13)[2:]         # how often each sum 2..12 appeared
frac_ge10 = (X >= 10).mean()                      # fraction of runs where X >= 10
frac_7 = (X == 7).mean()
coin = rng.integers(0, 2, size=10)                # Y = 1 if heads else 0, ten flips

if __name__ == "__main__":
    print("first 6 rolls :", dice[:6].tolist())
    print("their sums X  :", X[:6])
    print("sums 2..12 seen:")
    print(counts)
    print("total runs    :", counts.sum())
    print("share X >= 10 :", frac_ge10, " (counting: 6/36 = 0.1667)")
    print("share X == 7  :", frac_7, "  (counting: 6/36 = 0.1667)")
    print("ten coin flips Y:", coin)
