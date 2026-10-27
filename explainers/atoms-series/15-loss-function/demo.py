"""Loss function demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

y_true = np.array([2, 4, 6])               # toy labels
y_pred = np.array([1, 4, 8])               # toy predictions
sq = (y_true - y_pred) ** 2                # squared error per prediction
ab = np.abs(y_true - y_pred)               # absolute error per prediction

x = np.array([1, 2, 3, 4])                 # toy points for the line y = w * x
y = np.array([2, 5, 6, 8])
ws = np.linspace(0, 4, 401)                # try 401 slopes: 0.00, 0.01, ..., 4.00
L = np.array([np.mean((y - w * x) ** 2) for w in ws])
w_best = ws[L.argmin()]
L_at = {w: float(np.mean((y - w * x) ** 2)) for w in (1, 2, 3)}

xc = np.array([1, 2, 3, 4, 5, 6])          # toy 1-D classification
lab = np.array([0, 0, 1, 0, 1, 1])         # rule: predict 1 if x > b
bs = np.linspace(0, 7, 701)
L01 = np.array([np.mean((xc > b).astype(int) != lab) for b in bs])

xo = np.array([1, 2, 3, 4, 5])             # toy points with one outlier
yo = np.array([2, 4, 6, 8, 30])
Lsq = np.array([np.mean((yo - w * xo) ** 2) for w in ws])
Lab = np.array([np.mean(np.abs(yo - w * xo)) for w in ws])
w_sq, w_ab = ws[Lsq.argmin()], ws[Lab.argmin()]

if __name__ == "__main__":
    print("squared errors   =", sq, " mean:", round(sq.mean(), 2))
    print("absolute errors  =", ab, " mean:", ab.mean())
    print("L(1), L(2), L(3) =", [round(L_at[w], 2) for w in (1, 2, 3)])
    print("best slope w     =", round(w_best, 2), " loss:", round(L.min(), 3))
    print("0-1 loss values  =", sorted({float(v) for v in np.round(L01, 3)}))
    print("0-1 loss: 701 tries of b, only", len(set(np.round(L01, 3))), "different values")
    print("with outlier: squared  -> w =", round(w_sq, 2))
    print("with outlier: absolute -> w =", round(w_ab, 2))
