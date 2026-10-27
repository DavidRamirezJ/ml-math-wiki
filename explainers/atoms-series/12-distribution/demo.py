"""Probability distribution demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

faces = np.arange(1, 7)
sums = np.add.outer(faces, faces)                 # 6x6 grid of all 36 outcomes
counts = np.bincount(sums.ravel(), minlength=13)[2:]
p_sum = counts / 36                               # distribution of X = sum of two dice
# toy joint table: rows = weather (rain, no rain), columns = (umbrella, no umbrella)
joint = np.array([[0.25, 0.05],
                  [0.10, 0.60]])
p_w = joint.sum(axis=1)                           # sum rule: add across each row
p_u = joint.sum(axis=0)                           # add down each column
p_u_given_rain = joint[0] / p_w[0]                # conditional: rain row, renormalized
product = p_u_given_rain[0] * p_w[0]              # product rule gives back 0.25
indep = np.outer(p_w, p_u)                        # what the table would be if independent
bayes = p_u_given_rain[0] * p_w[0] / p_u[0]       # p(rain | umbrella)

if __name__ == "__main__":
    print("dice sum counts 2..12:", counts)
    print("P(X=7) =", round(p_sum[5], 4), "  total =", p_sum.sum())
    print("joint table:\n", joint, "  total =", joint.sum())
    print("p(weather)  =", p_w, "   p(umbrella) =", p_u)
    print("p(umbrella | rain) =", np.round(p_u_given_rain, 3))
    print("p(umbrella|rain) * p(rain) =", round(product, 4))
    print("p(rain) * p(umbrella) =", round(indep[0, 0], 4), " vs joint", joint[0, 0])
    print("Bayes p(rain | umbrella) =", round(bayes, 3))
