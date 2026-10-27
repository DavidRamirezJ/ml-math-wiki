"""exp & log demo — real NumPy output shown in the video. Toy numbers only."""
import numpy as np

e = np.exp(1.0)
exp_rule = (np.exp(2) * np.exp(3), np.exp(5))          # e^2 * e^3 vs e^5
log_rule = (np.log(2) + np.log(3), np.log(6))          # ln 2 + ln 3 vs ln 6
tiny_log = np.log(0.0001)
p = np.full(1000, 0.01)                                 # 1000 probabilities of 0.01
product = np.prod(p)                                    # underflows to 0.0
sum_logs = np.sum(np.log(p))                            # fine: -4605.17
run = np.cumprod(p)
first_zero = int(np.argmax(run == 0.0)) + 1             # how many factors until 0.0
scores = np.array([2.0, 1.0, -1.0])                     # toy scores for softmax preview
probs = np.exp(scores) / np.exp(scores).sum()

if __name__ == "__main__":
    print("e                =", round(e, 6))
    print("e^2 * e^3        =", round(exp_rule[0], 4))
    print("e^5              =", round(exp_rule[1], 4))
    print("ln 2 + ln 3      =", round(log_rule[0], 4))
    print("ln 6             =", round(log_rule[1], 4))
    print("ln 0.0001        =", round(tiny_log, 2))
    print("product of 1000 x 0.01 =", product)
    print("sum of 1000 logs =", round(sum_logs, 2))
    print("product is 0.0 after", first_zero, "factors")
    print("softmax([2, 1, -1]) =", np.round(probs, 3))
