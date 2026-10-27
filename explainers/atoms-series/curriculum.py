"""ML Math Atoms — one video per study atom (wiki/maps/atoms-and-molecules.md).
(folder, title, wiki concept pages covered, axiom it comes from)"""
VIDEOS = [
 ("01-vector",          "Vector",                          ["vector", "vector-space"],                       "A"),
 ("02-matrix",          "Matrix",                          ["matrix", "matrix-multiplication"],              "A"),
 ("03-basis-and-rank",  "Basis & rank",                    ["linear-independence", "basis-and-rank"],        "A"),
 ("04-linear-mapping",  "Linear mapping",                  ["linear-mapping"],                               "A"),
 ("05-inner-product",   "Inner product",                   ["inner-product", "angle-between-vectors", "orthogonality"], "B"),
 ("06-norm",            "Norm",                            ["norm", "metric"],                               "B"),
 ("07-derivative",      "Derivative",                      ["derivative", "taylor-series"],                  "C"),
 ("08-gradient",        "Gradient",                        ["partial-derivative", "gradient"],               "C"),
 ("09-chain-rule",      "Chain rule",                      ["chain-rule", "jacobian"],                       "C"),
 ("10-exp-and-log",     "exp & log",                       ["log-probability"],                              "floor"),
 ("11-random-variable", "Random variable",                 ["random-variable"],                              "D"),
 ("12-distribution",    "Probability distribution",        ["probability-distribution", "statistical-independence"], "D"),
 ("13-expected-value",  "Expected value",                  ["expected-value-and-mean"],                      "D"),
 ("14-variance",        "Variance & covariance",           ["covariance-and-variance"],                      "D"),
 ("15-loss-function",   "Loss function",                   ["loss-function", "empirical-risk-minimization"], "E"),
 ("16-activation",      "Activation function",             ["activation-function", "relu"],                  "A"),
]
