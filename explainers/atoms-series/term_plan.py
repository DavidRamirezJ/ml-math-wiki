"""Which video OWNS each technical term (shows the yellow NEW TERM card).
Later videos use a blue reminder_card once; earlier videos must not card a later-owned term."""
OWNER = {
    # 01 vector
    "vector": "01", "component": "01", "scalar": "01", "linear combination": "01",
    "span": "01", "vector space": "01",
    # 02 matrix
    "matrix": "02", "matrix multiplication": "02", "transpose": "02", "identity matrix": "02",
    # 03 basis & rank
    "linear independence": "03", "basis": "03", "dimension": "03", "rank": "03",
    # 04 linear mapping
    "linear mapping": "04", "kernel (null space)": "04", "image (range)": "04",
    # 05 inner product
    "inner product": "05", "dot product": "05", "orthogonal": "05", "cosine similarity": "05",
    # 06 norm
    "norm": "06", "unit vector": "06", "distance": "06",
    # 07 derivative
    "derivative": "07", "limit": "07", "local linear approximation": "07",
    # 08 gradient
    "partial derivative": "08", "gradient": "08",
    # 09 chain rule
    "chain rule": "09", "composition": "09", "Jacobian": "09",
    # 10 exp & log
    "exponential": "10", "logarithm": "10",
    # 11 random variable
    "random variable": "11", "outcome": "11",
    # 12 distribution
    "probability distribution": "12", "sum rule": "12", "product rule": "12",
    "conditional probability": "12", "independence": "12",
    # 13 expected value
    "expected value": "13", "mean": "13",
    # 14 variance & covariance
    "variance": "14", "standard deviation": "14", "covariance": "14",
    # 15 loss function
    "loss function": "15", "parameter": "15", "optimization": "15",
    # 16 activation
    "activation function": "16", "ReLU": "16", "nonlinearity": "16",
}
