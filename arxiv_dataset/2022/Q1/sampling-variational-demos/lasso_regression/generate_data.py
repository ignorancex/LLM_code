import json
import numpy as np

N, P = 100, 30
sigmaY = 1.0
beta0 = 0.0

# True values of beta drawn from a laplace (double exponential) distribution
beta = np.random.exponential(scale=1.0, size=(P,))
# Randomly flip signs to make it a double-exponential
beta = beta * np.sign(np.random.random(P) - 0.5)

X = np.random.randn(N, P)
Y = beta0 + X @ beta + np.random.randn(N) * sigmaY

data = {
    'N': N,
    'P': P,
    'X': X.tolist(),
    'Y': Y.tolist()
}

with open("data.json", "w") as f:
    json.dump(data, f)

gt = {
    'N': N,
    'P': P,
    'sigmaY': sigmaY,
    'beta0': beta0,
    'beta': beta.tolist()
}

with open("ground_truth.json", "w") as f:
    json.dump(gt, f)