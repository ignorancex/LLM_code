import random
from simanneal import Annealer
import numpy as np

class SubsetOptimization(Annealer):
    def __init__(self, x, y, k, score_function, seed):
        random.seed(seed)
        self.x = x
        self.y = y
        self.k = k
        self.score_function = score_function
        self.n = len(x)
        initial_state = random.sample(range(self.n), k)
        super().__init__(initial_state)

    def energy(self):
        subset = self.state
        x_subset = [self.x[i] for i in subset]
        y_subset = [self.y[i] for i in subset]
        return -self.score_function(x_subset, y_subset)

    def move(self):
        subset = self.state
        swap_out_idx = random.randrange(self.k)
        swap_in_elem = random.choice([i for i in range(self.n) if i not in subset])
        subset[swap_out_idx] = swap_in_elem

def fit_ransac(x, y, min_samples):
    # Fit line using all data
    lr = linear_model.LinearRegression()
    lr.fit(x, y)

    # Robustly fit linear model with RANSAC algorithm
    ransac = linear_model.RANSACRegressor(min_samples=min_samples)
    ransac.fit(x, y)

    # Get R2 score
    r2 = ransac.score(x, y)

    return r2
