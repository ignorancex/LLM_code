import math
import random
import numpy as np

class Model:
    def __init__(self, width, height, N, periodic, heuristic):
        # Initialize the grid dimensions, number of possible tiles (T), and heuristic
        self.MX = width
        self.MY = height
        self.N = N
        self.periodic = periodic
        self.heuristic = heuristic
        
        self.T = 2  # Number of possible states (tiles), can be modified depending on your tile set
        self.wave = None
        self.compatible = None
        self.propagator = None
        self.weights = None
        self.weight_log_weights = None
        self.distribution = None
        self.observed = None
        self.stack = None
        self.stacksize = 0
        self.observed_so_far = 0
        self.sums_of_ones = None
        self.sums_of_weights = None
        self.sums_of_weight_log_weights = None
        self.entropies = None
        self.sum_of_weights = 0
        self.sum_of_weight_log_weights = 0
        self.starting_entropy = 0

    def init(self):
        self.wave = np.full((self.MX * self.MY, self.T), True)
        self.compatible = np.zeros((self.MX * self.MY, self.T, 4), dtype=int)
        self.distribution = np.zeros(self.T)
        self.observed = np.full(self.MX * self.MY, -1)

        self.weight_log_weights = np.zeros(self.T)
        self.sums_of_ones = np.zeros(self.MX * self.MY, dtype=int)
        self.sums_of_weights = np.zeros(self.MX * self.MY)
        self.sums_of_weight_log_weights = np.zeros(self.MX * self.MY)
        self.entropies = np.zeros(self.MX * self.MY)

        self.stack = []
        self.stacksize = 0
        self.observed_so_far = 0

        # Initialize weights, propagator, and entropy-related calculations
        self.weights = np.ones(self.T)  # Modify with actual weights for each tile
        for t in range(self.T):
            self.weight_log_weights[t] = self.weights[t] * math.log(self.weights[t])
            self.sum_of_weights += self.weights[t]
            self.sum_of_weight_log_weights += self.weight_log_weights[t]

        self.starting_entropy = math.log(self.sum_of_weights) - self.sum_of_weight_log_weights / self.sum_of_weights

        # Fill sums arrays and entropies
        for i in range(self.MX * self.MY):
            self.sums_of_ones[i] = self.T
            self.sums_of_weights[i] = self.sum_of_weights
            self.sums_of_weight_log_weights[i] = self.sum_of_weight_log_weights
            self.entropies[i] = self.starting_entropy

    def run(self, seed, limit):
        if self.wave is None:
            self.init()

        self.clear()
        random.seed(seed)

        for l in range(limit if limit >= 0 else float('inf')):
            node = self.next_unobserved_node(random)
            if node >= 0:
                self.observe(node, random)
                success = self.propagate()
                if not success:
                    return False
            else:
                for i in range(self.MX * self.MY):
                    for t in range(self.T):
                        if self.wave[i][t]:
                            self.observed[i] = t
                            break
                return True
        return True

    def next_unobserved_node(self, random):
        if self.heuristic == 'Scanline':
            for i in range(self.observed_so_far, self.MX * self.MY):
                if self.sums_of_ones[i] > 1:
                    self.observed_so_far = i + 1
                    return i
            return -1

        min_entropy = float('inf')
        argmin = -1
        for i in range(self.MX * self.MY):
            if self.sums_of_ones[i] > 1:
                entropy = self.entropies[i] if self.heuristic == 'Entropy' else self.sums_of_ones[i]
                if entropy <= min_entropy:
                    min_entropy = entropy
                    argmin = i
        return argmin

    def observe(self, node, random):
        w = self.wave[node]
        self.distribution = np.where(w, self.weights, 0.0)
        r = np.random.choice(np.arange(self.T), p=self.distribution / self.distribution.sum())
        for t in range(self.T):
            if w[t] != (t == r):
                self.ban(node, t)

    def propagate(self):
        while self.stacksize > 0:
            i1, t1 = self.stack.pop()
            x1, y1 = divmod(i1, self.MX)

            for d in range(4):
                x2, y2 = x1 + dx[d], y1 + dy[d]
                if not self.periodic and (x2 < 0 or y2 < 0 or x2 + self.N > self.MX or y2 + self.N > self.MY):
                    continue

                x2 %= self.MX
                y2 %= self.MY

                i2 = x2 + y2 * self.MX
                p = self.propagator[d][t1]
                compat = self.compatible[i2]

                for t2 in p:
                    comp = compat[t2]
                    comp[d] -= 1
                    if comp[d] == 0:
                        self.ban(i2, t2)

        return self.sums_of_ones[0] > 0

    def ban(self, i, t):
        self.wave[i][t] = False
        for d in range(4):
            self.compatible[i][t][d] = 0
        self.stack.append((i, t))
        self.stacksize += 1

        self.sums_of_ones[i] -= 1
        self.sums_of_weights[i] -= self.weights[t]
        self.sums_of_weight_log_weights[i] -= self.weight_log_weights[t]

        sum_weights = self.sums_of_weights[i]
        self.entropies[i] = math.log(sum_weights) - self.sums_of_weight_log_weights[i] / sum_weights

    def clear(self):
        for i in range(self.MX * self.MY):
            for t in range(self.T):
                self.wave[i][t] = True
                for d in range(4):
                    self.compatible[i][t][d] = len(self.propagator[d][t])
            self.sums_of_ones[i] = self.T
            self.sums_of_weights[i] = self.sum_of_weights
            self.sums_of_weight_log_weights[i] = self.sum_of_weight_log_weights
            self.entropies[i] = self.starting_entropy
            self.observed[i] = -1

        self.observed_so_far = 0

    def save(self, filename):
        # Placeholder for the save function, depending on how you want to save the generated grid
        np.save(filename, self.observed)

# Direction vectors for neighbors
dx = [-1, 0, 1, 0]
dy = [0, 1, 0, -1]

# Example usage
width, height = 5, 5  # Define the size of the grid
model = Model(width, height, N=2, periodic=True, heuristic='Entropy')
model.run(seed=42, limit=100)
