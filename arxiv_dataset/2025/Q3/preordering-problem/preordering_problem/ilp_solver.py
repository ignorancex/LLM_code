import gurobipy
import numpy as np
import matplotlib.pyplot as plt
from time import time
from preorder_odd_closed_walk_separation import separate


class Preorder:

    def __init__(self, costs: np.ndarray, binary: bool = False, suppress_log=False):
        assert costs.ndim == 2
        assert costs.shape[0] == costs.shape[1]

        self.n = costs.shape[0]
        self.costs = costs

        self.model = gurobipy.Model()

        if suppress_log:
            self.model.setParam('OutputFlag', 0)

        self.vars = np.empty((self.n, self.n), dtype=gurobipy.Var)
        self.var_type = gurobipy.GRB.BINARY if binary else gurobipy.GRB.CONTINUOUS
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.vars[i, j] = self.model.addVar(0, 1, 0, self.var_type, f"x_({i}, {j})")
        self.model.setObjective(sum(costs[i, j] * self.vars[i, j]
                                    for i in range(self.n) for j in range(self.n) if i != j),
                                gurobipy.GRB.MAXIMIZE)

        # activate lazy constraint
        self.model.Params.LazyConstraints = 1
        if not binary:
            # add dummy binary variable to support lazy constraints
            self.dummy = self.model.addVar(0, 1, 0, gurobipy.GRB.BINARY, "dummy")

        self.separate_odd_closed_walk = 0
        self.separate_all_triangle = True
        self.cb_time = 0.0

    def set_solution(self, adjacency: np.ndarray):
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.vars[i, j].Start = adjacency[i, j]

    def separate(self, _, where):
        if where != gurobipy.GRB.Callback.MIPSOL:
            return

        t_0 = time()
        x = np.ones((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    x[i, j] = self.model.cbGetSolution(self.vars[i, j])
        num_triangle = self.lazy_triangle(x)
        if num_triangle == 0:
            if self.separate_odd_closed_walk >= 5:
                self.lazy_odd_closed_walk(x)
            elif self.separate_odd_closed_walk >= 3:
                self.lazy_chorded3cycle(x)
        self.cb_time += time() - t_0

    def lazy_triangle(self, x):
        lhs = x.reshape((self.n, self.n, 1)) + x.reshape((1, self.n, self.n)) - x.reshape((self.n, 1, self.n))
        counter = 0
        if self.separate_all_triangle:
            idx = np.argwhere(lhs > 1 + 1e-3)
            for i, j, k in idx:
                self.model.cbLazy(self.vars[i, j] + self.vars[j, k] - self.vars[i, k] <= 1, "transitivity")
                counter += 1
        else:
            val = np.max(lhs, axis=1)
            for i, k in np.argwhere(val > 1 + 1e-3):
                j = np.argmax(lhs[i, :, k])
                self.model.cbLazy(self.vars[i, j] + self.vars[j, k] - self.vars[i, k] <= 1, "transitivity")
                counter += 1
        return counter

    def lazy_odd_closed_walk(self, x):
        for walk in separate(x, self.separate_odd_closed_walk):
            expr = 0
            for i in range(len(walk)):
                expr += self.vars[walk[i], walk[(i-1) % len(walk)]] - self.vars[walk[i], walk[(i-2) % len(walk)]]
            self.model.cbLazy(expr <= len(walk) // 2, "odd-closed-walk")

    def lazy_chorded3cycle(self, x):
        # compute n x n x n matrix t with t[i, j, k] = x[i, j] + x[j, k] + x[k, i]
        t = x.reshape((self.n, self.n, 1)) + x.reshape((1, self.n, self.n)) + x.T.reshape((self.n, 1, self.n))
        # compute left hand side with lhs[i, j, k] = t[i, j, k] - t[k, j, i]
        lhs = t - t.swapaxes(0, 2)
        idx = np.argwhere(lhs > 1 + 1e-3)
        for i, j, k in idx:
            if i < j < k or i < k < j:
                self.model.cbLazy(self.vars[i, j] + self.vars[j, k] + self.vars[k, i]
                                  - self.vars[k, j] - self.vars[j, i] - self.vars[i, k] <= 1, "chorded3cycle")

    def add_symmetric_constraints(self):
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.model.addConstr(self.vars[i, j] == self.vars[j, i], "symmetry")

    def add_anti_symmetric_constraints(self):
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.model.addConstr(self.vars[i, j] + self.vars[j, i] <= 1, "anti-symmetry")

    def add_totality_constraints(self):
        for i in range(self.n):
            for j in range(self.n):
                if i == j:
                    continue
                self.model.addConstr(self.vars[i, j] + self.vars[j, i] >= 1, "totality")

    def solve(self, time_limit=None):
        if time_limit is not None:
            self.model.setParam('TimeLimit', time_limit)
        self.model.optimize(self.separate)
        return self.model.objVal

    def get_variable_values(self):
        values = np.ones((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                if i != j:
                    values[i, j] = self.vars[i, j].X
        return values


def example():
    costs = np.array([
        [ 0,  2,  0,  0,  0, -1],
        [ 3,  0, -2,  0,  0,  4],
        [ 0,  5,  0, -1,  1,  0],
        [ 0,  0, -2,  0,  2, -1],
        [ 0, -3,  0, -2,  0,  3],
        [-3,  0,  0,  0, -1,  0]
    ])
    preorder = Preorder(costs, True)
    preorder.solve()

    from preordering_problem.drawing import PreorderPlot
    plotter = PreorderPlot(preorder.get_variable_values(), costs)
    plotter.plot()
    plt.show()


if __name__ == "__main__":
    example()
