import numpy as np
import mlx.core as mx


class HopfieldNetwork:
    def __init__(self, neurons, polydegree, continous=False):
        self.neurons = neurons
        self.polydegree = polydegree
        self.continuous = continous
        mx.set_default_device(mx.cpu)

        excitation = np.random.choice([-1, 1], size=self.neurons)
        self.excitation: mx.array = mx.array(excitation).astype(mx.float32)

        # store memories
        self.memories = mx.array([]).astype(mx.float32)
        self.energy = 0

    def __interaction_function(self, x):
        # polynomial energy function (not rectified polynomial!) Hopfield & Krotov 2016
        if self.polydegree is None:
            return mx.exp(x)
        else:
            x_scaled = mx.divide(x, 10000.0)
            return mx.power(x_scaled, self.polydegree)

    # async update, one neuron after the other, selected randomly, binary values only
    def update(self, state):
        # copy to avoid call by reference
        state_copy = mx.array(state)
        self.excitation = state_copy

        for i in range(self.neurons):
            jsum = mx.sum(
                mx.where(
                    mx.expand_dims(mx.arange(self.neurons), 1) != i,
                    self.excitation[:, None] * self.memories.T,
                    0,
                ),
                axis=0,
            )
            pos_terms = self.__interaction_function(self.memories[:, i] + jsum)
            neg_terms = self.__interaction_function(-self.memories[:, i] + jsum)
            result = mx.sum(pos_terms - neg_terms)

            self.__activation_function(result, i)

    def __activation_function(self, result, i):
        if not self.continuous:
            self.excitation[i] = mx.array(1.0) if result >= 0 else mx.array(-1.0)
        else:
            beta = mx.reciprocal(mx.array([self.polydegree]))
            self.excitation[i] = mx.tanh(beta * result)

    def learn(self, memories):
        self.memories = mx.array(memories).astype(mx.float32)

    def get_state(self):
        return self.excitation

    def get_num_neurons(self):
        return self.neurons

    def get_poly_degree(self):
        return self.polydegree

    def set_polydegree(self, polydegree):
        self.polydegree = polydegree

    def __str__(self):
        return "Energy of Network:" + str(self.energy)
