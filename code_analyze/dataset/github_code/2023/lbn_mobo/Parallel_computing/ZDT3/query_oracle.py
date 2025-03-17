import numpy as np
from pymoo.problems import get_problem

class Oracle_function:
    def __init__(self):
        self.problem = get_problem("zdt3")


    def Oracle_eval(self, designs):

        performance = self.problem.evaluate(designs)
        return performance