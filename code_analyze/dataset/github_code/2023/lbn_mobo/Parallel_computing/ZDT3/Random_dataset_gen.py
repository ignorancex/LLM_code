import numpy as np
from pymoo.problems import get_problem


def random_dset_gen(problem_name):

    init_samples = 1000
    problem = get_problem(problem_name)
    #################################################################

    input_size = 30
    random_design = np.random.rand(init_samples,input_size)



    Performance = problem.evaluate(random_design)

    random_design = random_design
    random_performance = Performance
    data = {'X':random_design, 'Y':random_performance} 
    return data
