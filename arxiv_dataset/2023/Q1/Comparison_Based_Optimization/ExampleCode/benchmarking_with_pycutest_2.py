from __future__ import print_function
import copy
import pycutest
from matplotlib import pyplot as plt
from pycutest_utils import run_STP_pycutest, run_GLD_pycutest, run_CMA_pycutest, run_SCOBO_pycutest, \
    run_signOPT_pycutest

# Find unconstrained, variable-dimension problems.
probs = pycutest.find_problems(constraints='U', userN=True)
print('number of problems: ', len(probs))
print(sorted(probs))

for problem in probs:
    print(problem + ': ' + str(pycutest.problem_properties(problem)))
# functions to run.


# STP.
def run_STP_pycutest_newfile(problem, x0, function_budget):
    """ STP. """
    print('RUNNING ALGORITHM STP....')
    p = problem
    # direction_vector_type = 0  # original.
    direction_vector_type = 1  # gaussian.
    # direction_vector_type = 2  # uniform from sphere.
    # direction_vector_type = 3  # rademacher.
    target_fun_val = 0.01 * p.obj(x0)
    # STP instance.
    stp_f_vals, stp_function_evals = run_STP_pycutest(p,
                                                      copy.copy(x0),
                                                      function_budget,
                                                      target_fun_val)
    return stp_f_vals


# GLD.
def run_GLD_pycutest_newfile(problem, x0, function_budget):
    """ GLD. """
    print('RUNNING ALGORITHM GLD....')
    p = problem
    target_fun_val = 0.01 * p.obj(x0)
    gld_f_vals, gld_function_evals = run_GLD_pycutest(p,
                                                      copy.copy(x0),
                                                      function_budget,
                                                      target_fun_val)
    return gld_f_vals


# SignOPT.
def run_signOPT_pycutest_newfile(problem, x0, function_budget):
    """ SignOPT. """
    print('RUNNING ALGORITHM SIGNOPT....')
    p = problem
    target_fun_val = 0.01 * p.obj(x0)
    signopt_f_vals, signopt_function_evals = run_signOPT_pycutest(p,
                                                                  copy.copy(x0),
                                                                  function_budget,
                                                                  target_fun_val)
    return signopt_f_vals


# SCOBO.
def run_SCOBO_pycutest_newfile(problem, x0, function_budget):
    # SCOBO.
    print('RUNNING ALGORITHM SCOBO....')
    p = problem
    # SCOBO instance.
    target_fun_val = 0.01 * p.obj(x0)
    scobo_f_vals, scobo_function_evals = run_SCOBO_pycutest(p,
                                                            copy.copy(x0),
                                                            function_budget,
                                                            target_fun_val)
    return scobo_f_vals


# CMA.
def run_CMA_pycutest_newfile(problem, x0, function_budget):
    """ CMA. """
    print('RUNNING ALGORITHM CMA....')
    p = problem
    # CMA instance.
    target_fun_val = 0.01 * p.obj(x0)
    cma_f_vals, cma_function_evals = run_CMA_pycutest(p,
                                                      copy.copy(x0),
                                                      function_budget,
                                                      target_fun_val)
    return cma_f_vals


# function evaluations (to be plotted).
stp_func_list = []
gld_func_list = []
signOPT_func_list = []
scobo_func_list = []
cma_func_list = []
three_problems = ['VAREIGVL', 'LUKSAN17LS', 'CHNRSNBM']

# for problem in problems list (you designate this yourself)....
for problem in three_problems:
    p_invoke_ = pycutest.import_problem(problem)
    x0_invoke_ = p_invoke_.x0
    print('dimension of problem: ', len(x0_invoke_))
    function_budget_ = 1000
    # STP.
    print('invoking STP in a loop....')
    stp_func_list = run_STP_pycutest_newfile(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # GLD.
    print('invoking GLD in a loop....')
    gld_func_list = run_GLD_pycutest_newfile(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # SignOPT.
    print('invoking SignOPT in a loop....')
    signOPT_func_list = run_signOPT_pycutest_newfile(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # SCOBO.
    print('invoking SCOBO in a loop....')
    scobo_func_list = run_SCOBO_pycutest_newfile(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')
    # CMA.
    print('invoking CMA in a loop....')
    cma_func_list = run_CMA_pycutest_newfile(p_invoke_, copy.copy(x0_invoke_), function_budget_)
    print('\n')

    # plot.
    plt.figure()
    # plots represent each algorithm's performance.

    """ New addition. """
    x = range(0, 1002, 2)
    x_2 = range(0, 1000, 2)
    x_3 = range(0, 1010, 10)
    scobo_len = int(1000 / (len(scobo_func_list)-1))
    print('SCOBO LENGTH: ', scobo_len)
    x_4 = range(0, 1000+scobo_len, scobo_len)
    x_5 = range(0, 980, int(1000 / len(cma_func_list)))
    print('stp: ', len(stp_func_list))
    print('gld: ', len(gld_func_list))
    print('signOPT: ', len(signOPT_func_list))
    print('scobo: ', len(scobo_func_list))
    print('cma: ', len(cma_func_list))

    plt.plot(x, stp_func_list, color='orange', label='STP')
    plt.plot(x_2, gld_func_list, color='blue', label='GLD')
    plt.plot(x_3, signOPT_func_list, color='black', label='SignOPT')
    plt.plot(x_4, scobo_func_list, color='purple', label='SCOBO')
    plt.plot(x_5, cma_func_list, color='green', label='CMA')
    # name axes & show graph.
    plt.xlabel('number of oracle queries')
    plt.ylabel('function values')
    plt.legend()
    plt.title('problem: ' + str(problem))
    plt.show()
    plt.close()
