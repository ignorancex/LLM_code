##### Measuring Diagnostic Test Performance Using Imperfect Reference Tests: A Partial Identification Approach #####
### Calculation script
## Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu

""" Script intended for applications. Yields estimated identified sets for the true sensitivity and specificity.
    Forms confidence sets for sensitivity and specificity.

    Procedure is computationally intensive. It can take up to several hours to complete depending on the available
    resources and chosen parameters. Example below takes 10 minutes to complete on a Ryzen 9 3900X with 12 cores.

    The example is for the Abbott BinaxNow COVID-19 Ag EUA study results.
"""

############################################ Import Functions ##########################################################

import functions

############################################# Set Parameters ###########################################################

### Computational parameters

grid_steps = 316            # Number of points in the grid in EACH dimension for theta.
                            # Total grid points grid_steps^2. Integer scalar.
gridsteps_s = 10            # Number of points in the grid in EACH dimension for S. Assumes S is rectangular.
                            # If only one of s1 or s0 known exactly, total number of points in grid is gridsteps_s,
                            # otherwise gridsteps_s^2. Integer scalar.
                            # In total, grid_steps*grid_steps*gridsteps_s points in the grid over parameter space below.

boot_samples = 500          # Number of bootstrap draws for the RSW testing procedure. Integer scalar.
parallel = True             # Parallelizes the procedure over multiple threads. May reduce computation time significantly.
num_threads = -1            # Sets number of used threads to maximum available. Can be changed to other integer values.
graph_name = 'EUA_example'  # Name of produced graph. Saved in the 'Graphs' folder.
include_apparent = False    # Plots estimates and 95% Clopper-Pearson projection confidence intervals for apparent measures.

### Test properties

wrongly_agree_0 = False     # True if tests have a tendency to wrongly agree for y=0
wrongly_agree_1 = True      # True if tests have a tendency to wrongly agree for y=1

s1 = 0.9 # Assumed sensitivity of the reference test. Can be a float or interval (list). (e.g. s1=0.9 or s1=[0.8,0.9])
s0 = 1   # Assumed specificity of the reference test. Can be a float or interval (list). (e.g. s0=0.9 or s0=[0.8,0.9])

############################################### Study Data #############################################################

t1r1 = 99   # Number of positives on both tests
t1r0 = 5    # Number of participants positive on the index and negative on the reference test
t0r1 = 18   # Number of participants negative on the index and positive on the reference test
t0r0 = 338  # Number of negatives on both tests

############################################### Computation ############################################################

estimated_set, conf_set = functions.calculate(s1, s0, t1r1, t1r0, t0r1, t0r0,
                                              wrongly_agree_0=wrongly_agree_0, wrongly_agree_1=wrongly_agree_1,
                                              grid_steps_conf=grid_steps, gridsteps_s=gridsteps_s,
                                              boot_samples=boot_samples, parallel=parallel, num_threads=num_threads,
                                              filename=graph_name, include_apparent = include_apparent)



