##### Measuring Diagnostic Test Performance Using Imperfect Reference Tests: A Partial Identification Approach #####
### Replication script
## Developed by: Filip Obradović. Email: obradovicfilip@u.northwestern.edu

""" Replication script for generating all results in the paper.

    Procedure is computationally intensive. The script can take several hours to complete depending on the available
    resources.
"""


############################################ Import Functions ##########################################################

import functions

############################################# Set Parameters ###########################################################

seed = 100           # Set seed for replicability
grid_steps = 316     # 316*316 grid imposed over parameter space for theta
gridsteps_s = 10     # 10 points imposed over set S, if not a singleton
                     # In total 316*316*10 points in the grid over parameter space.

############################################### Study Data #############################################################

## Sources
# EUA Study - https://www.fda.gov/media/141570/download
# Shah et al (2021) - https://academic.oup.com/cid/article/73/Supplement_1/S54/6257223

data = [['EUA_study', 99, 5, 18, 338], ['Shah_symptomatic', 199, 2, 44, 684], ['Shah_asymptomatic', 33, 5, 15, 824]]

# Data in the form ['study', positives on both tests - t1r1, positives only on index - t1r0,
#               positives only on reference - t0r1, negatives on both tests - t0r0]

############################################### Computation ############################################################

### s1 assumed to be 0.9

s1 = 0.9
s0 = 1

for d in data:
    graph_name,t1r1,t1r0,t0r1,t0r0 = d
    graph_name = graph_name+"_known_exact"
    functions.calculate(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0=False, wrongly_agree_1=True,
                        grid_steps_conf=grid_steps, gridsteps_s=gridsteps_s,
                        method='2', boot_samples=500, parallel=True, num_threads=-1, filename=graph_name, seed = seed)

### s1 assumed to be in [0.8,0.9]

s1 = [0.8,0.9]
s0 = 1

for d in data:
    graph_name, t1r1, t1r0, t0r1, t0r0 = d
    graph_name = graph_name + "_known_imperf"
    functions.calculate(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0=False, wrongly_agree_1=True,
                        grid_steps_conf=grid_steps, gridsteps_s=gridsteps_s,
                        boot_samples=500, parallel=True, num_threads=-1, filename=graph_name, seed = seed)


# ############################################# Method Comparison ########################################################

### s1 assumed to be 0.9

s1 = 0.9
s0 = 1

for d in data:
    graph_name,t1r1,t1r0,t0r1,t0r0 = d
    functions.compare(s1, s0, t1r1, t1r0, t0r1, t0r0, wrongly_agree_0=False, wrongly_agree_1=True,
                      grid_steps_estimate=100,filename=graph_name, include_apparent=True)

