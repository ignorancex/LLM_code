# -*- coding: utf-8 -*-
# This script repeatedly generates independent pseudodata and records the chi2 at the best
# fit point. It belongs to Fig D.1 in the smefit3 paper.

import subprocess

projection_runcard = "./run_projection.yaml"

fit_runcard = "./run_closure.yaml"

# run fits repeatedly on different pseudo datasets
n_exp = 1
for i in range(n_exp):
    # run projection module
    subprocess.run(["smefit", "PROJ", "--closure", projection_runcard])

    # run analytic module without samples
    subprocess.run(["smefit", "A", fit_runcard])
