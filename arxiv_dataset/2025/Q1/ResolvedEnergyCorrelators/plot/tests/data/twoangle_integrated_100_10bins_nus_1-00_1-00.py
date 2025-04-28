import numpy as np

# ==================================
# Information
# ==================================
# Function call ```./write/new_enc/integrated_twoangle --use_opendata --use_deltaR --use_pt --weights 1 1 --minbin -6 --n_events 100 --nbins 10 --file_prefix 100_10bins ```

# Number of events:
n_events = 100
# Process
energy = 14000
level = "data"
pid_1, pid_2 = 2212, 2212
outstate_str = "qcd"
# Weight information:
weight = (1, 1)
# CMS 2011A Jet Dataset:
jet_alg = "anti-kt"
jet_scheme = "E?"
jet_rad = 0.5

# ==================================
# Output Histogram:
# ==================================
theta1_edges = [
	0, 1e-06, 5.70493e-06, 3.25462e-05, 0.000185674, 0.00105925, 0.00604296, 0.0344747, 0.196675, 1.12202, np.inf
]

theta1_centers = [
	0, 2.3885e-06, 1.36262e-05, 7.77365e-05, 0.000443481, 0.00253003, 0.0144336, 0.0823427, 0.469759, np.inf
]

hist = [
	0.03579857214, 0, 0, 5.467058271e-07, 0.004989000238, 0.08834883996, 0.4322332971, 0.4787213306, 0.2119701217, 0
]