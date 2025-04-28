import numpy as np

# ==================================
# Information
# ==================================
# Function call ```./write/new_enc/oneangle --use_opendata --use_deltaR --use_pt --weights 1 --minbin -6 --n_events 100 --nbins 10 --file_prefix 100_10bins ```

# Number of events:
n_events = 100
# Process
energy = 14000
level = "data"
pid_1, pid_2 = 2212, 2212
outstate_str = "qcd"
# Weight information:
weight = (1)
# CMS 2011A Jet Dataset:
jet_alg = "anti-kt"
jet_scheme = "E?"
jet_rad = 0.5

# ==================================
# Output Histogram:
# ==================================
theta1_edges = [
	0, 1e-06, 7.49894e-06, 5.62341e-05, 0.000421697, 0.00316228, 0.0237137, 0.177828, 1.33352, 10, np.inf
]

theta1_centers = [
	0, 2.73842e-06, 2.05353e-05, 0.000153993, 0.00115478, 0.00865964, 0.0649382, 0.486968, 3.65174, np.inf
]

hist = [
	0.1260420131, 0, 0, 3.404202272e-05, 0.03994673859, 0.3744280623, 0.4120938665, 0.1778219676, 0, 0
]