import numpy as np

# ==================================
# Information
# ==================================
# Function call ```./write/new_enc/twoangle --use_opendata --use_deltaR --use_pt --weights 1 1 --minbin -6 --n_events 1000 --nphibins 1 --nbins 50 --file_prefix 1k_1phi_50bins ```

# Number of events:
n_events = 1000
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
