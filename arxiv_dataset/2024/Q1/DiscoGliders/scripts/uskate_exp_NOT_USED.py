import os
"""
Experiment for U-Skate World glider persistence, 
using normalized correlation peaks as a proxy for 
recognizing glider pattern integrity. 

IMPORTANT NOTE: correlation is unreliable, as becomes readily apparent
when examining audit images saved in this experiment. 

Disco Gliders instead used manual assessment of glider persistence.
See the u_skate_world.ipynb notebook for a manual grid search. 
"""

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "..", "logs", "uskate_exp")

# u-skate patterns
log_filepath = os.path.join(log_directory, "log_uskate_exp.txt")
glider_patterns = f"uskate_glider001 uskate_daedalus001 uskate_berrycup001 "
system_type = "RxnDfn"
save_images = "1"
max_steps = "131072"
device = "cuda"
time_limit = 14400

kernel_radius_bounds = "1 1"
dx_bounds = "0.001 0.015"

# gliders in u-skate world travel in straight lines, 
# the failure condition is based max correlation with the starting pattern normalized to 
# max autocorrelation of the starting pattern with itself
correlation_limits = "0.8 5.0"

my_cmd = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 53809de1ced462f4b6bc517e388f5ec116c43901 "\
        f"--correlation_limits {correlation_limits} "\
        f"-k {kernel_radius_bounds} > {log_filepath}"

print(f"Commence experiment with command {my_cmd}")
os.system(my_cmd)
