import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "glaberish_exp")
log_filepath = os.path.join(log_directory, "log_exp_glaberish.txt")

if os.path.exists(log_directory):
    pass
else:
    os.system(f"mkdir -p {log_directory}")

my_seed = 1337
save_images = "0"
log_filepath_glaberish = os.path.join(log_directory, "glaberish_exp.txt")
glider_patterns = f" s11_config_evolved_ca_slow_glider000 "\
	"s613_s613_frog000 s613_fast_wobble_glider000 "\
	"s643_s643_frog000 s643_s643_3wide_wobble_glider000 " 
system_type = "CCA"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "1 101"
dx_bounds = "0.01"
time_limit = 14400 # 4 hour experiment

gain_limits = "0.30"

my_cmd = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 53809de1ced462f4b6bc517e388f5ec116c43901 "\
        f"--gain_limits {gain_limits}  -r {my_seed} " \
        f"-k {kernel_radius_bounds} > {log_filepath}"

print(f"Commence experiment with command {my_cmd}")
os.system(my_cmd)
