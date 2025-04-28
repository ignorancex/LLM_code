import os

script_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
log_directory = os.path.join(script_path, "logs", "lenia_exp")
log_filepath = os.path.join(log_directory, "log_exp_lenia.txt")

if os.path.exists(log_directory):
    pass
else:
    os.system(f"mkdir -p {log_directory}")

my_seed = 1337
save_images = "0"
# patterns from Lenia
log_filepath_lenia = os.path.join(log_directory, "exp_0x0r06_lenia.txt")
glider_patterns = f" orbium_orbium000 geminium_hydrogeminium_natans000 geminium2_wobble_glider000 "\
	"scutium_gravidus_single000 triscutium_solidus_triscutium000 "
system_type = "CCA"
max_steps = "1024"
device = "cuda"
kernel_radius_bounds = "1 101"
dx_bounds = "0.01"
time_limit = 14400 # 4 hour experiment

# gliders in Lenia tend to rotate a bit, so gain in mean cell value is used instead of correlation, 
gain_limits = "0.30"

my_cmd = f"python -m disco.walk -l {time_limit} -c {system_type} "\
        f"-g {glider_patterns}"\
        f"-s {save_images} -m {max_steps} -d {device}  -x {dx_bounds} "\
        f"-y 53809de1ced462f4b6bc517e388f5ec116c43901 "\
        f"--gain_limits {gain_limits}  -r {my_seed} " \
        f"-k {kernel_radius_bounds} > {log_filepath}"

print(f"Commence experiment with command {my_cmd}")
os.system(my_cmd)
