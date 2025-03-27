"""
Author: Theresa Wagner <theresa.wagner@mathematik.tu-chemnitz.de>

Corresponding publication:
"A Preconditioned Interior Point Method for Support Vector Machines Using an
ANOVA-Decomposition and NFFT-Based Matrix–Vector Products"
by T. Wagner, John W. Pearson, M. Stoll (2023)

Execute this file to test the run files.
"""
################
# dynamically determine location if test.py and adjust paths accordingly
import os
import sys
# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Add the directory to sys.path to allow importing the run files
sys.path.insert(0, script_dir)

#################
import subprocess

# List of all scripts to execute
scripts = [
    "run_test_precond_timings.py",
    "run_test_IPMGMRES_iters.py",
    "run_test_IPMGMRES_tols.py",
    "run_test_dratio.py",
    "run_finaltest.py"
]

# Define testing arguments
dict_args = {"run_test_precond_timings.py": "--kernel 1 --Ndata 1000 --prec chol_greedy --rank 50",
             "run_test_IPMGMRES_iters.py" : "--kernel 1 --Ndata 1000 --prec chol_greedy --rank 50",
             "run_test_IPMGMRES_tols.py" : "--kernel 1 --Ndata 1000 --prec chol_greedy --rank 50 --iRS 5",
             "run_test_dratio.py" : "--kernel 1 --Ndata 1000 --prec chol_greedy --rank 50 --iRS 5 --data cod_rna",
             "run_finaltest.py" : "--kernel 1 --Ndata 1000 --prec chol_greedy --rank 50 --iRS 5 --data cod_rna"
}

# Track successfully tested scripts
tested_scripts = []

# Loop through each script and execute it
for script in scripts:
    try:
        # Construct the absolute path to the specific file in the test folder
        file_path = os.path.join(script_dir, script)
        print(f"Running {script}...")
        # Split the arguments string into a list and prepend the script name
        args = ["python", file_path] + dict_args[script].split()
        print("args:", args)
        # Run the script
        subprocess.run(args, check=True)
        print(f"Finished {script}\n")
        tested_scripts.append(script)
    except subprocess.CalledProcessError as e:
        print(f"Error occurred while running {script}: {e}")
        break

else:
    # If no errors occurred in the loop
    print(f"All scripts were tested successfully! Tested scripts: {', '.join(tested_scripts)}")

# In case of errors, still show the list of successfully tested scripts
if len(tested_scripts) < len(scripts):
    print(f"Tested scripts before the error: {', '.join(tested_scripts)}")
