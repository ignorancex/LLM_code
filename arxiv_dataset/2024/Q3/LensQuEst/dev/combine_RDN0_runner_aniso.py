import subprocess
from datetime import datetime
from tqdm import trange


# Loop over the jobs
for i in trange(501):
    # Generate job name with index
    job_name = f"combine-RDN0-aniso-{i}-nbins51"


    nBins = 51
    savefname = '/oak/stanford/orgs/kipac/users/delon/LensQuEst/RDN0-combined-aniso-%d-nBins%d.pkl'%(i, nBins)
    from os.path import exists
    file_exists = exists(savefname)
    if file_exists:
        print('skipping', i)
        continue
    # Define output and error log file paths
    date_str = datetime.now().strftime("%Y-%m-%d")
    output_log = f"logs/{date_str}-{job_name}.out"
    error_log = f"logs/{date_str}-{job_name}.err"

    # Construct the command to run
    cmd = f"python combine_RDN0.py {i} 51 aniso > {output_log} 2> {error_log}"

    # Run the command
    subprocess.run(cmd, shell=True, check=True)
