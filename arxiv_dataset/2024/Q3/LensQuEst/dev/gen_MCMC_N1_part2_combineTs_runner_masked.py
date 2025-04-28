import subprocess
from tqdm import trange
for i in trange(20):
    for j in trange(20):
        line = 'python -u gen_MCMC_N1_part2_combineTs.py %d %d masked &> logs/2023-09-12-MCN1-part2-%d-%d-masked'%(i, j, i, j)
        result = subprocess.run(line, shell=True, check=True)
