import subprocess
from tqdm import trange
for i in range(20):
    print(i)
    for j in trange(20):
        if(i < 10 and j < 10):
            continue
        line = 'python -u gen_MCMC_N1_part2_combineTs.py %d %d &> logs/2023-08-23-MCN1-part2-%d-%d'%(i, j, i, j)
        result = subprocess.run(line, shell=True, check=True)
