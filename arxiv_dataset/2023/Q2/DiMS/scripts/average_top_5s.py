import sys
import os
import subprocess

if len(sys.argv)>2: s=sys.argv[2]
else s=7
print(f'Using {s} as step')
best_checkpoints = subprocess.Popen(f"ls {sys.argv[1]}/checkpoint.best_bleu*_{s}.pt", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()[0].decode(encoding='utf-8').split('\n')[:-1]
best_checkpoints=sorted(best_checkpoints)

for i in range(1, 5):
    batch=best_checkpoints[-i*5:]
    print(batch)
    target=f'{sys.argv[1]}/average_{i*5}.pt'
    print(target)
    os.system(f'python /mnt/scripts/average_checkpoints.py --input {" ".join(batch)} --output {target}')

