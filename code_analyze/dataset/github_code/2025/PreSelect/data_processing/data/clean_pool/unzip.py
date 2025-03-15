# Importing required module
import random
import os
import subprocess
import sys
count = 0
for i in range(1,11):
    for j in range(0,10):
        random.seed(i*10 + j)
        L1 = random.sample(range(1, 3000), 2999)

        for k in range(0,40):
            index = str(L1[k]).zfill(8)
            global_index = str(i).zfill(2)
            decompressed_file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl"
            compressed_file_name = f"global_{global_index}_local_{j}_shard_{index}_processed.jsonl"
            if not os.path.exists("/home/pool-unzip/" + decompressed_file_name):
                try:
                    subprocess.Popen(f"cp /home/pool/{compressed_file_name} /home/pool-unzip/{decompressed_file_name}",shell=True)
                    print(f"Number {k}: {compressed_file_name}  unzip success")
                except:
                    print(f"{compressed_file_name} unzip failed")