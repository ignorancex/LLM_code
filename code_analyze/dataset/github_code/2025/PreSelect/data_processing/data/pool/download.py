import os
from huggingface_hub import snapshot_download

os.environ["HF_ENDPOINT"]="https://hf-mirror.com"
pools = ["mlfoundations/dclm-pool-400m-1x"]

pool_dir = "/workspace/pool"
def update_pattern(number_downloaded, pool_name):
    print(f"restarted from {number_downloaded}")
    ignore_patterns = os.listdir(f"{pool_dir}/{pool_name}")
    
    return ignore_patterns

for pool in pools:
    pool_name = pool.split("/")[1]
    while True:
        try:
            number_downloaded = int(os.popen(f"ls {pool_dir}/{pool_name} -l | grep '^-' | wc -l").readlines()[0].strip('\n'))
            ignore_patterns = update_pattern(number_downloaded, pool_name)
            snapshot_download(repo_id=pool,
                              repo_type="dataset",
                              local_dir=f"{pool_dir}/{pool}",
                              resume_download=True,
                              max_workers=4,
                              ignore_patterns=ignore_patterns,
                              local_dir_use_symlinks=False)
            break
        except KeyboardInterrupt:
            print("Program interrupted by the user.")
            break
        except Exception as e:
            print(f"An error occurred: {e}, will restart soon")

        print("Download successful")
