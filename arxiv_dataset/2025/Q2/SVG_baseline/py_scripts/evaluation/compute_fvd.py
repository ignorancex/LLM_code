import os
import sys
import subprocess
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "third_party", "stylegan-v"))


def call_script():
    args = sys.argv[1:]
    dirname = os.path.join(os.path.dirname(__file__), "../../third_party/stylegan-v")
    cmd = ["python", os.path.join(dirname, "src/scripts/calc_metrics_for_dataset.py")] + args
    subprocess.run(cmd, cwd=dirname)

if __name__ == "__main__":
    call_script()