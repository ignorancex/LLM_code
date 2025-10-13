from .exp import (
    start,
    RUNPOD_RUN_CFGS,
    RUNPOD_DATA_PATHS,
    RUNPOD_RUN_NAMES,
    PROJECT_ROOT,
    EXPERIMENT_SCRATCH,
)
from . import exp
import os
from dotenv import dotenv_values
import subprocess

exp.EXECUTOR = "RUNPOD"


def start_runpod():
    exp_num = "_".join(RUNPOD_RUN_NAMES[0].split("_")[:2])  # name is exp_n_...
    runpod_startup = (PROJECT_ROOT / "runpod_startup.sh").read_text()
    runpod_launch = (PROJECT_ROOT / "runpod_launch.sh").read_text()
    runpod_startup_path = EXPERIMENT_SCRATCH / f"runpod_startup_{exp_num}.sh"
    runpod_launch_path = EXPERIMENT_SCRATCH / f"runpod_launch_{exp_num}.sh"
    cmds: list[str] = []
    for i, (run_cfg, data_path, run_name) in enumerate(
        zip(RUNPOD_RUN_CFGS, RUNPOD_DATA_PATHS, RUNPOD_RUN_NAMES)
    ):
        current_exp_num = "_".join(run_name.split("_")[:2])
        args = f"--cfg {run_cfg} --data_path {data_path} --name {run_name}"
        cmd = f"export WANDB_GROUP={current_exp_num}\nuv run train {args}"
        cmds.append(cmd)
    all_commands = "\n".join(cmds)
    env_values = dotenv_values(PROJECT_ROOT / ".env")
    env_values["TORCHINDUCTOR_CACHE_DIR"] = "/workspace/.torchinductor"
    for k, v in env_values.items():
        all_commands = f"export {k}={v}\n{all_commands}"

    runpod_startup = runpod_startup.replace("{{CMD}}", all_commands)

    runpod_launch = runpod_launch.replace(
        "{{ RUNPOD_STARTUP_SH }}", str(runpod_startup_path)
    )
    runpod_launch = runpod_launch.replace("{{ POD_NAME }}", exp_num)
    runpod_api_key = os.environ["RUNPOD_API_KEY"]
    runpod_launch = runpod_launch.replace("{{ RUNPOD_API_KEY }}", runpod_api_key)
    runpod_startup_path.write_text(runpod_startup)
    runpod_launch_path.write_text(runpod_launch)
    os.chdir(EXPERIMENT_SCRATCH)
    subprocess.run(". " + str(runpod_launch_path.absolute()), shell=True)


def main():
    start()
    start_runpod()


if __name__ == "__main__":
    main()
