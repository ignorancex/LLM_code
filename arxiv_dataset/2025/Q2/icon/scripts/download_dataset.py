import os
import click
import subprocess
from pathlib import Path

TASK = ["close_microwave", "close_drawer", "take_lid_off_saucepan", "open_box", "put_rubbish_in_bin", "lift_cube",\
         "open_door", "stack_cube", "lift_cube_kinova", "stack_cube_kinova", "lift_cube_iiwa", "stack_cube_iiwa"]

@click.command(help="Download datasets from the web.")
@click.option("-d", "--directory", type=str, default="data", help="Target directory.")
@click.option("-t", "--task", type=str, default="", help="Task name.")
def main(directory, task):
    directory = Path(os.path.expanduser(directory)).absolute()
    directory.mkdir(parents=True, exist_ok=True)
    # Download datasets from the web
    tasks = list()
    if any(task):
        assert task in TASK, f"No datasets provided for \"{task}\"!"
        tasks.append(task)
    else:
        tasks.extend(TASK)

    for t in tasks:
        subprocess.run([
            "wget",
            "-P",
            str(directory),
            f"https://huggingface.co/datasets/HenryWJL/icon/resolve/main/data/{t}.zip",
            "--no-check-certificate"
        ])
    # Unzip and remove zip files
    zip_files = list(directory.glob("*.zip"))
    for zip_file in zip_files:
        subprocess.run([
            "unzip",
            "-q",
            str(zip_file),
            "-d",
            str(directory)
        ])
        subprocess.run([
            "rm",
            "-r",
            str(zip_file)
        ])


if __name__ == "__main__":
    main()