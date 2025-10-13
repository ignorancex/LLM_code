import subprocess


def run(command, cwd=None, capture_output=False):
    """Run a shell command in the specified directory."""
    return subprocess.run(
        command,
        check=True,
        shell=True,
        text=True,
        capture_output=capture_output,
        cwd=cwd,
    )
