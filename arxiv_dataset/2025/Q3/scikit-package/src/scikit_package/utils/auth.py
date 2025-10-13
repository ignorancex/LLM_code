import subprocess


def get_github_username():
    """Get the GitHub username using the GitHub CLI."""
    try:
        username = subprocess.check_output(
            ["gh", "api", "user", "--jq", ".login"], text=True
        ).strip()
        return username
    except subprocess.CalledProcessError:
        raise RuntimeError(
            "Could not retrieve GitHub username using GitHub CLI. "
            "Please make sure your local machine is authenticated with GitHub."
        )


def get_current_branch():
    result = subprocess.check_output(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"], text=True
    )
    return result.strip()
