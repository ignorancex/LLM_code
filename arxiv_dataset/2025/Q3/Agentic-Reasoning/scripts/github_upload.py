import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, cwd=None, check=True):
    """Execute a shell command and return the output"""
    try:
        result = subprocess.run(command, shell=True, check=check, text=True, 
                              capture_output=True, cwd=cwd)
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {command}")
        print(f"Error message: {e.stderr}")
        return None

def setup_github_repo(repo_name, repo_url, project_path, branch_name="main", is_new_repo=True):
    """Setup and push to GitHub repository"""
    
    print(f"\nSetting up repository: {repo_name}")
    
    # Change to project directory
    os.chdir(project_path)
    
    # Initialize git if needed
    if not Path(project_path / ".git").exists():
        run_command("git init")
        print("Initialized git repository")

    # Set main branch
    run_command("git branch -M main")
    print("Set main as default branch")

    # Add remote if not already added
    remotes = run_command("git remote -v")
    if not remotes or "origin" not in remotes:
        run_command(f"git remote add origin {repo_url}")
        print("Added remote origin")

    # Create .gitignore if it doesn't exist
    gitignore_path = Path(project_path) / ".gitignore"
    if not gitignore_path.exists():
        with open(gitignore_path, "w") as f:
            f.write("""
# Python
__pycache__/
*.py[cod]
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Project specific
cache/
local_mem/
research_output/
test_storm_output/
scripts/tools/temp.py

# Secrets
.env
""")
        print("Created .gitignore")

    # Try to fetch from remote
    fetch_result = run_command("git fetch origin main", check=False)
    
    # Add all files
    run_command("git add .")
    print("Added all files to git")

    # Commit changes
    run_command('git commit -m "Update repository"', check=False)
    
    # Try to push
    push_result = run_command(f"git push -u origin main", check=False)
    
    if push_result is None:
        print("Push failed, trying force push...")
        # Attempt force push
        force_push = run_command("git push -f origin main")
        if force_push is not None:
            print("Successfully force pushed to remote")
        else:
            print("Force push failed. Please check repository permissions")
    else:
        print("Successfully pushed to GitHub")

    print(f"\nRepository URL: {repo_url}")

def main():
    parser = argparse.ArgumentParser(description="Upload project to GitHub repository")
    parser.add_argument("--repo-name", required=True, help="Name of the repository")
    parser.add_argument("--repo-url", required=True, help="GitHub repository URL")
    parser.add_argument("--project-path", default=".", help="Path to the project directory")
    
    args = parser.parse_args()
    
    # Convert project path to absolute path
    project_path = Path(args.project_path).resolve()
    
    # Verify project path exists
    if not project_path.exists():
        print(f"Error: Project path {project_path} does not exist!")
        return

    setup_github_repo(args.repo_name, args.repo_url, project_path)

if __name__ == "__main__":
    main() 