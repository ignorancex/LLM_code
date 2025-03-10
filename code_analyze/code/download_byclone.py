import os
import subprocess
import shutil
import fnmatch
import stat
import time
import json
from datetime import datetime
import pytz


def remove_readonly(func, path, exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


def get_file_last_modified_time(file_path, repo_dir):
    relative_file_path = os.path.relpath(file_path, repo_dir)
    result = subprocess.run(
        ["git", "log", "-1", "--format=%cd", relative_file_path],
        cwd=repo_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    if result.returncode == 0:
        return result.stdout.strip()
    return "Unknown"


def convert_to_utc(local_time_str):
    try:
        local_time = datetime.strptime(local_time_str, "%a %b %d %H:%M:%S %Y %z")
        utc_time = local_time.astimezone(pytz.UTC)
        return utc_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    except ValueError as e:
        print(f"Error converting time: {e}")
        return "Unknown"


def convert_to_ssh_url(https_url):
    if https_url.startswith("https://github.com/"):
        return "git@github.com:" + https_url[len("https://github.com/"):] + ".git"
    return https_url


def clone_and_extract_py_files(github_url, destination="github_code/2019"):
    github_url = convert_to_ssh_url(github_url)
    print(github_url)
    repo_name = github_url.split("/")[-1].replace(".git", "")
    final_destination = os.path.join(destination, repo_name)
    time_info_file = os.path.join(final_destination, "time_info.txt")

    if os.path.exists(time_info_file):
        print(f"Skipping {repo_name}, time_info.txt already exists.")
        return

    temp_dir = "temp_repo"
    subprocess.run(["git", "clone", github_url, temp_dir], check=True)
    os.makedirs(final_destination, exist_ok=True)

    with open(time_info_file, "w") as time_info:
        for root, _, files in os.walk(temp_dir):
            for filename in fnmatch.filter(files, "*.py"):
                source_path = os.path.join(root, filename)
                relative_path = os.path.relpath(source_path, temp_dir)
                destination_path = os.path.join(final_destination, relative_path)
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
                shutil.copy2(source_path, destination_path)
                last_modified_local = get_file_last_modified_time(source_path, temp_dir)
                last_modified_utc = convert_to_utc(last_modified_local)
                time_info.write(f"{relative_path}: {last_modified_utc}\n")

    time.sleep(3)
    subprocess.run(["git", "gc", "--prune=now"], cwd=temp_dir)
    subprocess.run(["git", "rm", "-r", "--cached", "."], cwd=temp_dir)
    shutil.rmtree(temp_dir, onerror=remove_readonly)
    print(f"Python files saved in: {final_destination}")
    print(f"Modification times saved in: {time_info_file}")


def process_github_links(json_file):
    with open(json_file, "r") as f:
        data = json.load(f)
    github_links = data.get("github_links", [])
    for link in github_links:
        clone_and_extract_py_files(link)


# 假设 JSON 文件名为 github_links.json
json_filename = "link_2019.json"
process_github_links(json_filename)