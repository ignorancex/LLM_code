import os
import subprocess
import shutil
import fnmatch
import stat
import time

def remove_readonly(func, path, exc_info):
    """解除只读权限并删除"""
    os.chmod(path, stat.S_IWRITE)
    func(path)

def clone_and_extract_py_files(github_url, destination="filtered_repo"):
    repo_name = github_url.split("/")[-1].replace(".git", "")
    temp_dir = "temp_repo"

    # Clone repository
    subprocess.run(["git", "clone", "--depth", "1", github_url, temp_dir], check=True)

    # Create destination folder
    os.makedirs(destination, exist_ok=True)

    # Copy only .py files
    for root, _, files in os.walk(temp_dir):
        for filename in fnmatch.filter(files, "*.py"):
            source_path = os.path.join(root, filename)
            relative_path = os.path.relpath(source_path, temp_dir)
            destination_path = os.path.join(destination, relative_path)

            os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            shutil.copy2(source_path, destination_path)

    # Allow time for Git to release files
    time.sleep(3)

    # Cleanup cloned repo
    subprocess.run(["git", "gc", "--prune=now"], cwd=temp_dir)
    subprocess.run(["git", "rm", "-r", "--cached", "."], cwd=temp_dir)

    shutil.rmtree(temp_dir, onerror=remove_readonly)

    print(f"Python files saved in: {destination}")

# Example usage
github_repo_url = "https://github.com/mattbierbaum/arxiv-public-datasets/"
#github_repo_url = "https://github.com/HSM316/LLM_Wikipedia"
clone_and_extract_py_files(github_repo_url)


# import os
# import subprocess
# import shutil
# import fnmatch
# import stat
# import time
#
# def remove_readonly(func, path, exc_info):
#     """解除只读权限并删除"""
#     os.chmod(path, stat.S_IWRITE)
#     func(path)
#
# def clone_and_extract_py_files(github_url, destination="filtered_repo"):
#     repo_name = github_url.split("/")[-1].replace(".git", "")
#     temp_dir = "temp_repo"
#
#     # 尝试克隆仓库
#     try:
#         subprocess.run(["git", "clone", "--depth", "1", github_url, temp_dir], check=True)
#     except subprocess.CalledProcessError as e:
#         print(f"❌ Git 克隆失败: {e}")
#         return
#
#     print("✅ 克隆成功，开始提取 Python 文件...")
#
#     # 创建目标文件夹
#     os.makedirs(destination, exist_ok=True)
#
#     # 复制 .py 文件
#     for root, _, files in os.walk(temp_dir):
#         for filename in fnmatch.filter(files, "*.py"):
#             source_path = os.path.join(root, filename)
#             relative_path = os.path.relpath(source_path, temp_dir)
#             destination_path = os.path.join(destination, relative_path)
#
#             os.makedirs(os.path.dirname(destination_path), exist_ok=True)
#             shutil.copy2(source_path, destination_path)
#
#     print(f"✅ Python 文件提取完成，已保存到: {destination}")
#
#     # 尝试删除 `.git` 目录
#     git_dir = os.path.join(temp_dir, ".git")
#     if os.path.exists(git_dir):
#         try:
#             shutil.rmtree(git_dir, onerror=remove_readonly)
#             print("✅ .git 目录删除成功")
#         except Exception as e:
#             print(f"⚠️ 删除 .git 目录失败: {e}")
#
#     # 尝试删除 temp_repo 目录
#     max_retries = 5
#     for i in range(max_retries):
#         try:
#             shutil.rmtree(temp_dir, onerror=remove_readonly)
#             print("✅ 临时文件删除成功")
#             break
#         except PermissionError:
#             print(f"⚠️ 删除失败，等待3秒后重试 ({i+1}/{max_retries})")
#             time.sleep(3)
#
# # 示例
# github_repo_url = "https://github.com/mattbierbaum/arxiv-public-datasets/"
# clone_and_extract_py_files(github_repo_url)
