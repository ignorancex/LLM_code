import os
import json

def delete_folders_with_only_time_info(root_dir, link_file_path, new_link_file_path):
    deleted_repos = []

    # 删除符合条件的文件夹并记录仓库名
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if len(filenames) == 1 and filenames[0] == 'time_info.txt' and not dirnames:
            repo_name = os.path.basename(dirpath)
            # print(f"Deleting folder: {dirpath}")
            try:
                os.remove(os.path.join(dirpath, 'time_info.txt'))  # 删除文件
                os.rmdir(dirpath)  # 删除空文件夹
                deleted_repos.append(repo_name)
            except Exception as e:
                print(f"Error deleting {dirpath}: {e}")

    # 如果有删除的仓库名，读取原 JSON 并保存一个新的副本
    if deleted_repos:
        try:
            with open(link_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            original_count = len(data.get("github_links", []))
            data["github_links"] = [
                link for link in data.get("github_links", [])
                if not any(repo in link for repo in deleted_repos)
            ]
            new_count = len(data["github_links"])

            with open(new_link_file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)

            print(f"Removed {original_count - new_count} links.")
            print(f"New JSON saved to: {new_link_file_path}")
        except Exception as e:
            print(f"Error creating new JSON file: {e}")
    else:
        print("No folders were deleted. JSON file remains unchanged.")

# 设置路径
year = '2024'
root_directory = f'LLM_code/dataset/github_code/{year}'
link_json_path = f'LLM_code/dataset/github_links/links_empty_included/link_{year}.json'
new_link_json_path = f'LLM_code/dataset/github_links/links_non_empty/link_{year}_filtered.json'

delete_folders_with_only_time_info(root_directory, link_json_path, new_link_json_path)
