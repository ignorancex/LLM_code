import os
import json

# 输入输出路径
target_file = "target/target_2025Q3_c.jsonl"
dataset_dir = "arxiv_dataset_cpp/2025/Q3"
output_file = "target/target_2025Q3_c_filtered.jsonl"

def get_repo_name(link: str) -> str:
    """
    从 GitHub 链接中提取仓库名
    例如: https://github.com/12wang3/mllp -> mllp
    """
    return link.rstrip("/").split("/")[-1]

def main():
    valid_links = []
    
    # 逐行读取 jsonl
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            link = data.get("link", "")
            repo_name = get_repo_name(link)
            repo_path = os.path.join(dataset_dir, repo_name)
            
            if os.path.isdir(repo_path):
                valid_links.append(data)
            else:
                print(f"❌ 删除: {link} (没有找到 {repo_path})")

    # 保存过滤后的结果
    with open(output_file, "w", encoding="utf-8") as f:
        for item in valid_links:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ 完成过滤: {len(valid_links)} 条保留, 已保存到 {output_file}")

if __name__ == "__main__":
    main()
