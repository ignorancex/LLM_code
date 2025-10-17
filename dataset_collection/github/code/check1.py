import os
import json

target_file = "target/target_2025Q3_c.jsonl"
dataset_dir = "arxiv_dataset_cpp/2025/Q3"
output_file = "target/target_2025Q3_c_filtered.jsonl"

def get_repo_name(link: str) -> str:
    return link.rstrip("/").split("/")[-1]

def main():
    valid_links = []
    
    with open(target_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            link = data.get("link", "")
            repo_name = get_repo_name(link)
            repo_path = os.path.join(dataset_dir, repo_name)
            
            if os.path.isdir(repo_path):
                valid_links.append(data)
            else:
                print(f"❌ Delete: {link} (Cannot find {repo_path})")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in valid_links:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n✅ Done filtering: {len(valid_links)} left, saved to {output_file}")

if __name__ == "__main__":
    main()
