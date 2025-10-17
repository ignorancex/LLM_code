import os
import json

filtered_file = "target/target_2025Q3_c_filtered.jsonl"
links_file = "dataset_collection/github/links/new_github_links.json"
output_file = "dataset_collection/github/links/cpp_dataset_links.json"

quarter = "2025Q3"

def main():

    with open(links_file, "r", encoding="utf-8") as f:
        all_links_data = json.load(f)

    link_to_cat = {}
    for item in all_links_data:
        link = item.get("github_links")
        cat = item.get("categories", "")
        if link:
            link_to_cat[link.rstrip("/")] = cat


    filtered_links = []
    with open(filtered_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            link = data.get("link", "").rstrip("/")
            if not link:
                continue
            cat = link_to_cat.get(link, "")
            filtered_links.append({"link": link, "categories": cat})


    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            dataset_links = json.load(f)
    else:
        dataset_links = {}

    if quarter not in dataset_links:
        dataset_links[quarter] = []

    dataset_links[quarter].extend(filtered_links)


    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dataset_links, f, ensure_ascii=False, indent=2)

    print(f"✅ Done merging {len(filtered_links)} to {output_file}[{quarter}]")

if __name__ == "__main__":
    main()
