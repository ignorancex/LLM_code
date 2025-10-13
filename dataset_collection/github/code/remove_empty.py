import os

def remove_empty_dirs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Delete empty folder: {dirpath}")
            except OSError as e:
                print(f"Delete failed: {dirpath}, Error: {e}")

if __name__ == "__main__":
    target_path = "arxiv_dataset/2025/Q2"  
    remove_empty_dirs(target_path)
