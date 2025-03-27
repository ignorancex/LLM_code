import os

def print_directory_tree(root_dir, prefix=""):
    items = sorted([item for item in os.listdir(root_dir) if item != '.git'])
    pointers = ['├── '] * (len(items) - 1) + ['└── ']

    for pointer, item in zip(pointers, items):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path):
            print(prefix + pointer + item + "/")
            print_directory_tree(path, prefix + '│   ')
        else:
            print(prefix + pointer + item)

# Specify the directory you want to visualize
directory = "./CHIC"
print_directory_tree(directory)

