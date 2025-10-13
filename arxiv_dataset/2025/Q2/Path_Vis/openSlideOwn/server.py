import os
def get_directory_structure(rootdir):
    """
    Creates a nested dictionary that represents the folder structure of rootdir.
    """
    dir_structure = {}
    for dirpath, dirnames, filenames in os.walk(rootdir):
        dirpath = dirpath.replace(rootdir, '', 1)
        subdir_dict = dir_structure
        if dirpath != '':
            # For nested directories, get or create the sub-dictionary
            for sub_dir in dirpath.strip(os.sep).split(os.sep):
                subdir_dict = subdir_dict.setdefault(sub_dir, {})
        for filename in filenames:
            subdir_dict[filename] = None
    return dir_structure

# Example usage:
path = '/Users/jaiprakashveerla/Documents/Jai/Data/DATABASE'
directory_structure = get_directory_structure(path)
print(directory_structure)
