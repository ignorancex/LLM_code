"""
Convert the tarball of the entire MESA output to binary.
This allows showyourwork to cache the binary results on zenodo sandbox
for faster reproduction
"""
__author__ = "Mathieu Renzo"
from MESAreader import getSrcCol
import os
import paths
import tarfile


def clean_data_from_tarball(member):
    print("in clean_data_from_tarball")
    bin_fname = paths.data / str(member.name[:-4] + ".npy")
    if not os.path.isfile(bin_fname):
        # file does not exist, extract
        tarball.extractall(path=paths.data, members=[member])
        src, col = getSrcCol(paths.data / member.name, True, True, bin_fname=bin_fname)


if __name__ == "__main__":
    tarball = tarfile.open(paths.data / "MESA_output.tar.gz", "r:gz")
    # skip folders and only consider *.data files
    data_files = [member for member in tarball.getmembers() if ".data" in str(member)]
    for member in data_files:
        clean_data_from_tarball(member)
    tarball.close()
