import os, shutil
import sys

os.chdir(sys.argv[1])

for f in os.listdir("."):

    if os.path.isdir(f):
        continue
    folderName = f[-10:-4]

    if not os.path.exists(folderName):
        os.mkdir(folderName)
        shutil.copy(os.path.join('.', f), folderName)
    else:
        shutil.copy(os.path.join('.', f), folderName)
