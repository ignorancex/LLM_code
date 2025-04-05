import sys
import os
import glob
import shutil

def create_dir(path):
    try:
        os.mkdir(path)
    except:
        print "Folder "+path+" already created"


def copy_extension(source_dir, dest_dir, extension):
    files = glob.glob(source_dir+'/'+extension)
    for file in files:
        if os.path.isfile(file):
            shutil.copy2(file, dest_dir)



folder = sys.argv[1]
nights = os.listdir(folder+'11_REDUCED/')

# Make DIGEST directory
create_dir(folder+'/22_REDUCED_DIGEST')
digest = folder+'/22_REDUCED_DIGEST/'

for night in nights:
    if not os.path.isdir(digest+'/'+night):
		create_dir(digest+'/'+night)
		nfolder = digest+'/'+night+'/'
		create_dir(digest+'/'+night+'/reduced')
		create_dir(digest+'/'+night+'/auxiliar')
		copy_extension(folder+'11_REDUCED/'+night+'/reduced/',digest+'/'+night+'/reduced/','*.fits')
		copy_extension(folder+'11_REDUCED/'+night+'/auxiliar/',digest+'/'+night+'/auxiliar/','*.pdf')

