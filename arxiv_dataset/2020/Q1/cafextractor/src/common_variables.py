import sys
import os
import shutil

def cv(argum):

	# Working directory
	root_path 	= os.getcwd()

	# Raw data
	rawdir		= sys.argv[1]
	rawdir_path = root_path+'/00_RAW/'

	# Reduced data
	redid		= sys.argv[2]
	reddir		= sys.argv[1]
	reddir_path	= root_path+'/11_REDUCED/'+red_id
