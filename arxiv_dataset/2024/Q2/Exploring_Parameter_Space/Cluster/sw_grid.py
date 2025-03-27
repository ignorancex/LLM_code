##############################################
# created 17-11-2020
#
# Grid_Call_Stroopwafel.py
#
# Python script meant to call stroopwafel_interface for a variety of grid values. 
# 
# This file calls stroopwafel_interface.py, which calls COMPAS in a bunch of slurm scripts ##
# 
##############################################
import numpy as np
import os
import time
from subprocess import Popen, PIPE, call
import subprocess
import sys
import pickle
import math
import shutil
import fileinput
import itertools


###############################################
###		 
###############################################
class GridVar():
	""" class to store info of each COMPAS parameter that youd like to vary
	name 		= string to name folder where you would like to store your stuff
	pyName	= name for this parameter in the PythonSubmit.py 
	pySubmit_i  = line number of your param in PythonSubmit.py 
	values	= The actual values you want to run over """
	def __init__(self,name,pyName,pySubmit_i, values):
		self.name = name
		self.pyName = pyName
		self.pySubmit_i = pySubmit_i
		self.values = values
		self.storage = {self.name: [self.pyName, self.pySubmit_i, self.values]}

	def getStorage(self):
		return self.storage

###############################################
###		 
###############################################
def grid_vectToDict(list_of_grid_vect):
	"""Helper function to automate dictionary filling
	list_of_grid_vect = list of objects of class GridVar
	"""
	# creating a dictionary to fill 
	gridDictionary = {}
	for item in list_of_grid_vect:
		gridDictionary[item.name] = (item.pyName, item.pySubmit_i, item.values)
	return gridDictionary


#################################################################
## 
##	Should be Changed by user ##
##
#################################################################
#################################################################
root_out_dir		= "/n/holystore01/LABS/berger_lab/Users/aboesky/CompasOutputs"
file_name 			= 'COMPAS_Output_test.h5'
user_email 			= "aboesky@college.harvard.edu"

# What scripts do you want to run?
RunStroopwafel 			= True
RunPostProcessing 		= True
RunCosmicIntegration    = True

# Do you want to run CI locally?
Run_cluster 			= True


##		Grid and Grid directories
#######################################################################
# What metallicity paramters do you want to loop over?
mu0List				=[0.025]#[0.035]#[0.025]#
muzList				=[-0.05]#[-0.23]#[-0.05]#[-0.15, -0.23,-0.3, -0.4, -0.5]
sigma0List			=[1.125]#1.125] #[1.125]#[0.39]#[1.125]
sigmazList			=[0.05]#[0.0]#[0.05]
alphaList			=[-1.77]#[0.0]#[-1.77]

#aSF, bSF, cSF, dSF
SFRparams           = [0.02, 1.48, 4.45, 5.90] #[0.01, 2.77, 2.9, 4.7]  #[0.02, 1.48, 4.45, 5.90] 
 
# What physics variations do you want to loop over?
# Make as many grid vectors as you need, See class GridVar   
# To see available options: $COMPAS_ROOT_DIR/COMPAS/COMPAS --help
verbose = True
grid_variable1 = GridVar('LBV','luminous_blue_variable_prescription', 181,['\"NONE\"', '\"HURLEY_ADD\"','\"BELCZYNSKI\"'])
grid_variable2 = GridVar('beta','mass_transfer_fa', 192, [0.0, 0.25, 0.5, 0.75, 1.0])# [0.5]
grid_variable3 = GridVar('fWR','wolf_rayet_multiplier', 184, [0.0, 1.0])

## !!! Add each vector to the list below
gridDictionary = grid_vectToDict([grid_variable1,grid_variable2,grid_variable3])

##################################################################
# This is the slurm script youre using
#SBATCH --partition=%s 				# Partition to submit to
##################################################################
SlurmJobString="""#!/bin/bash
#SBATCH --job-name=%s 				#job name
#SBATCH --nodes=%s 					# Number of nodes
#SBATCH --ntasks=%s 				# Number of cores
#SBATCH --output=%s 				# output storage file
#SBATCH --error=%s 					# error storage file
#SBATCH --time=%s 					# Runtime in minutes
#SBATCH --mem=%s 					# Memory per cpu in MB (see also --mem-per-cpu)
#SBATCH -p %s
#SBATCH --mail-user=%s 				# Send email to user
#SBATCH --mail-type=FAIL				#
#
#Print some stuff on screen
echo $SLURM_JOB_ID
echo $SLURM_JOB_NAME
echo $SLURM_ARRAY_TASK_ID
#
#Set variables
export OUT_DIR=%s
export QT_QPA_PLATFORM=offscreen # To avoid the X Display error
#
#CD to output directory
cd $OUT_DIR/masterfolder/%s
#
# Run your job
%
"""

###############################################
###		 
###############################################
def get_Grid_Indces(gridDictionary):
	"""
	This is where the magic happens for fast grid creation
	Sinice we don't know how many grid_variables and of what len
	the user will supply, we have to be a bit smart w.r.t iterating
	our grid points. We use itertools, that returns a list of index 
	combinations based on the shape of your object

	gridDictionary =  Dictionary containing grid variables
	gridDictionary[name][0] = pyName 
	gridDictionary[name][1] = pySubmit_i 
	gridDictionary[name][2] = values
	"""
	######################################
	nGridPoints = 1
	shape = []
	for i, name in enumerate(gridDictionary.keys() ):
		if verbose:
			print('name', name)
			print('pyName', gridDictionary[name][0])
			print('pySubmit_i', gridDictionary[name][1])
			print('values', gridDictionary[name][2])
		nGridPoints *= len(gridDictionary[name][2])
		shape.append(len(gridDictionary[name][2]))

	if verbose:		
		print('You will make nGridPoints=', nGridPoints)
		print('shape', shape)

	######################
	# Create the "values" each for-loop would iter over
	loopover = [range(s) for s in shape]
	# Unpack the list using "*" operator 
	prod = itertools.product(*loopover)
	return prod


###############################################
###		 
###############################################
def replaceFileLine(file_dir, line_num, replacestr):
	"""
	file_dir   = The file of which you would like to change a line
	line_num   = The line number that you want to change
	replacestr = The string that you want to replace this line wtth
	"""
	#Open file of interest
	with open (file_dir, "r") as myfile:
		data = myfile.readlines()
	#Replace line
	data[line_num-1] = replacestr
	# Write everything back
	with open (file_dir, "w") as wfile:
		wfile.writelines(data)


###############################################
###		 
###############################################
def MakeSlurmBatch(OUT_DIR = None, sub_dir = 'MainRun/', python_name = "stroopwafel_interface", job_name = "SWrun",\
 number_of_nodes = 1, number_of_cores = 1, partition='hernquist', flags=" ",\
 walltime = '05:00:00' ,memory = '16000', email = None):

	outfile = OUT_DIR +'/masterfolder/'+sub_dir + job_name+ '.out'
	errfile = OUT_DIR +'/masterfolder/'+sub_dir + job_name+ '.err'

	job_line = "python "+python_name+".py "+flags+" > $OUT_DIR/masterfolder/"+sub_dir+job_name+".log"

	# Make slurm script string
	interface_job_string = SlurmJobString % (job_name, number_of_nodes, number_of_cores, \
		outfile, errfile, walltime, memory, partition, user_email,\
		OUT_DIR, sub_dir, job_line)

	sbatchFile = open(OUT_DIR + '/masterfolder/'+sub_dir +job_name+'.sbatch','w')
	print('writing ', OUT_DIR + '/masterfolder/'+sub_dir +job_name+'.sbatch')
	sbatchFile.write(interface_job_string)
	sbatchFile.close()

	return interface_job_string


###############################################
###		 
###############################################
def RunSlurmBatch(run_dir = None, job_name = "stroopwafel_interface", dependency = False, dependent_ID = None):

	if not dependency:
		sbatchArrayCommand = 'sbatch ' + os.path.join(run_dir+job_name+'.sbatch') 
	else:
		sbatchArrayCommand = 'sbatch --dependency=afterok:' + str(int(dependent_ID)) + ' ' + os.path.join(run_dir+job_name+'.sbatch') 

	# Open a pipe to the sbatch command.
	proc = Popen(sbatchArrayCommand, shell=True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True)

	# Send job_string to sbatch
	if (sys.version_info > (3, 0)):
		proc.stdin.write(sbatchArrayCommand.encode('utf-8'))
	else:
		proc.stdin.write(sbatchArrayCommand)

	print('sbatchArrayCommand:', sbatchArrayCommand)
	out, err = proc.communicate()
	print("out = ", out)
	job_id = out.split()[-1]
	print("job_id", job_id)
	return job_id



def CallAGridpoint(root_out_dir = None, file_name=None , Run_SW = True, Run_PP = True, Run_CI = True,\
 user_email = None, mu0_list=[0.035], muzList=[-0.23], sigma0List=[0.39] ,sigmazList=[0.], alphaList=[0.],cluster_run=True):
	###############################################
	# copy the masterfolder to the ROOT_OUT_DIR
	###############################################i
	if Run_SW:
		shutil.copytree('masterfolder', root_out_dir+'/masterfolder')  


	###############################################
	# Start making slurm scripts and running em :)
	###############################################
	###############################################
	# Make Stroopwafel batch and submit it
	if Run_SW:
		print(10* "*" + ' You are Going to Run stroopwafel_interface.py')
		# Make and safe a slurm command  #"stroopwafel_interface",\
		SW_job_string = MakeSlurmBatch(OUT_DIR = root_out_dir, sub_dir = 'MainRun/', python_name = "stroopwafel_interface",\
		 job_name = "SWrun", number_of_nodes = 1, number_of_cores = 1, partition='demink',\
		 walltime = "40:00:00", memory = "15000", email = user_email) #150000

		# Submit the job to sbatch! 
		SWjob_id = RunSlurmBatch(run_dir = root_out_dir+'/masterfolder/MainRun/', job_name = "SWrun",\
		 dependency = False, dependent_ID = None)

	###############################################
	# Make Post Processing batch and submit it
	# This job depends on the stroopwafel job to be done
	if Run_PP:
		###############################################
		# First the COMPAS Post processing
		DEPEND = True
		print(10* "*" + ' You are Going to Run PostProcessing.py')
		if not Run_SW:
			print('You are just re-doing the PP, copy the postProcessing files')
			if os.path.exists( root_out_dir+'masterfolder/postProcessing'):
				shutil.rmtree( root_out_dir+'masterfolder/postProcessing')
			shutil.copytree('masterfolder/postProcessing', root_out_dir+'/masterfolder/postProcessing')
			DEPEND, SWjob_id = False, 0
		###############################################
		# Make and safe a slurm command
		h5Flags = ' '+root_out_dir+'/output/ -r 2 -o ../../output/COMPAS_Output.h5'
		PP_job_string = MakeSlurmBatch(OUT_DIR = root_out_dir, sub_dir = 'postProcessing/', python_name = "h5copy",\
		 job_name = "COMPAS_PP", number_of_nodes = 1, number_of_cores = 1, partition='demink,conroy,hernquist',\
		 walltime = "3:00:00" ,memory = "4000", email = user_email, flags= h5Flags)

		# Submit the job to sbatch! 
		PPjob_id = RunSlurmBatch(run_dir = root_out_dir+'/masterfolder/postProcessing/', job_name = "COMPAS_PP",\
		 dependency = DEPEND, dependent_ID = SWjob_id)

		###############################################
		# Then append the SW weights
		print(10* "*" + ' You are Going to Run append_weights.py')
		# Make and safe a slurm command
		append_job_string = MakeSlurmBatch(OUT_DIR = root_out_dir, sub_dir = 'postProcessing/', python_name = "append_weights",\
		 job_name = "append_weights", number_of_nodes = 1, number_of_cores = 1, partition='demink,conroy,hernquist',\
		 walltime = "3:00:00", memory = "50000", email = user_email)

		# Submit the job to sbatch! 
		append_job_id = RunSlurmBatch(run_dir = root_out_dir+'/masterfolder/postProcessing/', job_name = "append_weights",\
		 dependency = True, dependent_ID = PPjob_id)

	###############################################
	# Make Post Processing batch and submit it
	# This job depends on the stroopwafel job to be done
	if Run_CI:
		DEPEND = True
		print(10* "*" + ' You are Going to Run FastCosmicIntegration.py')
		if not Run_SW:
			print('You are just re-doing the CI, copy the CI files')
			if os.path.exists( root_out_dir+'masterfolder/CosmicIntegration'):
				shutil.rmtree( root_out_dir+'masterfolder/CosmicIntegration')
			shutil.copytree('masterfolder/CosmicIntegration', root_out_dir+'/masterfolder/CosmicIntegration')  
			DEPEND, append_job_id = False, 0
		# Run over your metallicity density parameters of interest
		# This is pretty ugly... but it shouldnt matter too much?
		n_CI = 1


		for mu0 in mu0_list:
			for muz in muzList:
				for sig0 in sigma0List:
					for sigz in sigmazList:
						for al in alphaList:
							Flags = " --path "+root_out_dir+"/output/"+" --filename "+file_name+\
							" --mu0 " +str(mu0)+" --muz "+str(muz)+" --sigma0 "+str(sig0)+" --sigmaz "+str(sigz)+" --alpha "+str(al)+\
							" --aSF " +str(SFRparams[0])+" --bSF "+str(SFRparams[1])+" --cSF "+str(SFRparams[2])+" --dSF "+str(SFRparams[3])+\
							" --weight "+"mixture_weight"+ " --zstep "+"0.01"+" --sens "+"O3"+ " --m1min "+"10."+ " --dco_type BBH"+\
							" --BinAppend "+ " --redshiftBinSize "+"0.05"

							if cluster_run:
								# Make and safe a slurm command
								CI_job_string = MakeSlurmBatch(OUT_DIR = root_out_dir, sub_dir = 'CosmicIntegration/', python_name = "FastCosmicIntegration",\
								 job_name = "COMPAS_CI"+str(n_CI), number_of_nodes = 1, number_of_cores = 1, partition='demink,conroy,hernquist,shared',\
								 walltime = "5:00:00", memory = "150000", email = user_email, flags= Flags)

								# Submit the job to sbatch! 
								CIjob_id = RunSlurmBatch(run_dir = root_out_dir+'/masterfolder/CosmicIntegration/', job_name = "COMPAS_CI"+str(n_CI),\
								 dependency = DEPEND, dependent_ID = append_job_id)
								n_CI += i
								DEPEND, append_job_id = True, CIjob_id
							else:
								print('You will run the CI locally')
								# Change the current working directory
								os.chdir(root_out_dir+'masterfolder/CosmicIntegration')
								# execute this job locally (slow)
								job_line = "python FastCosmicIntegration.py "+Flags+" > "+"COMPAS_CI"+str(n_CI)+".log"
								print('job_line', job_line)

								with open("./COMPAS_CI"+str(n_CI)+".err", "w+") as f:
									subprocess.call(job_line, shell=True, stdin=PIPE, stdout=f, stderr=f)


	print(10* "*" + " You are all done with this job! " + 10* "*")



###############################################
###		 Run your Code!	###
###############################################
if __name__ == '__main__':

	if not Run_cluster:
		print('start from main dir')
		run_dir = os.getcwd()


	###############################################
	# Make the output directory if it doesn't exist
	if not os.path.exists(root_out_dir):
		print('making ', root_out_dir)
		os.mkdir(root_out_dir)
		# copy this python script to the ROOT out dir
		shutil.copyfile('Grid_Call_Stroopwafel.py', root_out_dir+'Grid_Call_Stroopwafel.py')
	else:
		ValueError("The output folder already exists. Either remove it, or choose a new output folder name")


	############################################
	# Make a list of grid variable combinations
	# that you want to loop over
	prod = get_Grid_Indces(gridDictionary)

	############################################
	# Loop over each grid point 
	# i.e. loop over each unique combination of indices in prod
	for idx in prod:
		GridPointDirName = ''
		#print(idx) # index combo to call your grid_vectors
		# Loop over the COMPAS variables you're changing
		for i, name in enumerate(gridDictionary.keys() ):
			index = idx[i]
			print(name, gridDictionary[name][2][index])

			############################################
			# Append the name and value to gridPointDirName
			GridPointDirName = GridPointDirName+name+str(gridDictionary[name][2][index]).replace('"', '') 

			if RunStroopwafel:
				############################################
				# Replace the corresponding line in PythonSubmit.py
				new_line = "    "+str(gridDictionary[name][0])+" = "+str(gridDictionary[name][2][index])+"\n"
				print(new_line)
				replaceFileLine("./masterfolder/MainRun/compasConfigDefault.yaml", gridDictionary[name][1],new_line)

		############################################
		# You are now ready to call your grid point
		print('GridPointDirName ', GridPointDirName)

		############################################
		CallAGridpoint(root_out_dir = root_out_dir+GridPointDirName+'/', file_name=file_name , user_email = user_email,
			Run_SW = RunStroopwafel, Run_PP = RunPostProcessing, Run_CI = RunCosmicIntegration,
			 mu0_list=mu0List, muzList=muzList, sigma0List=sigma0List, sigmazList=sigmazList, alphaList=alphaList, cluster_run=Run_cluster)
		
		if not Run_cluster:
			# Change the current working directory to the run dir
			os.chdir(run_dir) 




