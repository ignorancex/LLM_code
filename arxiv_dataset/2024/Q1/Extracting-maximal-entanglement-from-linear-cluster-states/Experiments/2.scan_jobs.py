# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 22:43:40 2021

@author: Jarnd
"""

## Global imports
import os
import _pickle as pickle



## Local imports
from tools import retrieve_nonlocal_backend


#%% Scrape the jobs/unprocessed folder
basefolder = './Jobs/Unprocessed'

experiments = [subdir for subdir in os.walk(basefolder)]

base_folder_contents = experiments.pop(0)
first_subfolders = base_folder_contents[1]

unfinished_jobs = 0
#%% Loop through all experiments
for index, experiment in enumerate(experiments):
    folder_name = first_subfolders[index]
    
    print(f'Scraping subfolder "{folder_name}"')
    ## Check if we're in the correct subfolder by checking if the path is the same as the folder name
    assert folder_name == experiment[0][-len(folder_name):], f"Warning, {folder_name} is not the name of the current folder {experiment[0]}"
    
    # Loop through every job in the folder
    for job_textfile in experiment[2]:
        
        # Create the full (relative) filepath
        filepath = experiment[0] + '/' + job_textfile
        
        # Open the file
        with open(filepath, 'r') as filehandle:
            job_id, backend_name = filehandle.readline().split(' ')
            
        
        print(f'  Job {job_id} at backend {backend_name}')
        
        # Retrieve nonlocal backend and retrieve job from it
        backend = retrieve_nonlocal_backend(backend_name)
        job = backend.retrieve_job(job_id)
        
        # Check if the job is finished
        is_finished = job.in_final_state()
        print(f'    Job {job_id} is finished: {is_finished}')
        
        # If the job is finished, process it
        if is_finished:
            # Check if the job is succesfully finished
            if job.done():
                # Job has succesfully run, retrieve the result
                print(f'        Job {job_id} has succesfully run. Retrieving results and moving to Done folder')
                result = job.result()
                result_dictionary = result.to_dict()
                
                ## Write the result
                destination_directory = f'./Jobs/Done/{folder_name}/'
                # Check for destination folder
                os.makedirs(destination_directory, exist_ok = True)
                
                ## Serialize the dictionary and save it to file
                with open(destination_directory + f'{job_textfile}', 'wb') as filehandle:
                    pickle.dump(result.to_dict(), filehandle)
                
                ## Remove the original jobid file
                os.remove(filepath)
            
            elif not job.done():
                # Job is finished but has not succesfully run.
                print(f'        Job {job_id} has completed but failed. Moving job_id file to failed Folder.')
                # Move the jobid text file to the 'failed' folder
                destination_directory = f'./Jobs/Failed/{folder_name}/'
                # Check for destination folder
                os.makedirs(destination_directory, exist_ok = True)
                
                os.replace(filepath, destination_directory + f'{job_textfile}')
        
        elif not is_finished:
            # Job is not finished. Skipping
            unfinished_jobs += 1


#%%
print(f'Done scraping the unprocessed folder. There are {unfinished_jobs} unfinished jobs remaining.')               