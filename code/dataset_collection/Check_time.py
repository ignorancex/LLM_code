import os 
import shutil 
years =[str (y )for y in range (2020 ,2026 )]
seasons =['Q1','Q2','Q3','Q4']
all_missing_or_empty =[]
for year in years :
    for season in seasons :
        repo_base_path =f'LLM_code/arxiv_dataset_cpp/{year}/{season}'
        if not os .path .exists (repo_base_path ):
            continue 
        subfolders =[f for f in os .listdir (repo_base_path )if os .path .isdir (os .path .join (repo_base_path ,f ))]
        for folder in subfolders :
            time_info_path =os .path .join (repo_base_path ,folder ,'time_info_cpp.txt')
            if not os .path .exists (time_info_path )or os .path .getsize (time_info_path )==0 :
                all_missing_or_empty .append (os .path .join (repo_base_path ,folder ))
if all_missing_or_empty :
    for folder in all_missing_or_empty :
    choice =input ('\nDo you want to delete these folders? (y/n): ').strip ().lower ()
    if choice =='y':
        for folder in all_missing_or_empty :
            shutil .rmtree (folder )