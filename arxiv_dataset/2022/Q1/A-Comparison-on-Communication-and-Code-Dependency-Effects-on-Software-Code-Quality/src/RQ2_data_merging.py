#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 11:50:56 2018

@author: suvodeepmajumder
"""

from interaction import social_interaction
from interaction import code_interaction
import main.git_log.buggy_commit as buggy_commit
import pandas as pd
import numpy as np
import csv
from os.path import dirname as up
import os
from pathlib import Path
import platform

if platform.system() == 'Darwin' or platform.system() == 'Linux':
    source_projects = os.getcwd() + '/project_list.csv'
else:
    source_projects = os.getcwd() + '\\project_list.csv'
project_list = pd.read_csv(source_projects)

for i in range(project_list.shape[0]):
    try:
        access_token = project_list.loc[i,'access_token']
        repo_owner = project_list.loc[i,'repo_owner']
        source_type = project_list.loc[i,'source_type']
        git_url = project_list.loc[i,'git_url']
        api_base_url = project_list.loc[i,'api_base_url']
        repo_name = project_list.loc[i,'repo_name'] 

        if platform.system() == 'Darwin' or platform.system() == 'Linux':
            data_path = os.getcwd() + '/data/' + repo_name + '/'
        else:
            data_path = os.getcwd() + '\\data\\' + repo_name + '\\'
        
        if not Path(data_path).is_dir():
            os.makedirs(Path(data_path))
        
        sg = social_interaction.create_social_inteaction_graph(repo_name)
        cg = code_interaction.create_code_interaction_graph(git_url,repo_name)
        
        bugs_data = buggy_commit.buggy_commit_maker(repo_name,git_url,repo_name)
        bugs_data.get_buggy_commits()
        buggy_commit_data = bugs_data.get_buggy_committer()

               
        
        sg_data = sg.get_user_node_degree()
        cg_data,bug_creator_df_final = cg.get_user_node_degree()


        #bug_creator_df_final.to_pickle(data_path + 'bug_introducing_commits.pkl')

        
        commit_data = bugs_data.get_commit_count()
        
        sg_data_list = []
        for key, value in sg_data.items():
            temp = [key,value]
            sg_data_list.append(temp)
        cg_data_list = []
        for key, value in cg_data.items():
            temp = [key,value]
            cg_data_list.append(temp)
            
        buggy_commit_data_df = pd.DataFrame(buggy_commit_data, columns = ['committer', 'count'])
        commit_data_df = pd.DataFrame(commit_data, columns = ['committer', 'count'])
        sg_data_df = pd.DataFrame(sg_data_list, columns = ['committer', 'count'])
        cg_data_df = pd.DataFrame(cg_data_list, columns = ['committer', 'count'])
        
        committer_count = []
        for i in range(bugs_data.commit.shape[0]):
            commit_id = bugs_data.commit.loc[i,'commit_number']
            user = bugs_data.repo.get(commit_id).committer
            committer_count.append([user.name, commit_id, 1])
        committer_count_df = pd.DataFrame(committer_count, columns = ['committer', 'commit_id', 'ob'])
        committer_count_df = committer_count_df.drop_duplicates()
        df = committer_count_df.groupby( ['committer']).count()
        commit_count = []
        for key,value in df.iterrows():
            user = key
            count = value.values.tolist()[0]
            commit_count.append([user,count])
        
        print("step1")
        results = [[buggy_commit_data_df,commit_data_df,commit_count,sg_data_df,cg_data_df]]
        result_pd = pd.DataFrame(results,columns = ['buggy_commit_data_df','commit_data_df',
                                                    'commit_count','sg_data_df','cg_data_df'])
        print("step2")
        result_pd.to_pickle(os.getcwd() + '/Processed_data/' + repo_name + '_final_results.pkl')

    except Exception as e:
        print("Exception occured for ",git_url)
        print(e)
