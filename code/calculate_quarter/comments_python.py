import os 
import tokenize 
import json 
import csv 
from tqdm import tqdm 
def load_repo_field_mapping (json_path ):
    with open (json_path ,'r',encoding ='utf-8')as f :
        data =json .load (f )
    repo_field_map ={}
    for (quarter ,entries )in data .items ():
        for item in entries :
            try :
                repo_url =item ['link']
                repo_name =repo_url .rstrip ('/').split ('/')[-1 ]
                category =item .get ('categories','')
                if category .startswith ('cs.'):
                    repo_field_map [repo_name ]='cs'
                else :
                    repo_field_map [repo_name ]='non-cs'
            except Exception as e :
    return repo_field_map 
repo_field_map =load_repo_field_mapping ('LLM_code/code/github_links/python_dataset_links_1.json')
def count_comment_and_total_lines_tokenize (file_path ):
    comment_lines =set ()
    total_lines =0 
    try :
        with open (file_path ,'rb')as f :
            tokens =tokenize .tokenize (f .readline )
            for token in tokens :
                if token .type ==tokenize .COMMENT :
                    comment_lines .add (token .start [0 ])
                elif token .type in (tokenize .NEWLINE ,tokenize .NL ):
                    total_lines +=1 
    except Exception as e :
        return (0 ,0 )
    return (len (comment_lines ),total_lines )
def compute_quarter_avg_comment_ratio (quarter_path ):
    cs_ratios =[]
    noncs_ratios =[]
    for repo in os .listdir (quarter_path ):
        repo_path =os .path .join (quarter_path ,repo )
        if not os .path .isdir (repo_path ):
            continue 
        repo_comment =0 
        repo_total =0 
        for (root ,_ ,files )in os .walk (repo_path ):
            for file in files :
                if file .endswith ('.py'):
                    file_path =os .path .join (root ,file )
                    (c_lines ,t_lines )=count_comment_and_total_lines_tokenize (file_path )
                    repo_comment +=c_lines 
                    repo_total +=t_lines 
        if repo_total >0 :
            ratio =repo_comment /repo_total 
            field =repo_field_map .get (repo ,'unknown')
            if field =='cs':
                cs_ratios .append (ratio )
            elif field =='non-cs':
                noncs_ratios .append (ratio )
    cs_avg =sum (cs_ratios )/len (cs_ratios )if cs_ratios else 0.0 
    noncs_avg =sum (noncs_ratios )/len (noncs_ratios )if noncs_ratios else 0.0 
    return (cs_avg ,noncs_avg )
base_dir ='LLM_code/arxiv_dataset'
results ={}
for year in sorted (os .listdir (base_dir )):
    if not year .isdigit ()or not 2020 <=int (year )<=2025 :
        continue 
    year_path =os .path .join (base_dir ,year )
    if not os .path .isdir (year_path ):
        continue 
    for quarter in sorted (os .listdir (year_path )):
        quarter_path =os .path .join (year_path ,quarter )
        if not os .path .isdir (quarter_path ):
            continue 
        quarter_key =f'{year}_{quarter}'
        (cs_avg ,noncs_avg )=compute_quarter_avg_comment_ratio (quarter_path )
        results [quarter_key ]={'cs':cs_avg ,'noncs':noncs_avg }
csv_path ='LLM_code/arxiv_result/comments/comment_ratio_python_by_group.csv'
os .makedirs (os .path .dirname (csv_path ),exist_ok =True )
with open (csv_path ,'w',newline ='')as f :
    writer =csv .writer (f )
    writer .writerow (['Quarter','CS_Comment_Ratio','NonCS_Comment_Ratio'])
    for (q ,r )in results .items ():
        writer .writerow ([q ,f"{r['cs']:.4f}",f"{r['noncs']:.4f}"])