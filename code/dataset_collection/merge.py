import json 
json_files =['LLM_code/dataset/github_links/links_non_empty/link_2020_filtered.json','LLM_code/dataset/github_links/links_non_empty/link_2021_filtered.json','LLM_code/dataset/github_links/links_non_empty/link_2022_filtered.json','LLM_code/dataset/github_links/links_non_empty/link_2023_filtered.json','LLM_code/dataset/github_links/links_non_empty/link_2024_filtered.json']
output_path ='LLM_code/dataset/github_links/links_non_empty.json'
all_links =[]
for file_path in json_files :
    try :
        with open (file_path ,'r',encoding ='utf-8')as f :
            data =json .load (f )
            links =data .get ('github_links',[])
            all_links .extend (links )
    except Exception as e :
all_links =list (set (all_links ))
with open (output_path ,'w',encoding ='utf-8')as f :
    json .dump ({'github_links':all_links },f ,indent =4 )