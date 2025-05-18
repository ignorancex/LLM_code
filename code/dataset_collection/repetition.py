import json 
from collections import defaultdict 
from datetime import datetime 
input_file ='LLM_code/dataset/github_links/Only_links.json'
output_file ='LLM_code/dataset/github_links/Only_links_set.json'
with open (input_file ,'r',encoding ='utf-8')as f :
    data =json .load (f )
link_groups =defaultdict (list )
for item in data :
    link =item .get ('github_links')
    if link :
        link_groups [link ].append (item )
deduplicated_data =[]
for (link ,items )in link_groups .items ():
    if len (items )==1 :
        deduplicated_data .append (items [0 ])
    else :
        try :
            dates =[datetime .strptime (item ['update_date'],'%Y-%m-%d')for item in items ]
            max_date =max (dates )
            min_date =min (dates )
            delta_days =(max_date -min_date ).days 
            if delta_days <=365 :
                latest_item =max (items ,key =lambda x :x ['update_date'])
                deduplicated_data .append (latest_item )
            else :
                continue 
        except Exception as e :
with open (output_file ,'w',encoding ='utf-8')as f :
    json .dump (deduplicated_data ,f ,indent =4 )