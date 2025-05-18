import json 
from datetime import datetime 
from collections import defaultdict 
input_path ='LLM_code/code/github_links/Only_links_set.json'
output_path ='LLM_code/code/github_links/Only_links_by_quarter.json'
start_date =datetime (2020 ,1 ,1 )
end_date =datetime (2025 ,3 ,31 )
def get_quarter (date_str ):
    try :
        dt =datetime .strptime (date_str ,'%Y-%m-%d')
        if start_date <=dt <=end_date :
            quarter =(dt .month -1 )//3 +1 
            return f'{dt.year}Q{quarter}'
    except Exception as e :
    return None 
with open (input_path ,'r',encoding ='utf-8')as f :
    full_data =json .load (f )
grouped =defaultdict (list )
for item in full_data :
    link =item .get ('github_links')
    update_date =item .get ('update_date')
    quarter =get_quarter (update_date )
    if quarter and link :
        grouped [quarter ].append (link )
def quarter_sort_key (q ):
    (year ,qtr )=q .split ('Q')
    return int (year )*10 +int (qtr )
sorted_grouped ={quarter :grouped [quarter ]for quarter in sorted (grouped .keys (),key =quarter_sort_key )}
with open (output_path ,'w',encoding ='utf-8')as f :
    json .dump (sorted_grouped ,f ,indent =4 )