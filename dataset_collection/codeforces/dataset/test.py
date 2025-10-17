import re 
def remove_control_characters (json_string ):
    return re .sub ('[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f]','',json_string )
input_path ='/media/sata3/siming/LLM_code/codeforces/problem.json'
output_path ='/media/sata3/siming/LLM_code/codeforces/problem_cleaned.json'
with open (input_path ,'r',encoding ='utf-8')as f :
    raw =f .read ()
cleaned =remove_control_characters (raw )
import json 
try :
    parsed =json .loads (cleaned )
    with open (output_path ,'w',encoding ='utf-8')as f :
        json .dump (parsed ,f ,ensure_ascii =False ,indent =2 )
except json .JSONDecodeError as e :