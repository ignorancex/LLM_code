import json 
input_file ='LLM_code/codeforces/problem_descriptions.jsonl'
output_file ='LLM_code/codeforces/problem_1000.jsonl'
results =[]
with open (input_file ,'r',encoding ='utf-8')as f :
    for line in f :
        if not line .strip ():
            continue 
        try :
            item =json .loads (line )
            if item .get ('difficulty')==1000 :
                results .append (item )
        except json .JSONDecodeError :
with open (output_file ,'w',encoding ='utf-8')as f :
    for item in results :
        json .dump (item ,f ,ensure_ascii =False )
        f .write ('\n')