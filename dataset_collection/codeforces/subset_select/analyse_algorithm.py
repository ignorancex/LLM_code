import json 
from collections import defaultdict 
def get_difficulty_bucket (difficulty ):
    if 800 <=difficulty <=1199 :
        return '800-1199'
    elif 1200 <=difficulty <=1599 :
        return '1200-1599'
    elif 1600 <=difficulty <=1999 :
        return '1600-1999'
    elif difficulty >=2000 :
        return '2000+'
    else :
        return None 
input_path ='LLM_code/codeforces/subset_select/benchmark_200.jsonl'
stats =defaultdict (lambda :defaultdict (int ))
with open (input_path ,'r',encoding ='utf-8')as f :
    for line in f :
        data =json .loads (line )
        difficulty =data .get ('difficulty')
        algorithm =data .get ('main_algorithm')
        if difficulty is None or algorithm is None :
            continue 
        bucket =get_difficulty_bucket (difficulty )
        if bucket :
            stats [bucket ][algorithm ]+=1 
for bucket in ['800-1199','1200-1599','1600-1999','2000+']:
    for (algo ,count )in sorted (stats [bucket ].items (),key =lambda x :-x [1 ]):