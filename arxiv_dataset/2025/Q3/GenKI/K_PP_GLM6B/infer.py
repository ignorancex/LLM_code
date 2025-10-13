from transformers import AutoModel
import torch
from transformers import AutoTokenizer
from peft import PeftModel
import json
from cover_alpaca2jsonl import format_example
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import os
def infer(indexfrom,indexto,indexsave):
    #sleep(indexsave)
    #print("!!")
    model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, load_in_8bit=True, device_map='auto')
    tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, "#output_dir")
    instructions = json.load(open("#data.json"))
    answers = []
    
    directory_path = "#ans_output_dir"
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    file_ans=open(directory_path+"/Answer"+str(indexsave)+".txt",'w',encoding="utf8")
    file_ans.write('')
    file_ans.close()
    with torch.no_grad():
        for idx, item in enumerate(instructions[indexfrom:indexto]):
            file_ans=open(directory_path+"/Answer"+str(indexsave)+".txt",'a',encoding="utf8")
            print(idx)
            feature = format_example(item)
            input_text = feature['context']
            ids = tokenizer.encode(input_text)
            input_ids = torch.LongTensor([ids])
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=130,
                do_sample=False,
                temperature=0
            )

            out_text = tokenizer.decode(out[0])
            file_ans.write(out_text+"\n@@@\n")
            #print("\n@@@output\n"+out_text)
            #answer = out_text.replace(input_text, "").replace("\nEND", "").strip()
            #item['infer_answer'] = answer
            #print("\n@@@input\n"+input_text[0:70]+"\n@@@answer:\n"+answer)
            #print(f"### {idx+1}.Answer:\n", item.get('output'), '\n\n')
            #answers.append({'index': idx, **item})
            file_ans.close()


parser = argparse.ArgumentParser(description='Demo of argparse')
 
# 2. 添加命令行参数
parser.add_argument('--saveI', type=int, default=0)
args = parser.parse_args()

infer(int((args.saveI-1)*700),int(args.saveI*700),args.saveI)