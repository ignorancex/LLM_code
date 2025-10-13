import os
#cache_dir = os.getenv("TMPDIR")
cache_dir = "/projects/prjs1335/cache/"

from tqdm import tqdm
import argparse
import json
import numpy as np
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mname", type=str, default="CohereForAI/aya-expanse-8b", help="LLM name on HF")
    parser.add_argument("--task", type=str, default="XQUAD_open", help="Task: XQUAD_open, MKQA_open")
    parser.add_argument("--CTI", type=float, default=1.0, help="AVG+CTI*STD")

    args = parser.parse_args()    
    # for mname_id, modelname in enumerate(["CohereForAI/aya-expanse-8b", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct"]):
    for mname_id, modelname in enumerate(["CohereForAI/aya-expanse-8b"]):
        mname = modelname.split('/')[-1]
        mirage_dir = f"mirage/{args.task}/{mname}/"
        if args.task == 'XQUAD_open':
            langs = ['en', 'es', 'de', 'ro', 'vi', 'tr', 'el', 'zh', 'ar', 'hi', 'ru', 'th']
            postfix = ['en', 'ar', 'de', 'el', 'es', 'hi', 'ro', 'ru', 'th', 'tr', 'vi', 'zh']
        elif args.task == 'MKQA_open':
            langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh']
            postfix = ["inlang_0", "inlang_1", "inlang_2", "outlang_0", "outlang_1", "outlang_2"]
        else:
            raise NotImplementedError
        
        inlang = []
        outlang = []
        for lang in langs:
            with open(f"results/{args.task}/{mname}_{lang}.json", encoding='utf-8') as f:
                data = json.load(f)
                f.close()
            count = defaultdict(int)
            total = defaultdict(int)
            for ins_id, ins in enumerate(data):
                if ins_id >= 500: break
                for _postfix in postfix:
                    attribution = json.load(open(f"{mirage_dir}{ins_id}_{lang}_{_postfix}.json"))
                    if attribution['cti_scores']:
                        if max(attribution['cti_scores']) > np.mean(attribution['cti_scores']) + args.CTI * np.std(attribution['cti_scores']): 
                            count[_postfix] += 1
                        total[_postfix] += 1
                        
            for k, v in count.items():
                count[k] = round(v/total[k], 2)
                if k == lang or 'inlang' in k:
                    inlang.append(round(v/total[k], 2))
                else:
                    outlang.append(round(v/total[k], 2))
                
        in_lang_res = round(sum(inlang)/len(inlang), 2)
        out_lang_res = round(sum(outlang)/len(outlang), 2)
        with open(f"count_mirage_{args.task}_{args.CTI}.txt", 'w') as f:
            f.write(str(in_lang_res)+' '+str(out_lang_res)+'\n')

if __name__ == "__main__":
    main()
