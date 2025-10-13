import os
#cache_dir = os.getenv("TMPDIR")
cache_dir = "/projects/prjs1335/cache/"

from tqdm import tqdm
import argparse
import json
import numpy as np
from collections import defaultdict

def mirage_cite(res_mirage, cti_threshold, start_pos_sent, end_pos_sent, topk_CCI, doc_seps):
    res = []

    sum_weight = 0
    sum_value = np.zeros(len(res_mirage['input_context_tokens']))
    
    for i in res_mirage['cci_scores']:
        # CTI Filtering
        if not (i["cti_idx"] >= start_pos_sent and i["cti_idx"] < end_pos_sent): continue
        if i['cti_score'] >= cti_threshold:
            # CCI Focus
            CCI_value = np.array(i['input_context_scores'])
            if topk_CCI == 0:
                cci_threshold = np.mean(CCI_value)
            elif topk_CCI < 0:
                cci_threshold = (1+topk_CCI/100) * np.max(CCI_value) - topk_CCI/100 * np.min(CCI_value)
            else:
                cci_threshold = np.sort(CCI_value)[-topk_CCI]
            zero_idx = CCI_value < cci_threshold
            CCI_value[zero_idx] = 0

            sum_value += CCI_value

        if i['cti_score'] < cti_threshold: break

    sum_tmp = 0
    for i, v in enumerate(sum_value):
        sum_tmp += v
        if doc_seps[i] or (i == len(sum_value)-1): # meet '\n'
            res.append(sum_tmp)
            sum_tmp = 0
    return res

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--mname", type=str, default="CohereForAI/aya-expanse-8b", help="LLM name on HF")
    parser.add_argument("--task", type=str, default="MKQA_open_multi", help="Task: XQUAD_open, MKQA_open")
    parser.add_argument("--CTI", type=float, default=1.0, help="AVG+CTI*STD")
    parser.add_argument("--CCI", type=float, default=-5, help="Threshold for CCI")

    args = parser.parse_args()    
    # for mname_id, modelname in enumerate(["CohereForAI/aya-expanse-8b", "meta-llama/Llama-3.2-3B-Instruct", "google/gemma-2-9b-it", "Qwen/Qwen2.5-7B-Instruct"]):
    for mname_id, modelname in enumerate(["CohereForAI/aya-expanse-8b"]):
        mname = modelname.split('/')[-1]
        mirage_dir = f"mirage/{args.task}/{mname}/"
        if args.task == 'MKQA_open_multi':
            langs = ['ar', 'da', 'de', 'en', 'es', 'fi', 'fr', 'he', 'hu', 'it', 'ja', 'ko', 'km', 'ms', 'nl', 'no', 'pl', 'pt', 'ru', 'sv', 'th', 'tr', 'vi', 'zh']
            postfix = ['1o3i_0', '1o3o_0', '3o1i_0', '3o1o_0']
        elif args.task == 'GMMLU_choice_multi':
            langs = ['ar', 'bn', 'de', 'en', 'es', 'fr', 'hi', 'id', 'it', 'ja', 'ko', 'pt', 'sw', 'yo', 'zh', 'cs', 'fa', 'tr', 'ro', 'si', 'am', 'te', 'uk', 'vi', 'ru', 'ms', 'el', 'fil', 'ha', 'he', 'ig', 'ky', 'lt', 'mg', 'ne', 'nl', 'ny', 'pl', 'sn', 'so', 'sr', 'sv']
            postfix = ['1o3i_0', '1o3o_0', '3o1i_0', '3o1o_0']
        else:
            raise NotImplementedError
        
        output_results = dict()
        # Go over all languages
        for lang in tqdm(langs):
            print(lang)
            ilo_1o = 0.0
            ilo_3o = 0.0
            total = 0.0

            in_1o = 0.0
            out_1o = 0.0
            in_3o = 0.0
            out_3o = 0.0

            favor_in_1o = 0.0
            favor_out_1o = 0.0
            favor_in_3o = 0.0
            favor_out_3o = 0.0

            with open(f"results/{args.task}/{mname}_{lang}.json", encoding='utf-8') as f:
                data = json.load(f)
            f.close()
            with open(f"{mirage_dir}/success_{lang}.json") as f:
                success_list = json.load(f)
            f.close()
                    
            for ins_id in success_list:
                ins = data[ins_id]
                res = []
                res_num = []
                res_favor = []
                for _postfix in postfix:
                    gold_prefix = _postfix.split('_')[0][:2]
                    gold_docs = ins[f'prompt_{gold_prefix}_0'].split('<|USER_TOKEN|>')[1].split('<|END_OF_TURN_TOKEN|>')[0].split('\n')[:-1]
                    gold_docs = [gd[3:] for gd in gold_docs]
                    
                    full_docs = ins[f'prompt_{_postfix}'].split('<|USER_TOKEN|>')[1].split('<|END_OF_TURN_TOKEN|>')[0].split('\n')[:4]
                    full_docs = [gd[3:] for gd in full_docs]
                    mark_dist = [False if dc in gold_docs else True for dc in full_docs]

                    # print(f"{mirage_dir}{ins_id}_{lang}_{_postfix}.json")
                    res_mirage = json.load(open(f"{mirage_dir}{ins_id}_{lang}_{_postfix}.json"))
                    cti_threshold = np.mean(res_mirage["cti_scores"]) + args.CTI * np.std(res_mirage["cti_scores"])
                    doc_seps = np.array(res_mirage["input_context_tokens"])
                    doc_seps = doc_seps == '\u010a'
                    cci_scores = mirage_cite(res_mirage, cti_threshold, 0, 50000, args.CCI, doc_seps)
                    assert sum(mark_dist) == 1 or sum(mark_dist) == 3
                    
                    avg_cci = np.mean([i for _i, i in enumerate(cci_scores) if mark_dist[_i]])
                    res.append(avg_cci)

                    avg_num = np.sum([True if mark_dist[_i] and i > 0 else False for _i, i in enumerate(cci_scores)])
                    res_num.append(avg_num)

                    highest_CCI_gold = np.max([i for _i, i in enumerate(cci_scores) if not mark_dist[_i]])
                    favor = np.sum([True if mark_dist[_i] and i > highest_CCI_gold else False for _i, i in enumerate(cci_scores)])
                    res_favor.append(favor)

                # print(res)
                if res[0] >= res[1]: ilo_1o += 1
                if res[2] >= res[3]: ilo_3o += 1
                
                in_1o += res_num[0]
                out_1o += res_num[1]
                in_3o += res_num[2]
                out_3o += res_num[3]

                favor_in_1o += res_favor[0]
                favor_out_1o += res_favor[1]
                favor_in_3o += res_favor[2]
                favor_out_3o += res_favor[3]

                total += 1
            
            output_results[lang] = [
                round(ilo_1o/total*100, 0), 
                round(ilo_3o/total*100, 0),
                round(in_1o/total, 1),
                round(out_1o/total, 1),
                round(in_3o/total, 1),
                round(out_3o/total, 1),
                round(favor_in_1o/total, 1),
                round(favor_out_1o/total, 1),
                round(favor_in_3o/total, 1),
                round(favor_out_3o/total, 1),
                ilo_1o, ilo_3o, total, in_1o, out_1o, in_3o, out_3o]
            print(output_results[lang])
        with open(f"count_mirage_{args.task}_{args.CTI}_{args.CCI}.json", 'w') as f:
            json.dump(output_results, f)

if __name__ == "__main__":
    main()
