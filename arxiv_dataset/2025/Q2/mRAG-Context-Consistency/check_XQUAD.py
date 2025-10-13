import re
import json
from tqdm import tqdm
from collections import defaultdict

def contains_option(option_letter, response):
    pattern = r'\b[A-D]\b'
    match = re.search(pattern, response)
    if not match:
        return False
    else:
        if option_letter == match.group(0): return True
        else: return False

def check_substring(gold_answer_list, response):
    res = False
    for ans in set(gold_answer_list):
        if ans.lower() in response.lower():
            res = True
            break
    return res

langs = ['en', 'es', 'de', 'ro', 'vi', 'tr', 'el', 'zh', 'ar', 'hi', 'ru', 'th']


for mname_id, mname in enumerate(["aya-expanse-8b_{}.json", "Llama-3.2-3B-Instruct_{}.json", "gemma-2-9b-it_{}.json", "Qwen2.5-7B-Instruct_{}.json"]):
    print("======")
    print(mname)
    
    avg = [0 for _ in range(8)]
    heatmap = dict()
    heatmap_out = dict()
    for lang in langs:
        heatmap[lang] = [0 for _ in range(len(langs))]
        heatmap_out[lang] = [0 for _ in range(len(langs))]

    for lang_id, lang in enumerate(langs):
        load_path = f"results/XQUAD_open/"+mname.format(lang)
        with open(load_path, encoding="utf-8") as f:
            data = json.load(f)
        f.close()
        
        correct_non_ctx = 0
        correct_ctx_inlang = 0
        
        correct_ctx_outlang = defaultdict(int)
        correct_ctx_outlang_out = defaultdict(int)

        for d in data:
            if d['correct_non_ctx']: correct_non_ctx += 1
            if d[f'correct_ctx_{lang}']: correct_ctx_inlang += 1
            for plang_id, plang in enumerate(langs):
                if d[f'correct_ctx_{plang}']: heatmap[lang][plang_id] += 1
                if plang == lang: continue
                if d[f'correct_ctx_{plang}_out'] and not d[f'correct_ctx_{plang}']: heatmap_out[lang][plang_id] += 1
                
                if d[f'correct_ctx_{plang}']:
                    correct_ctx_outlang[plang] += 1
                if d[f'correct_ctx_{plang}_out'] and not d[f'correct_ctx_{plang}']:
                    correct_ctx_outlang_out[plang] += 1
        
        correct_ctx_outlang = {k: v/len(data) for k, v in correct_ctx_outlang.items()}
        correct_ctx_outlang_out = {k: v/len(data) for k, v in correct_ctx_outlang_out.items()}

        heatmap[lang] = [int(round(100*k/len(data), 0)) for k in heatmap[lang]]
        heatmap_out[lang] = [int(round(100*k/len(data), 0)) for k in heatmap_out[lang]]

        length = len(data)
        
        r_correct_ctx_outlang = {v: k for k, v in correct_ctx_outlang.items()}
        max_l = r_correct_ctx_outlang[max(correct_ctx_outlang.values())]
        min_l = r_correct_ctx_outlang[min(correct_ctx_outlang.values())]

        print(f"{lang} & {round(100 * correct_non_ctx/length, 1)} & {round(100*correct_ctx_inlang/length, 1)} & {round(100* sum(correct_ctx_outlang.values())/len(correct_ctx_outlang.values()), 1)} (+{round(100*sum(correct_ctx_outlang_out.values())/len(correct_ctx_outlang_out.values()), 1)}) & {round(100 * correct_ctx_outlang[max_l], 1)} (+{round(100*correct_ctx_outlang_out[max_l],1)}) & {round(100 * correct_ctx_outlang[min_l], 1)} (+{round(100*correct_ctx_outlang_out[min_l],1)})  \\\\")
        avg[0] += round(100 * correct_non_ctx/length, 1)
        avg[1] += round(100 * correct_ctx_inlang/length, 1)
        avg[2] += round(100 * sum(correct_ctx_outlang.values())/len(correct_ctx_outlang.values()), 1)
        avg[3] += round(100 * sum(correct_ctx_outlang_out.values())/len(correct_ctx_outlang_out.values()), 1)
        avg[4] += round(100 * correct_ctx_outlang[max_l], 1)
        avg[5] += round(100 * correct_ctx_outlang_out[max_l],1)
        avg[6] += round(100 * correct_ctx_outlang[min_l], 1)
        avg[7] += round(100 * correct_ctx_outlang_out[min_l],1)
    print('\\midrule')
    tmp = f"AVG"
    for i in range(3):
        tmp += f" & {round(avg[i]/12, 1)}"
    tmp += f" (+{round(avg[3]/12, 1)}) & {round(avg[4]/12, 1)} (+{round(avg[5]/12, 1)}) & {round(avg[6]/12, 1)} (+{round(avg[7]/12, 1)})"
    print(tmp + "\\\\")

    for k, v in heatmap.items():
        print(f"'{k}': {v},")
    for k, v in heatmap_out.items():
        print(f"'{k}': {v},")
