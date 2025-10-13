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

def check_substring(gold_ans_list, response):
    res = False
    for ans in set(gold_ans_list):
        if ans.lower() in response.lower():
            res = True
            break
    return res

langs = ['en', 'it', 'es', 'de', 'fr', 'pt', 'nl', 'sv', 'ru', 'fi', 'ja', 'pl', 'no', 'tr', 'hu', 'da', 'vi', 'he', 'ar', 'ms', 'ko', 'th', 'zh', 'km']

num_lang_line = 12

res = dict()
for mname_id, mname in enumerate(["aya-expanse-8b_{}.json", "Llama-3.2-3B-Instruct_{}.json", "gemma-2-9b-it_{}.json", "Qwen2.5-7B-Instruct_{}.json"]):
    res[mname] = dict()
    print("======")
    print(mname)
    
    for lang_id, lang in enumerate(tqdm(langs)):
        load_path = f"results/MKQA_open_multi/"+mname.format(lang)
        with open(load_path, encoding="utf-8") as f:
            data = json.load(f)
        f.close()

        # no ctx
        correct_non_ctx = 0

        # 1o
        correct_1o = 0
        correct_1o3i = 0
        correct_1o3o = 0

        correct_1o_out = 0
        correct_1o3i_out = 0
        correct_1o3o_out = 0
        correct_1o3o_dist = 0

        # 3o
        correct_3o = 0
        correct_3o1i = 0
        correct_3o1o = 0

        correct_3o_out = 0
        correct_3o1i_out = 0
        correct_3o1o_out = 0
        correct_3o1o_dist = 0

        # 1i
        correct_1i = 0
        correct_1i3i = 0
        correct_1i3o = 0

        correct_1i3o_dist = 0

        # 3i
        correct_3i = 0
        correct_3i1i = 0
        correct_3i1o = 0
        
        correct_3i1o_dist = 0


        count = 0
        for d in data:
            correct_non_ctx += d['correct_non_ctx']
            count+=1
            for i in range(3):
                correct_1o += d[f'correct_1o_{i}']
                correct_1o3i += d[f'correct_1o3i_{i}']
                correct_1o3o += d[f'correct_1o3o_{i}']

                correct_1o_out += (not d[f'correct_1o_{i}'] and check_substring(d[f'answer_out_1o3o_{i}'], d[f'response_1o_{i}']))
                correct_1o3i_out += (not d[f'correct_1o3i_{i}'] and check_substring(d[f'answer_out_1o3o_{i}'], d[f'response_1o3i_{i}']))
                correct_1o3o_out += (not d[f'correct_1o3o_{i}'] and d[f'correct_out_1o3o_{i}'])
                correct_1o3o_dist += (not d[f'correct_1o3o_{i}'] and not d[f'correct_out_1o3o_{i}'] and d[f'correct_out_dist_1o3o_{i}'])

                correct_3o += d[f'correct_3o_{i}']
                correct_3o1i += d[f'correct_3o1i_{i}']
                correct_3o1o += d[f'correct_3o1o_{i}']

                correct_3o_out += (not d[f'correct_3o_{i}'] and check_substring(d[f'answer_out_3o1o_{i}'], d[f'response_3o_{i}']))
                correct_3o1i_out += (not d[f'correct_3o1i_{i}'] and check_substring(d[f'answer_out_3o1o_{i}'], d[f'response_3o1i_{i}']))
                correct_3o1o_out += (not d[f'correct_3o1o_{i}'] and d[f'correct_out_3o1o_{i}'])
                correct_3o1o_dist += (not d[f'correct_3o1o_{i}'] and not d[f'correct_out_3o1o_{i}'] and d[f'correct_out_dist_3o1o_{i}'])

                correct_1i += d[f'correct_1i_{i}']
                correct_1i3i += d[f'correct_1i3i_{i}']
                correct_1i3o += d[f'correct_1i3o_{i}']

                correct_1i3o_dist += (not d[f'correct_1i3o_{i}'] and d[f'correct_out_dist_1i3o_{i}'])

                correct_3i += d[f'correct_3i_{i}']
                correct_3i1i += d[f'correct_3i1i_{i}']
                correct_3i1o += d[f'correct_3i1o_{i}']

                correct_3i1o_dist += (not d[f'correct_3i1o_{i}'] and d[f'correct_out_dist_3i1o_{i}'])


        # print(count)
        length = len(data)
        # print(length)
        correct_non_ctx = round(100*correct_non_ctx/length, 1)

        correct_1o = round(100*correct_1o/(3*length), 1)
        correct_1o3i = round(100*correct_1o3i/(3*length), 1)
        correct_1o3o = round(100*correct_1o3o/(3*length), 1)

        correct_1o_out = round(100*correct_1o_out/(3*length), 1)
        correct_1o3i_out = round(100*correct_1o3i_out/(3*length), 1)
        correct_1o3o_out = round(100*correct_1o3o_out/(3*length), 1)
        correct_1o3o_dist = round(100*correct_1o3o_dist/(3*length), 1)

        correct_3o = round(100*correct_3o/(3*length), 1)
        correct_3o1i = round(100*correct_3o1i/(3*length), 1)
        correct_3o1o = round(100*correct_3o1o/(3*length), 1)

        correct_3o_out = round(100*correct_3o_out/(3*length), 1)
        correct_3o1i_out = round(100*correct_3o1i_out/(3*length), 1)
        correct_3o1o_out = round(100*correct_3o1o_out/(3*length), 1)
        correct_3o1o_dist = round(100*correct_3o1o_dist/(3*length), 1)

        correct_1i = round(100*correct_1i/(3*length), 1)
        correct_1i3i = round(100*correct_1i3i/(3*length), 1)
        correct_1i3o = round(100*correct_1i3o/(3*length), 1)

        correct_1i3o_dist = round(100*correct_1i3o_dist/(3*length), 1)

        correct_3i = round(100*correct_3i/(3*length), 1)
        correct_3i1i = round(100*correct_3i1i/(3*length), 1)
        correct_3i1o = round(100*correct_3i1o/(3*length), 1)

        correct_3i1o_dist = round(100*correct_3i1o_dist/(3*length), 1)
        
        res[mname][lang] = {
            'correct_No Ctx': correct_non_ctx, 
            'correct_1o': correct_1o,
            'correct_1o3i': correct_1o3i,
            'correct_1o3o': correct_1o3o,
            'correct_1o_out': correct_1o_out,
            'correct_1o3i_out': correct_1o3i_out,
            'correct_1o3o_out': correct_1o3o_out,
            'correct_1o3o_dist': correct_1o3o_dist,

            'correct_3o': correct_3o,
            'correct_3o1i': correct_3o1i,
            'correct_3o1o': correct_3o1o,
            'correct_3o_out': correct_3o_out,
            'correct_3o1i_out': correct_3o1i_out,
            'correct_3o1o_out': correct_3o1o_out,
            'correct_3o1o_dist': correct_3o1o_dist,

            'correct_1i': correct_1i,
            'correct_1i3i': correct_1i3i,
            'correct_1i3o': correct_1i3o,
            'correct_1i3o_dist': correct_1i3o_dist,

            'correct_3i': correct_3i,
            'correct_3i1i': correct_3i1i,
            'correct_3i1o': correct_3i1o,
            'correct_3i1o_dist': correct_3i1o_dist,
        }

    row_name = ['Setups', 'No Ctx', '1o', '1o_out', '1o3i', '1o3i_out', '1o3o', '1o3o_out', '3o', '3o_out', '3o1i', '3o1i_out', '3o1o', '3o1o_out', '1i', '1i3i', '1i3o', '3i', '3i1i', '3i1o']
    lines = [i for i in row_name]
    sum_res = [0 for _ in range(len(row_name)-1)]
    for lang_id, lang in enumerate(langs):
        lines[0] += " & " + lang
        for line_id, line_name in enumerate(row_name[1:]):
            if 'out' not in line_name:
                lines[line_id+1] += " & " + str(res[mname][lang][f'correct_{line_name}'])
            else:
                lines[line_id+1] += f" (+{str(res[mname][lang][f'correct_{line_name}'])}) "
            sum_res[line_id] += res[mname][lang][f'correct_{line_name}']
        if (lang_id + 1) % num_lang_line == 0:
            for r, line in enumerate(lines):
                print(line + " \\\\")
                if r in [0, 1, 4, 7, 10]: print('\\midrule')
                if r == 13: print('\\bottomrule')
            lines = [i for i in row_name]
    print()
    for sum_res_id, sum_res_ in enumerate(sum_res):
        print(f"{row_name[sum_res_id+1]}: {round(sum_res_/24,1)}")
    print()
    for sum_res_id, sum_res_ in enumerate(sum_res):
        print(f" & {round(sum_res_/24,1)}")
    print()


