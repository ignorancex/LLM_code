import os
import json
import statistics

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

def write_json(path, data):
    with open(path, 'w') as f:
        json.dump(data,f, indent=4)
    print(f"Write to {path} done!")
    return

if False:
    # fix answer.. rewrite files
    #for mode in ['pine']:
    mode = 'orig'
    if True:
        results = []
        for i in range(24):
            path = f'./outputs/mmlu/mmlu_full/orig/mmlu_full/arrfeb/Qwen/Qwen1.5-7B-Chat/mode-{i}/result.json'
            print(path)
            result = read_json(path)
            results.append(result)
        for i in range(len(results[0])):
            gold_answer = results[0][i]['choices'][results[0][i]['answer']]
            for modenum in range(1, 24):
                try:
                    correct_answer = results[modenum][i]['choices'].index(gold_answer)
                except:
                    import pdb; pdb.set_trace()
                results[modenum][i]['answer'] = correct_answer
                pred_answer = {'A': 0, 'B': 1, 'C': 2, 'D': 3}[results[modenum][i]['prediction']]
                if pred_answer == correct_answer:
                    results[modenum][i]['accuracy'] = 1
                else:
                    results[modenum][i]['accuracy'] = 0
        for i in range(1, 24):
            path = f'./outputs/mmlu/mmlu_full/orig/mmlu_full/arrfeb/Qwen/Qwen1.5-7B-Chat/mode-{i}/result.json'
            #path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/sorting-method-freq/result.json'
            #path = f'./outputs/mmlu/mmlu_full/{mode}/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/result.json'
            write_json(path, results[i])
            #path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/sorting-method-freq/logs.json'
            path = f'./outputs/mmlu/mmlu_full/orig/mmlu_full/arrfeb/Qwen/Qwen1.5-7B-Chat/mode-{i}/logs.json'
            #path = f'./outputs/mmlu/mmlu_full/{mode}/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/logs.json'
            # get new acc
            accs = [x['accuracy'] for x in results[i]]
            new_acc = statistics.mean(accs)
            logdata = read_json(path)
            logdata['acc'] = new_acc
            write_json(path, logdata)



if False:
    # eval bulk for all 24 modes
    for mode in ['sorting-method-freq']:
    #for mode in ['pcw']:
    #for mode in ['orig', 'pine', 'ours']:
        accs = []
        for i in range(24):
            #path = f'./outputs/mmlu/mmlu_full/pine/mmlu_full/arrfeb/Qwen/Qwen1.5-7B-Chat/mode-{i}/logs.json'
            path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/arrfeb/Qwen/Qwen1.5-7B-Chat/mode-{i}/sorting-method-monot5/logs.json'
            #path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/sorting-method-freq/logs.json'
            #path = f'./outputs/mmlu/mmlu_full/{mode}/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/logs.json'
            try:
                result = read_json(path)
                acc = result['acc']
            except:
                acc = 0
            accs.append(acc)
        print(f"For mode: {mode}")
        print(", ".join([str(x) for x in accs]))

if False:
    # get best, worst
    #for mode in ['orig', 'ours']:
    for mode in ['pine']:
        total_res = []
        for i in range(24):
            path = f'./outputs/mmlu/mmlu_full/{mode}/mmlu_full/arrfeb/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/result.json'
            result = read_json(path)
            total_res.append(result)
        best_acc = []
        worst_acc = []
        half_corr = []
        consistency = []
        for variants in zip(*total_res):
            var_accs = [x['accuracy'] for x in variants]
            if len(set(var_accs)) == 1:
                consistency.append(1)
            else:
                consistency.append(0)
            if 0 in var_accs:
                worst_acc.append(0)
            else:
                worst_acc.append(1)
            if 1 in var_accs:
                best_acc.append(1)
            else:
                best_acc.append(0)
            if sum(var_accs) >= 12:
                half_corr.append(1)
            else:
                half_corr.append(0)
        best_acc = statistics.mean(best_acc)
        consistency_acc = statistics.mean(consistency)
        worst_acc = statistics.mean(worst_acc)
        half_corr = statistics.mean(half_corr)
        print(f"For mode: {mode}, best: {best_acc}, worst: {worst_acc}, consistency: {consistency_acc}, half_corr: {half_corr}")



if False:
    path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-0/result.json'
    origpath = f'./outputs/mmlu/mmlu_full/orig/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-0/result.json'
    orig, ours = [read_json(origpath), read_json(path)]
    total_acc = []
    ours_cnt = 0
    for x, y in zip(orig, ours):
        orig_c = x['confidence']
        ours_c = y['confidence']
        total_acc.append(y['accuracy'] or x['accuracy'])

    accs = statistics.mean(total_acc)
    print(accs)
    import pdb; pdb.set_trace()


if True:
    # get routing acc
    total_res = []
    for i in range(24):
        pair = []
        model_name = 'Qwen/Qwen1.5-4B-Chat'
        #model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        for mode in ['orig', 'ours']:
            if mode == 'ours':
                path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/arrfeb/Qwen/Qwen1.5-4B-Chat/mode-{i}/result.json'
                #path = f'./outputs/mmlu/mmlu_full/ours/mmlu_full/arrfeb/Qwen/Qwen1.5-7B-Chat/mode-{i}/sorting-method-freq/result.json'
                #path = f'./outputs/mmlu/mmlu_full/{mode}/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/sorting-method-freq/result.json'
            else:

                path = f'./outputs/mmlu/mmlu_full/orig/mmlu_full/arrfeb/Qwen/Qwen1.5-4B-Chat/mode-{i}/result.json'
                #path = f'./outputs/mmlu/mmlu_full/{mode}/mmlu_full/perm/meta-llama/Meta-Llama-3.1-8B-Instruct/mode-{i}/result.json'

            result = read_json(path)
            pair.append(result)
        total_res.append(pair)
    accs = []
    ours_p = []
    for i in range(24):
        orig, ours = total_res[i]
        total_acc = []
        ours_cnt = 0
        for x, y in zip(orig, ours):
            orig_c = x['confidence']
            ours_c = y['confidence']
            #total_acc.append(y['accuracy'] or x['accuracy'])
            if orig_c + 0.2 < ours_c: # route to ours
                total_acc.append(y['accuracy'])
                ours_cnt += 1
            else: # route to orig
                total_acc.append(x['accuracy'])
        accs.append(statistics.mean(total_acc))
        ours_p.append(ours_cnt / 14015)
    print(", ".join([str(x) for x in accs]))
    print(f'Ours cnt: ')
    print(", ".join([str(x) for x in ours_p]))

