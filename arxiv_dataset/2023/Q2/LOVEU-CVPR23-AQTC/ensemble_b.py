import json
import numpy as np

def softmax(x):
    exp_x = np.exp(x)
    sum_exp_x = np.sum(exp_x, axis=1, keepdims=True)
    softmax_x = exp_x / sum_exp_x
    return softmax_x

fileroot = '/data/zclfe/cvpr_comp/LOVEU-CVPR22-AQTC/outputs/cvpr_loveu2023/'

exp_names = []

seeds=[0,1,2,3,4,5,6,7,8, 42, 240,789,1024,2048,5555,77,99,120,400] # [0 1 2 3 4 5 6 7 8 42 240 789 1024 2048 5555 77 99 120 400]
dims=[256, 512,768, 1024,1536, 2048]

for seed in seeds:
    for dim in dims:
        exp_names.append(f'more_train_l4_stateinput_{dim}_seed_{seed}')

print(len(exp_names))
file_num = len(exp_names)
result = None

for name in exp_names:
    sub_name = fileroot+name+'/submit_test.json'
    with open(sub_name,'r') as fin:
        file = json.load(fin)
        if result is None:
            result = {}
            for mname in file:
                result[mname]=[]
                qa_list = file[mname]
                for qa in qa_list:
                    new_qa = {}
                    for k,v in qa.items():
                        if k=='question':
                            new_qa[k]=v
                        elif k=='scores':
                            new_qa[k]=[]
                            for step_a in v:
                                new_qa[k].append(np.zeros_like(np.array(step_a)))
                    result[mname].append(new_qa)

        for mname in file:
            qa_list = file[mname]
            for qa in qa_list:
                sum_all = result[mname]
                for sums in sum_all:
                    if sums['question'] == qa['question']:
                        for i, score in enumerate(qa['scores']):
                            score = np.array(score)
                            sums['scores'][i] += score/file_num
                        break

for mname in result:
    qa_list = result[mname]
    for qa in qa_list:
        for i in range(len(qa['scores'])):
            score = np.expand_dims(qa['scores'][i], axis=0)
            score = softmax(score)
            score = np.squeeze(score,axis=0)
            qa['scores'][i] = score.tolist()


with open('submit_test.json','w')as f:
    json.dump(result,f)
