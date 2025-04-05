import json
import argparse

def evaluate_acc(golden, preds):
    gold_answer = json.load(open(golden, 'r'))
    pred_answer = json.load(open(preds,'r'))
    acc = 0
    miss = 0
    gold_answer_dict = {}
    for g_ans in gold_answer:
        gold_answer_dict[g_ans['question_id']] = g_ans
    gold_answer= gold_answer_dict


    for p_ans in pred_answer:
        if p_ans['question_id'] not in gold_answer:
            miss += 1
            continue
        g_ans = gold_answer[p_ans['question_id']]
        
        if p_ans['prediction'] == list(g_ans['label'].keys())[0]:
            acc += 1
    print("missing ", miss)
    print("acc {:.4f}".format(acc/(len(gold_answer)-miss)))
    return acc/(len(gold_answer)-miss)

def evaluate_metrix1(golden, preds, aas_path):
    gold_answer = json.load(open(golden, 'r'))
    pred_answer = json.load(open(preds,'r'))
    aas = json.load(open(aas_path, "r"))
    metric1 = 0
    miss = 0

    for g_ans, p_ans in zip(gold_answer, pred_answer):
        if g_ans['question_id'] != p_ans['question_id']:
            miss+=0
            continue
        label = list(g_ans['label'].keys())[0]
        try:
            aas_l = aas[label] 
        except:
            continue
        
        try:
            if p_ans['prediction'] in aas_l:
                metric1 += aas_l[p_ans['prediction'] ]
        except:
            print(p_ans)
            print(aas_l)
            exit()
    print("# of missing answers: ", miss)
    print("metric1 {:.4f}".format(metric1/len(gold_answer)))
    
    return metric1/len(gold_answer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluate the model performance using aas soft metric')
    parser.add_argument('--prediction_file',  type=str, default='gqa_prediction/testdev_predict_lxmert.json',
                        help='the path of the prediction file')
    parser.add_argument('--golden_testing_file',  type=str, default='gqa_testdev.json',
                        help='the path of the prediction file')
    parser.add_argument('--dataset_type',  type=str, default='gqa',
                        help='choose gqa or vqa')
    args = parser.parse_args()

    if args.dataset_type == "gqa":
        aas_path = [ 'aas_gqa_files/wordNet_aas.json','aas_gqa_files/bert_aas.json', 'aas_gqa_files/counterfit_aas.json', 'aas_gqa_files/conceptNet_aas.json', 'aas_gqa_files/union_5_aas.json']
    
    
    for path in aas_path:
        print(path.split("/")[-1])
        score = evaluate_metrix1(args.golden_testing_file, args.prediction_file, path)
        
        print("="*30)
    
    