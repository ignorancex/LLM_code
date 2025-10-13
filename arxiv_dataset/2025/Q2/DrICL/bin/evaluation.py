import torch
from nltk.util import bigrams, trigrams
import json
from nltk.translate.bleu_score import sentence_bleu
from rouge_chinese import Rouge
import jieba
import sys
import re
from math import log2

# MAX_CONTEXT_LEN = 8e3
# MAX_CONTEXT_LEN = 3.2e4
MAX_CONTEXT_LEN = 1.28e5

MODEL_DIR = "../models/llama2-hf-chat-7B/"
from transformers import GPTSw3Tokenizer
tokenizer = GPTSw3Tokenizer.from_pretrained(
    pretrained_model_name_or_path=MODEL_DIR,
    trust_remote_code=True,
    max_length=MAX_CONTEXT_LEN)

# Dist-1/Dist-2/Dist-3 metrics
def eval_distinct_metrics(gold_text, predict_text):
    total_unigram_cnt = 0
    total_bigram_cnt = 0
    total_trigram_cnt = 0
    dist_unigram_tokens = set()
    dist_bigram_tokens = set()
    dist_trigram_tokens = set()
    for i in range(len(predict_text)):
        pred_sent = predict_text[i]
        unigram_tokens = list(pred_sent)
        bigram_tokens = list(bigrams(unigram_tokens))
        trigram_tokens = list(trigrams(unigram_tokens))
        total_unigram_cnt += len(unigram_tokens)
        total_bigram_cnt += len(bigram_tokens)
        total_trigram_cnt += len(trigram_tokens)
        dist_unigram_tokens = set.union(dist_unigram_tokens, set(unigram_tokens))
        dist_bigram_tokens = set.union(dist_bigram_tokens, set(bigram_tokens))
        dist_trigram_tokens = set.union(dist_trigram_tokens, set(trigram_tokens))
    #print('D-1: %s, D-2: %s, D-3: %s' % (len(dist_unigram_tokens), len(dist_bigram_tokens), len(dist_trigram_tokens)))
    # print('D-1-ratio: %.3f, D-2-ratio: %.3f, D-3-ratio: %.3f' % (len(dist_unigram_tokens)/total_unigram_cnt, len(dist_bigram_tokens)/total_bigram_cnt, len(dist_trigram_tokens)/total_trigram_cnt))
    # print('D-1-ratio: %.3f' % (len(dist_unigram_tokens)/total_unigram_cnt))
    return len(dist_trigram_tokens)/total_trigram_cnt

def eval_length_metrics(predict_text):
    total_unigram_cnt = 0
    for i in range(len(predict_text)):
        pred_sent = predict_text[i]
        # print(pred_sent)
        unigram_tokens = list(pred_sent)
        # print(unigram_tokens)
        total_unigram_cnt += len(unigram_tokens)
    # print('Token Length: %.3f' % (total_unigram_cnt/len(predict_text)))
    return total_unigram_cnt/len(predict_text)


def eval_rouge_metrics(gold_text, predict_text):
    gold_text = [' '.join(jieba.lcut(i)) for i in gold_text]
    predict_text = [' '.join(jieba.lcut(i)) for i in predict_text]
    rouge = Rouge()
    scores = rouge.get_scores(gold_text, predict_text)
    rouge_1_r = 0.0
    rouge_1_p = 0.0
    rouge_1_f = 0.0
    rouge_2_r = 0.0
    rouge_2_p = 0.0
    rouge_2_f = 0.0
    rouge_l_r = 0.0
    rouge_l_p = 0.0
    rouge_l_f = 0.0
    for s in scores:
        rouge_1_r += s['rouge-1']['r']
        rouge_1_p += s['rouge-1']['p']
        rouge_1_f += s['rouge-1']['f']
        rouge_2_r += s['rouge-2']['r']
        rouge_2_p += s['rouge-2']['p']
        rouge_2_f += s['rouge-2']['f']
        rouge_l_r += s['rouge-l']['r']
        rouge_l_p += s['rouge-l']['p']
        rouge_l_f += s['rouge-l']['f']
    rouge_1_r /= len(scores)
    rouge_1_p /= len(scores)
    rouge_1_f /= len(scores)
    rouge_2_r /= len(scores)
    rouge_2_p /= len(scores)
    rouge_2_f /= len(scores)
    rouge_l_r /= len(scores)
    rouge_l_p /= len(scores)
    rouge_l_f /= len(scores)
    # print('Rouge-1 metrics: r:%.3f p:%.3f f:%.3f' % (rouge_1_r, rouge_1_p, rouge_1_f))
    # print('Rouge-2 metrics: r:%.3f p:%.3f f:%.3f' % (rouge_2_r, rouge_2_p, rouge_2_f))
    # print('Rouge-L metrics: r:%.3f p:%.3f f:%.3f' % (rouge_l_r, rouge_l_p, rouge_l_f))
    return rouge_1_r, rouge_1_p, rouge_1_f

def eval_bleu_metrics(gold_text, predict_text):
    gold_text = [jieba.lcut(i) for i in gold_text]
    predict_text = [jieba.lcut(i) for i in predict_text]
    bleu_1_score = 0
    bleu_2_score = 0
    bleu_3_score = 0
    bleu_4_score = 0
    for i in range(len(gold_text)):
        bleu_1_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(1, 0, 0, 0))
        bleu_2_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(0, 1, 0, 0))
        bleu_3_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(0, 0, 1, 0))
        bleu_4_score += sentence_bleu([gold_text[i]], predict_text[i], weights=(0, 0, 0, 1))
#         print(gold_text[i])
#         print(predict_text[i])
    bleu_1_score /= len(gold_text)
    bleu_2_score /= len(gold_text)
    bleu_3_score /= len(gold_text)
    bleu_4_score /= len(gold_text)
    # print('BLEU-1: %.3f, BLEU-2: %.3f, BLEU-3: %.3f, BLEU-4: %.3f' % (bleu_1_score, bleu_2_score, bleu_3_score, bleu_4_score))
    return bleu_1_score

def eval_ppl_metrics(nlls):
    ppl = torch.exp(torch.stack(nlls).means())
    print('PPL: %.3f' % ppl)

def load_features(filename):
    gold_text = []
    predict_text = []
    #nlls = []
    for line in open(filename, "r", encoding="utf-8"):
        line = line.strip()
        predict_data = json.loads(line)
        infer_answer = predict_data["infer_answer"].strip()
        if len(infer_answer) == 0:
            continue
        print("infer_answer")
        print(infer_answer)
        gt_answer = predict_data["messages"][-1]["content"].strip()
        print("gt_answer")
        print(gt_answer)
        gold_text.append(gt_answer)
        predict_text.append(infer_answer)
    print(len(gold_text))
    return gold_text, predict_text

def load_features_with_k(test_file, predict_file, task_name):
    gold_text = {}
    predict_text = {}
    sequence_len = {}
    test_dict = {}
    k_dict = {}
    len_dict = {}
    for line in open(test_file):
        line = line.strip()
        test_data = json.loads(line)
        resource = test_data["resource"]
        if resource != task_name:
            continue
        _id = test_data["id"]
        if _id not in test_dict:
            test_dict.setdefault(_id, test_data["messages"][1]["content"])
            k_dict.setdefault(_id,  len(test_data["messages"][0]["content"].split("<output>:"))-2)
            len_dict.setdefault(_id, len(tokenizer.encode(test_data["messages"][0]["content"])))
    for line in open(predict_file, "r", encoding="utf-8"):
        line = line.strip()
        try:
            predict_data = json.loads(line)
            _id = predict_data["id"]
            # print(_id)
            if _id not in test_dict:
                continue
        except:
            continue
        # k = predict_data["k"]
        # k = len(predict_data["messages"][0]["content"].split("<output>:"))-2
        # total_len = len(tokenizer.encode(predict_data["messages"][0]["content"]))
        k = k_dict[_id]
        total_len = len_dict[_id]
        if k not in sequence_len:
            sequence_len.setdefault(k, total_len)
        else:
            sequence_len[k] += total_len
        # if k > 10 and k < 50:
        #     k = 50
        infer_answer = predict_data["Output"].strip()
        if len(infer_answer) == 0:
            infer_answer = "NULL"
            continue
        # gt_answer = predict_data["messages"][-1]["content"].strip()
        gt_answer = test_dict[_id]
        # gold_text.append(gt_answer)
        # predict_text.append(infer_answer)
        if k not in gold_text:
            gold_text.setdefault(k, [gt_answer])
            predict_text.setdefault(k, [infer_answer])
        else:
            gold_text[k].append(gt_answer)
            predict_text[k].append(infer_answer)
    return gold_text, predict_text, sequence_len

def eval_acc_metrics(gold_text, predict_text):
    # eval w/o k
    correct = sum(1 for a, b in zip(gold_text, predict_text) if a == b)
    acc = correct/len(gold_text)
    print(acc)
    return acc
    
def eval4generation(gold_text, predict_text, k):
    D3 = eval_distinct_metrics(gold_text, predict_text)
    TL = eval_length_metrics(predict_text)
    rouge_1_r, rouge_1_p, rouge_1_f = eval_rouge_metrics(gold_text, predict_text)
    bleu1 = eval_bleu_metrics(gold_text, predict_text)
    return D3, TL, rouge_1_r, rouge_1_p, rouge_1_f, bleu1

def precision_at_k(actual, predicted, k):
    """
    计算准确率（Precision@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - Precision@k
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / k if k != 0 else 0

def recall_at_k(actual, predicted, k):
    """
    计算召回率（Recall@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - Recall@k
    """
    actual_set = set(actual)
    predicted_set = set(predicted[:k])
    intersection = actual_set.intersection(predicted_set)
    return len(intersection) / len(actual_set) if len(actual_set) != 0 else 0

def ndcg_at_k(actual, predicted, k):
    """
    计算归一化折损累积（NDCG@k）

    Parameters:
    - actual: 实际的项目列表
    - predicted: 预测的项目列表
    - k: Top k 推荐

    Returns:
    - NDCG@k
    """
    dcg = sum(1 / (log2(i + 2)) if item in actual else 0 for i, item in enumerate(predicted[:k]))
    idcg = sum(1 / (log2(i + 2)) for i in range(min(k, len(actual))))
    return dcg / idcg if idcg != 0 else 0

def eval_items_ranking(actual_items, predicted_items, k):
    precision_k = precision_at_k(actual_items, predicted_items, k)
    recall_k = recall_at_k(actual_items, predicted_items, k)
    ndcg_k = ndcg_at_k(actual_items, predicted_items, k)
    return precision_k, recall_k, ndcg_k

def eval_reranking(gold_text, predict_text, task_name):
    if task_name == "cMedQAReranking" or task_name == "AskUbuntuDupQuestions":
        precision_k = []
        recall_k = []
        ndcg_k = []
        for i in range(len(predict_text)):
            gold_items = []
            predict_items = []
            try:
                predict_data = json.loads(predict_text[i])
                test_data = json.loads(gold_text[i])
                for key in predict_data:
                    try:
                        predict_items.append(predict_data[key])
                    except:
                        continue
                for key in test_data:
                    try:
                        gold_items.append(test_data[key])
                    except:
                        continue
            except:
                continue
            p, r, ndcg = eval_items_ranking(gold_items, predict_items, 10)
            precision_k.append(p)
            recall_k.append(r)
            ndcg_k.append(ndcg)
        precision_k = sum(precision_k)/len(precision_k)
        recall_k = sum(recall_k)/len(recall_k)
        ndcg_k = sum(ndcg_k)/len(ndcg_k)
        return precision_k, recall_k, ndcg_k

def eval_reasoning(gold_text, predict_text, task_name):
    if task_name == "GSM8K":
        acc = eval_gsm8k(gold_text, predict_text)
        return acc
    if task_name == "AR-LSAT":
        acc = eval_mmlu(gold_text, predict_text)
        return acc

def eval_retrieval(gold_text, predict_text, task_name):
    if task_name == "EcomRetrieval" or task_name == "VideoRetrieval":
        acc = eval_retrieval(gold_text, predict_text)
        return acc

def eval_clustering(gold_text, predict_text, task_name):
    if task_name == "CLSClusteringP2P" or task_name == "CLSClusteringS2S" or task_name == "TenkgnadClusteringP2P" \
        or task_name == "TenkgnadClusteringS2S" or task_name == "ArxivClusteringS2S":
        acc = eval_clustering(gold_text, predict_text)
        return acc

def eval_qa(gold_text, predict_text, task_name):
    # eval_acc_metrics(gold_text, predict_text)
    if task_name == "BoolQ":
        acc = eval_boolq(gold_text, predict_text)
        return acc
    if task_name == "TruthfulQA":
        acc = eval_truthfulqa(gold_text, predict_text)
        return acc
    if task_name == "MMLU" or task_name == "OpenbookQA" or task_name == "ARC":
        acc = eval_mmlu(gold_text, predict_text)
        return acc

def eval_classification(gold_text, predict_text, task_name):
    if task_name == "ToxicConversations" or task_name == "TweetSentimentExtraction":
        acc = eval_sentiment(gold_text, predict_text)
        return acc
    if task_name == "WinoWhy":
        acc = eval_winowhy(gold_text, predict_text)
        return acc
    
    
    

def eval_mmlu(gold_text, predict_text):
    correct = 0
    pattern = r"(option \(A\))|(option \(B\))|(option \(C\))|(option \(D\))|(option \(E\))|(is \(A\))|(is \(B\))|(is \(C\))|(is \(D\))|(is \(E\))|(option A)|(option B)|(option C)|(option D)|(is A)|(is B)|(is C)|(is D)|(\(A\))|(\(B\))|(\(C\))|(\(D\))|(\(E\))|(choice A)|(choice B)|(choice C)|(choice D)|(choice E)|(Answer: A)|(Answer: B)|(Answer: C)|(Answer: D)|(Answer: E)|(^A.)|(^B.)|(^C.)|(^D.)|(^A.)|(^B.)|(^C.)|(^D.)|(^E.)|(A)|(B)|(C)|(D)|(E)"
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        match = re.search(pattern, predict_text[i])
        match = match.group(0) if match is not None else None
        # print(match)
        match = re.search(r"[A-E]", match) if match is not None else None
        final_answer = match.group(0) if match is not None else None
        if gt_answer == final_answer:
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_clustering(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(Sport)|(International)|(Web)|(Kultur)|(Panorama)|(Etat)|(Wissenschaft)|(Inland)|(Wirtschaft)|(工学)|(理学)|(农学)|(哲学)|(艺术学)|(历史学)|(管理学)|(教育学)|(军事学)|(法学)|(经济学)|(文学)|(医学)|(\n(\d+))|(\d+)"
        match = re.search(pattern, predict_text[i])
        final_answer = match.group(0) if match is not None else predict_text[i]
        final_answer = final_answer.strip()
        if final_answer is not None and (gt_answer.lower() == final_answer.lower()):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_retrieval(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        if predict_text[i].find(gt_answer) != -1 or gt_answer.find(predict_text[i]) != -1:
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_imdb(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(Positive)|(Negative)|(positive)|(negative)"
        match = re.search(pattern, predict_text[i])
        final_answer = match.group(0) if match is not None else None
        if (gt_answer.lower() == final_answer.lower()):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_sts(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(Positive)|(Negative)|(positive)|(negative)"
        match = re.search(pattern, predict_text[i])
        final_answer = match.group(0) if match is not None else None
        if (gt_answer.lower() == final_answer.lower()):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_winowhy(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(incorrect)|(correct)|(yes)|(YES)|(Yes)|(no)|(NO)|(No)|(True)|(False)|(TRUE)|(true)|(FALSE)|(false)"
        match = re.search(pattern, predict_text[i])
        final_answer = match.group(0) if match is not None else None
        if final_answer is None:
            continue
        if (gt_answer.lower() == final_answer.lower()) or (gt_answer.lower() == "correct" and final_answer.lower() in ["yes", "true", "correct"]) \
            or (gt_answer.lower() == "incorrect" and final_answer.lower() in ["no", "false", "incorrect"]):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_boolq(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(True)|(False)|(Yes)|(No)"
        match = re.search(pattern, predict_text[i])
        final_answer = match.group(0) if match is not None else None
        if (gt_answer == final_answer) or (gt_answer == "True" and final_answer == "Yes") or (gt_answer == "False" and final_answer == "No"):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_truthfulqa(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(yes)|(no)|(Yes)|(No)"
        match = re.search(pattern, predict_text[i])
        final_answer = match.group(0) if match is not None else None
        if final_answer is None:
            continue
        gt_answer = gt_answer.lower()
        final_answer = final_answer.lower()
        if (gt_answer == final_answer):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc

def eval_sentiment(gold_text, predict_text):
    correct = 0
    for i in range(len(gold_text)):
        gt_answer = gold_text[i]
        pattern = r"(0)|(1)|(2)"
        matches = re.findall(pattern, predict_text[i])
        matches = [match for match in matches if any(match)]
        if len(matches) > 0:
            final_answer = next(filter(None, matches[-1]))
        else:
            continue
        if (gt_answer == final_answer):
            correct += 1
        # print("gt: %s predict: %s" % (gt_answer, final_answer))
    acc = correct/len(gold_text)
    return acc



def eval_gsm8k(gold_text, predict_text):
    correct = 0
    gt_pattern = r"#### ([+-]?([0-9]*[.,])?[0-9]+)"
    for i in range(len(gold_text)):
        m = re.findall(gt_pattern, gold_text[i])
        gt_number = None
        try:
            gt_number = float(m[0][0].replace(",", ""))
        except:
            # print(gold_text[i])
            continue
        # print("gt: %s gt_number=%s" % (gold_text[i], gt_number))
        new_pattern = r"([+-]?)>>(\d+)([+-]?)"
        new_pattern_1 = r"\$([+-]?([0-9]*[.,])?[0-9]+)"
        new_pattern_2 = r"([+-]?([0-9]*[.,])?[0-9]+)"
        predict_text[i] = predict_text[i].replace("\n", "\\n")
        m = re.findall(new_pattern, predict_text[i])
        predict_number = None
        try:
            m = re.findall(gt_pattern, predict_text[i])
            if len(m) > 0:
                predict_number = float(m[0][0].replace(",", ""))
                # predict_number = float(m[-1][0].replace(",", ""))
            else:
                m = re.findall(new_pattern, predict_text[i])
                if len(m) > 0:
                    predict_number = float(m[-1][1].replace(",", ""))
                    # predict_number = float(m[0][0].replace(",", ""))
                else:
                    m = re.findall(new_pattern_1, predict_text[i])
                    if len(m) > 0:
                        predict_number = float(m[-1][0].replace(",", ""))
                    else:
                        m = re.findall(new_pattern_2, predict_text[i])
                        predict_number = float(m[-1][0].replace(",", ""))
        except:
            # print(predict_text[i])
            continue
        # print("predict: %s predict_number=%s" % (predict_text[i], predict_number))
        if gt_number != None and predict_number != None and gt_number == predict_number:
            correct += 1
    return correct/len(gold_text)
        

if __name__ == '__main__':
    test_file = sys.argv[1]
    predict_file = sys.argv[2]
    # 评测任务：generation/classification/clustering/qa/reranking/retrieval/reasoning
    evaluation_type = sys.argv[3]
    task_name = sys.argv[4]
    gold_dict, predict_dict, sequence_len = load_features_with_k(test_file, predict_file, task_name)
    for k in gold_dict:
        if k in [0, 1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300]:
            gold_text = gold_dict[k]
            predict_text = predict_dict[k]
            if evaluation_type == "generation":
                D3, TL, rouge_1_r, rouge_1_p, rouge_1_f, bleu1 = eval4generation(gold_text, predict_text, k)
                print("k=%s sample_count=%s D-3-ratio=%s TL=%s rouge_1_r=%s rouge_1_p=%s rouge_1_f=%s bleu-1-score=%s avg_len=%s" % (k, len(predict_text), D3, TL, rouge_1_r, rouge_1_p, rouge_1_f, bleu1, sequence_len[k]/len(predict_text)))
            elif evaluation_type == "classification":
                acc = eval_classification(gold_text, predict_text, task_name)
                print("k=%s sample_count=%s acc=%s avg_len=%s" % (k, len(predict_text), acc, sequence_len[k]/len(predict_text)))
            elif evaluation_type == "qa":
                acc = eval_qa(gold_text, predict_text, task_name)
                print("k=%s sample_count=%s acc=%s avg_len=%s" % (k, len(predict_text), acc, sequence_len[k]/len(predict_text)))
            elif evaluation_type == "retrieval":
                acc = eval_retrieval(gold_text, predict_text, task_name)
                print("k=%s sample_count=%s acc=%s avg_len=%s" % (k, len(predict_text), acc, sequence_len[k]/len(predict_text)))
            elif evaluation_type == "reasoning":
                acc = eval_reasoning(gold_text, predict_text, task_name)
                print("k=%s sample_count=%s acc=%s avg_len=%s" % (k, len(predict_text), acc, sequence_len[k]/len(predict_text)))
            elif evaluation_type == "clustering":
                acc = eval_clustering(gold_text, predict_text, task_name)
                print("k=%s sample_count=%s acc=%s avg_len=%s" % (k, len(predict_text), acc, sequence_len[k]/len(predict_text)))
            elif evaluation_type == "reranking":
                precision_k, recall_k, ndcg_k = eval_reranking(gold_text, predict_text, task_name)
                print("k=%s sample_count=%s p@10=%s r@10=%s ngcg@10=%s avg_len=%s" % (k, len(predict_text), precision_k, recall_k, ndcg_k, sequence_len[k]/len(predict_text)))