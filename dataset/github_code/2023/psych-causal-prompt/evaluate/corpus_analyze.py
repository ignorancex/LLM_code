from postprocess_GPT3 import KL, find_predict_label, eval_and_offset, Entropy, cross_Entropy
from warping_dataset_like_GPT2 import GPT3TokenizerWarper
import numpy as np
from icecream import ic
from tqdm import trange
from scipy.stats import kurtosis, skew
from collections import Counter

def load_whole_dataset():
    from datasets import load_from_disk
    dataset = load_from_disk('./yelp_large_test_new')
    gt_labels = []
    for i in dataset:
        gt_labels.append(i['label'])
    predict_labels1, predict_prob1 = find_predict_label('gpt3_result_setup_1_short_tiny_new.npy', gt_labels, file_name2 = 'gpt3_result_setup_1_short_small_new.npy'
                                                        , file_name3 = 'gpt3_result_setup_1_short_large.npy'
                                                        , offset_filename = ['gpt3_result_setup_1_short_small_train.npy', './yelp_small_train_new'])
    #ic(predict_labels1[0:10])
    #ic(predict_prob1[0:10])
    predict_labels2, predict_prob2 = find_predict_label('gpt3_result_setup_2_short_tiny_new.npy', gt_labels, file_name2 = 'gpt3_result_setup_2_short_small_new.npy'
                                                        , file_name3 = 'gpt3_result_setup_2_short_large.npy'
                                                        , offset_filename = ['gpt3_result_setup_2_short_small_train.npy', './yelp_small_train_new'])
    predict_labels3, predict_prob3 = find_predict_label('gpt3_result_setup_3_short_tiny_new.npy', gt_labels, file_name2 = 'gpt3_result_setup_3_short_small_new.npy'
                                                        , file_name3 = 'gpt3_result_setup_3_short_large.npy'
                                                        , offset_filename = ['gpt3_result_setup_3_short_small_train.npy', './yelp_small_train_new'])
    ic(len(predict_labels1), len(predict_prob1))

    predict_label = np.array([predict_labels1, predict_labels2, predict_labels3])
    predict_prob = np.array([predict_prob1, predict_prob2, predict_prob3])
    """
    x, y = eval_and_offset((predict_prob[0]+predict_prob[1])/2, gt_labels)
    KL_list = []
    for j in y:
        KL_list.append(Entropy(j))
    ic(np.average(KL_list), np.var(KL_list), skew(KL_list), kurtosis(KL_list))
    """
    return dataset, predict_label, predict_prob

def calculate_max_KL(probs):
    max_KL = -1e9
    for i in probs:
        for j in probs:
            max_KL = max(max_KL, KL(i,j))
    return max_KL

# the format of dataset is dataframe
def extract_subset_byLength(dataset, predict_label, predict_prob):
    random_tokenizer = GPT3TokenizerWarper(["", ""])
    low_length, mid_length, high_length = [], [], []
    for i in trange(len(dataset)):
        length = random_tokenizer.get_length(dataset[i]['text'])
        if (length<50):
            low_length.append(i)
        elif (length<200):
            mid_length.append(i)
        else:
            high_length.append(i)
    ic(len(low_length), len(mid_length), len(high_length))
    return low_length, mid_length, high_length


def extract_subset_byLabel(dataset, predict_label, predict_prob):
    neg, neutral, pos = [], [], []

    for i in trange(len(dataset)):
        if (dataset[i]['label']==0):
            neg.append(i)
        elif (dataset[i]['label']==4):
            pos.append(i)
        else:
            neutral.append(i)

    ic(len(neg), len(neutral), len(pos))
    return neg, neutral, pos

def extract_subset_byCorpus(dataset, predict_label, predict_prob):
    corpus1, corpus2 = [], []

    for i in trange(len(dataset)):
        max_KL = calculate_max_KL(predict_prob[:, i])
        if (max_KL>1):
            corpus1.append(i)
        elif (max_KL<0.07):
            corpus2.append(i)
    ic(len(corpus1), len(corpus2))
    return corpus1, corpus2


def extract_subset_byPred(dataset, predict_label, predict_prob):
    allTrue, allFalse, disAgree = [], [], []
    for i in trange(len(dataset)):
        if (predict_label[0][i]!=predict_label[1][i] or predict_label[2][i]!=predict_label[1][i]):
            disAgree.append(i)
        elif (predict_label[0][i]==dataset[i]['label']):
            allTrue.append(i)
        else:
            allFalse.append(i)
    ic(len(allTrue),len(allFalse),len(disAgree))
    return allTrue, allFalse, disAgree



from scipy.special import softmax
def L1_norm(log_px, log_py):
    px = softmax(log_px)
    py = softmax(log_py)
    return np.sum(np.abs(py-px))

# L1 norm
def extract_unagreement_set(dataset, predict_label, predict_prob, subset):
    print(f"the length of subset is {len(subset)}, we extract top and low 10%, which is {len(subset)//10}")
    candidate_data = []
    for i in subset:
        agreement_score = max(max(L1_norm(predict_prob[0][i], predict_prob[1][i]), L1_norm(predict_prob[0][i], predict_prob[2][i]))
                              ,L1_norm(predict_prob[1][i], predict_prob[2][i]))
        candidate_data.append([agreement_score, i])
    candidate_data = sorted(candidate_data)
    length_selected = len(subset)//10
    strong_agreement, strong_disagreement = candidate_data[:length_selected], candidate_data[-length_selected:]
    strong_agreement = [i[1] for i in strong_agreement]
    strong_disagreement = [i[1] for i in strong_disagreement]
    return strong_agreement, strong_disagreement


# Extract the subset those is have a error
def extract_error_set(dataset, predict_label, predict_prob, subset, setup):
    new_subset = []
    for i in subset:
        if (predict_label[setup-1][i]!=dataset[i]['label']):
            new_subset.append(i)
    #ic(len(new_subset))
    return new_subset


def evaluate_subset_setupwise(dataset, predict_label, predict_prob, subset_idx):
    # TODO: Accuracy, average length, distribution of different label, distribution of some words
    from sklearn.metrics import classification_report
    random_tokenizer = GPT3TokenizerWarper(["", ""])
    gt_label_subset, predict_label_subset, prob_subset, text_subset = [], [], [], []
    sum_length = 0
    for i in subset_idx:
        gt_label_subset.append(dataset[i]['label'])
        predict_label_subset.append(predict_label[i])
        prob_subset.append(predict_prob[i])
        text_subset.append(dataset[i]['text'])
        length = random_tokenizer.get_length(dataset[i]['text'])
        sum_length += length
    print(classification_report(gt_label_subset, predict_label_subset, digits=4))
    ic(sum_length/len(subset_idx))
    ic(Counter(gt_label_subset))

def evaluate_subset(dataset, predict_label, predict_prob, subset_idx):
    from extra_corpus_analyze import load_word_list, count_word_list
    pos_words = load_word_list('pos_words.txt')
    neg_words = load_word_list('neg_words.txt')

    ic(len(subset_idx))
    gt_label_subset = []
    agreement_score_list = []
    random_tokenizer = GPT3TokenizerWarper(["", ""])
    length_list = []
    pos_word_list = []
    neg_word_list = []
    scarcasm_list = []
    #ic(subset_idx)
    for i in subset_idx:
        gt_label_subset.append(dataset[i]['label'])
        agreement_score = max(max(L1_norm(predict_prob[0][i], predict_prob[1][i]), L1_norm(predict_prob[0][i], predict_prob[2][i])),L1_norm(predict_prob[1][i], predict_prob[2][i]))/2
        agreement_score_list.append(agreement_score)
        length_list.append(random_tokenizer.get_length(dataset[i]['text']))
        pos_word_list.append(count_word_list(pos_words, dataset[i]['text']))
        neg_word_list.append(count_word_list(neg_words, dataset[i]['text']))

    print("########################################")
    ic(Counter(gt_label_subset))
    ic(np.average(agreement_score_list))
    ic(np.average(length_list))
    ic(np.average(pos_word_list))
    ic(np.var(pos_word_list))
    ic(np.average(neg_word_list))
    ic(np.var(neg_word_list))
    ic(np.average(np.abs(np.array(pos_word_list)/6.325-np.array(neg_word_list)/3.1165)))
    ic(np.var((np.array(pos_word_list)/6.325-np.array(neg_word_list)/3.1165)))
    ic(np.average((np.array(pos_word_list)/6.325+np.array(neg_word_list)/3.1165)))
    ic(np.var((np.array(pos_word_list)/6.325+np.array(neg_word_list)/3.1165)))
    ic(np.average(np.abs(np.array(pos_word_list)-np.array(neg_word_list))))
    print("########################################")

def setupwise_Entropy(predict_prob, dataset):
    print("The information Entropy")
    for i in range(3):
        Entropy_list = []
        for j in predict_prob[i]:
            Entropy_list.append(Entropy(j))
        ic(np.average(Entropy_list), np.var(Entropy_list), skew(Entropy_list), kurtosis(Entropy_list))
        #print(Entropy_list)
    return
    print("Now the Cross Entropy")
    for i in range(3):
        Entropy_list = []
        for j in range(len(predict_prob[i])):
            Entropy_list.append(cross_Entropy(predict_prob[i][j], dataset[j]['label']))
        ic(np.average(Entropy_list), np.var(Entropy_list), skew(Entropy_list), kurtosis(Entropy_list))
        print(Entropy_list)
    """
    Entropy_list = []
    for j in range(len(predict_prob[0])):
        Entropy_list.append(Entropy((predict_prob[0][j]+predict_prob[1][j])/2))
    ic(np.average(Entropy_list), np.var(Entropy_list), skew(Entropy_list), kurtosis(Entropy_list))
    """

def extract_agreement_subset(dataset, predict_label, predict_prob, output_path):
    candidate_data = []
    for i in range(len(dataset)):
        agreement_score = max(
            max(L1_norm(predict_prob[0][i], predict_prob[1][i]), L1_norm(predict_prob[0][i], predict_prob[2][i]))
            , L1_norm(predict_prob[1][i], predict_prob[2][i]))
        if (agreement_score<0.03 and len(dataset[i]['text'])<300):
            print(agreement_score, dataset[i])
            ic(predict_prob[0][i], predict_prob[1][i], predict_prob[2][i])
            ic(predict_label[0][i], predict_label[1][i], predict_label[2][i])
        candidate_data.append([agreement_score, i])
    candidate_data = sorted(candidate_data)

    mid_point = len(dataset)//2
    data_dict = {
        "text":[],
        "label":[],
        "score":[],
        "setup1_pred":[],
        "setup2_pred":[],
        "setup3_pred":[],
        "idx":[],
    }
    for i in range(len(dataset)):
        data_dict['text'].append(dataset[candidate_data[i][1]]['text'])
        data_dict['label'].append(dataset[candidate_data[i][1]]['label'])
        data_dict['score'].append(candidate_data[i][0])
        #data_dict['score_group'].append(i//mid_point)
        data_dict['setup1_pred'].append(predict_label[0][candidate_data[i][1]])
        data_dict['setup2_pred'].append(predict_label[1][candidate_data[i][1]])
        data_dict['setup3_pred'].append(predict_label[2][candidate_data[i][1]])
        data_dict['idx'].append(candidate_data[i][1])
    import pandas as pd
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(output_path)

def find_disagree_examples(dataset, predict_label, predict_prob):

    random_tokenizer = GPT3TokenizerWarper(["", ""])
    for i in range(len(dataset)):
        length = random_tokenizer.get_length(dataset[i]['text'])
        #if (predict_label[0][i]==predict_label[1][i] and predict_label[2][i]==predict_label[1][i]):
        if True:
            if (length<100):
                agreement_score = max(
                    max(L1_norm(predict_prob[0][i], predict_prob[1][i]),
                        L1_norm(predict_prob[0][i], predict_prob[2][i]))
                    , L1_norm(predict_prob[1][i], predict_prob[2][i]))
                Entropy1 = Entropy(predict_prob[0][i])
                Entropy2 = Entropy(predict_prob[1][i])
                Entropy3 = Entropy(predict_prob[2][i])
                if (Entropy3<0.2 and Entropy1>0.5):
                    print(dataset[i])
                    ic(Entropy1, Entropy2, Entropy3)
                    ic(agreement_score, softmax(predict_prob[0][i]), softmax(predict_prob[1][i]), softmax(predict_prob[2][i]))
                    ic(predict_label[0][i], predict_label[1][i],predict_label[2][i])

def extract_distribution(dataset, predict_label, predict_prob, output_path):
    from extra_corpus_analyze import load_word_list, count_word_list
    pos_words = load_word_list('pos_words.txt')
    neg_words = load_word_list('neg_words.txt')
    result_list = []
    for i in range(len(dataset)):
        pos_word_num = count_word_list(pos_words, dataset[i]['text'])
        neg_word_num = count_word_list(neg_words, dataset[i]['text'])
        agreement_score = max(max(L1_norm(predict_prob[0][i], predict_prob[1][i]), L1_norm(predict_prob[0][i], predict_prob[2][i])),L1_norm(predict_prob[1][i], predict_prob[2][i]))
        Setup1_Entropy = Entropy(predict_prob[0][i])
        Setup2_Entropy = Entropy(predict_prob[1][i])
        Setup3_Entropy = Entropy(predict_prob[2][i])
        result_list.append((pos_word_num, neg_word_num, agreement_score, Setup1_Entropy, Setup2_Entropy, Setup3_Entropy))
    a = np.array(result_list)
    np.save(output_path, a)


if __name__=='__main__':
    dataset, predict_label, predict_prob = load_whole_dataset()
    extract_agreement_subset(dataset, predict_label, predict_prob, "subset_agreement.csv")
    #find_disagree_examples(dataset, predict_label, predict_prob)

    """
    setupwise_Entropy(predict_prob, dataset)
    #extract_distribution(dataset, predict_label, predict_prob, "distribution.npy")

    allTrue, allFalse, disAgree = extract_subset_byPred(dataset, predict_label, predict_prob)
    #for i in range(3):
    evaluate_subset(dataset, predict_label, predict_prob, list(range(len(dataset))))

    evaluate_subset(dataset, predict_label, predict_prob, allTrue)
    evaluate_subset(dataset, predict_label, predict_prob, allFalse)
    evaluate_subset(dataset, predict_label, predict_prob, disAgree)


    agree_allTrue, disAgree_allTrue = extract_unagreement_set(dataset, predict_label, predict_prob, allTrue)

    print('agree alltrue')
    evaluate_subset(dataset, predict_label, predict_prob, agree_allTrue)
    print('disagree alltrue')
    evaluate_subset(dataset, predict_label, predict_prob, disAgree_allTrue)


    agree_allFalse, disAgree_allFalse = extract_unagreement_set(dataset, predict_label, predict_prob, allFalse)

    print('agree allerror')
    evaluate_subset(dataset, predict_label, predict_prob, agree_allFalse)

    print('disagree allerror')
    evaluate_subset(dataset, predict_label, predict_prob, disAgree_allFalse)

    setup1_subset = extract_error_set(dataset, predict_label, predict_prob, disAgree, 1)
    setup2_subset = extract_error_set(dataset, predict_label, predict_prob, disAgree, 2)
    setup3_subset = extract_error_set(dataset, predict_label, predict_prob, disAgree, 3)
    print('setup1 error')
    evaluate_subset(dataset, predict_label, predict_prob, setup1_subset)
    print('setup2 error')
    evaluate_subset(dataset, predict_label, predict_prob, setup2_subset)
    print('setup3 error')
    evaluate_subset(dataset, predict_label, predict_prob, setup3_subset)
    np.random.seed(41)
    a = list(range(len(dataset)))
    np.random.shuffle(a)
    print('random')
    evaluate_subset(dataset, predict_label, predict_prob, a[:500])


    """
    """
    low_length, mid_length, high_length = extract_subset_byLength(dataset, predict_label, predict_prob)
    for i in range(3):
        evaluate_subset_setupwise(dataset, predict_label[i], predict_prob[i], low_length)
        evaluate_subset_setupwise(dataset, predict_label[i], predict_prob[i], mid_length)
        evaluate_subset_setupwise(dataset, predict_label[i], predict_prob[i], high_length)


    neg, neutral, pos = extract_subset_byLabel(dataset, predict_label, predict_prob)

    for i in range(3):
        evaluate_subset_setupwise(dataset, predict_label[i], predict_prob[i], neg)
        evaluate_subset_setupwise(dataset, predict_label[i], predict_prob[i], neutral)
        evaluate_subset_setupwise(dataset, predict_label[i], predict_prob[i], pos)
    """

    """
    corpus1, corpus2 = extract_subset_byCorpus(dataset, predict_label, predict_prob)
    for i in range(3):
        #continue
        evaluate_subset(dataset, predict_label[i], predict_prob[i], corpus1)
        evaluate_subset(dataset, predict_label[i], predict_prob[i], corpus2)
        break
        pass
    """
# 19 29 30 15 17
# 16 14 23 15 11