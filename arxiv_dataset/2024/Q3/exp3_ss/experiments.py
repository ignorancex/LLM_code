from data import get_data, get_data_source
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging
from pathlib import Path
from recommendation import *
from fasttxt import load_ft_model, get_query_embeddings
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from math import exp

def run_bandit_arms(dt,setting):
    import transformers
    transformers.logging.set_verbosity_error()
    n_rounds = 500
    candidate_ix = [2, 3, 5, 10]
    
    df, X, anchor_ids, noof_anchors = get_data(dt)
    ft = load_ft_model()
    bandit = 'EXP3-SS'
    src = get_data_source(dt)
    regret = {}
    avg_sim = {}
    avg_dst = {}
    model_dict = {}
    if (setting == 'scratch'):
        from transformers import BertTokenizer
        vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
        tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        sep = ' [sep] '
        special_tokens = ['[sep]', '[bos]']
        model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
        from transformers import GPT2LMHeadModel
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = tokenizer
        model_dict['GPT']['spl_tok'] = special_tokens
        model_dict['GPT']['sep'] = sep

        model_dest = '../Data/semanticscholar/model/ctrl'
        from transformers import CTRLLMHeadModel
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = tokenizer
        model_dict['CTRL']['spl_tok'] = special_tokens
        model_dict['CTRL']['sep'] = sep
    else:
        from transformers import (TransfoXLLMHeadModel,TransfoXLTokenizer)
        model_dest = '../Data/semanticscholar/model/xl'
        model_dict['XL'] = {}
        model_dict['XL']['model'] = TransfoXLLMHeadModel.from_pretrained(model_dest)
        model_dict['XL']['tok'] = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model_dict['XL']['spl_tok'] = ['[<unk>]', '<unk>', '[bos]']
        model_dict['XL']['sep'] = ' [sep] '

        from transformers import (GPT2LMHeadModel,GPT2Tokenizer)
        model_dest = '../Data/semanticscholar/model/gpt2/pretrained'
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model_dict['GPT']['spl_tok'] = ['[sep]','<|endoftext|>']
        model_dict['GPT']['sep'] = ' [sep] '

        from transformers import (CTRLLMHeadModel, CTRLTokenizer)
        model_dest = '../Data/semanticscholar/model/ctrl/pretrained'
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = CTRLTokenizer.from_pretrained('sshleifer/tiny-ctrl')
        model_dict['CTRL']['spl_tok'] = ['[sep]']
        model_dict['CTRL']['sep'] = ' [sep] '
        
    for cand_sz in candidate_ix:
        regret[cand_sz] = {}
        avg_sim[cand_sz] = {}
        avg_dst[cand_sz] = {}
        log_file = Path('../Data/', src, 'logs', src+'_%s_%s_%d.log' %(setting,bandit,cand_sz))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm with candidate size %d" %(bandit, cand_sz))
        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            seq_err, avg_sim[cand_sz][anchor], avg_dst[cand_sz][anchor] = policy_evaluation(bandit, setting, model_dict, X, true_ids, n_rounds, cand_sz, ft)
            regret[cand_sz][anchor] = regret_calculation(seq_err)
            logging.info("finished with regret calculation")
        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

        simv = sum([sum(x)/noof_anchors for x in zip(*avg_sim[cand_sz].values())])/n_rounds
        dstv = sum([sum(x)/noof_anchors for x in zip(*avg_dst[cand_sz].values())])/n_rounds
        #print(sum(zip(*regret[bandit].values())))
        print("average similarity of %d is: %f" %(cand_sz, simv))
        print("average distance of %d is: %f" %(cand_sz, dstv))

    import matplotlib.pyplot as plt
    from matplotlib import rc
    f = plt.figure()
    f.clear()
    plt.clf()
    plt.close(f)
    with plt.style.context(("seaborn-darkgrid",)):
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col = {2:'b', 3:'r', 5:'k', 10:'c'}
        regret_file = '%s_cand_cum_regret.txt' %(setting)
        with open(regret_file, "w") as regret_fd:
            for cand_sz in candidate_ix:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[cand_sz].values())]
                val = str(cand_sz)+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[cand_sz], ls='-', label=r'$k = {}$'.format(cand_sz))
                ax.set_xlabel(r'k')
                ax.set_ylabel(r'cumulative regret')
                ax.legend()
            fig.savefig('arm_regret_%s.pdf' %(setting),format='pdf')
            f = plt.figure()
            f.clear()
            plt.close(f)


def run_bandit_round(dt,setting):
    from random import Random
    import transformers
    transformers.logging.set_verbosity_error()
    rnd = Random()
    rnd.seed(44)

    n_rounds = 500
    #n_rounds = 5
    cand_set_sz = 3
    experiment_bandit = list() 
    df, X, anchor_ids, noof_anchors = get_data(dt)
    model_dict = {}
    if setting == 'pretrained':
        experiment_bandit = ['EXP3-SS', 'EXP3', 'XL', 'GPT', 'CTRL']
        from transformers import (TransfoXLLMHeadModel,TransfoXLTokenizer)
        model_dest = '../Data/semanticscholar/model/xl'
        model_dict['XL'] = {}
        model_dict['XL']['model'] = TransfoXLLMHeadModel.from_pretrained(model_dest)
        model_dict['XL']['tok'] = TransfoXLTokenizer.from_pretrained('transfo-xl-wt103')
        model_dict['XL']['spl_tok'] = ['[<unk>]', '<unk>', '[bos]']
        model_dict['XL']['sep'] = ' [sep] '

        from transformers import (GPT2LMHeadModel,GPT2Tokenizer)
        model_dest = '../Data/semanticscholar/model/gpt2/pretrained'
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = GPT2Tokenizer.from_pretrained('gpt2-medium')
        model_dict['GPT']['spl_tok'] = ['[sep]','<|endoftext|>']
        model_dict['GPT']['sep'] = ' [sep] '

        from transformers import (CTRLLMHeadModel, CTRLTokenizer)
        model_dest = '../Data/semanticscholar/model/ctrl/pretrained'
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = CTRLTokenizer.from_pretrained('sshleifer/tiny-ctrl')
        model_dict['CTRL']['spl_tok'] = ['[sep]']
        model_dict['CTRL']['sep'] = ' [sep] '
    else:
        experiment_bandit = ['EXP3-SS', 'EXP3', 'GPT', 'CTRL']
        #model_dict['EXP3'] = 'all'
        from transformers import BertTokenizer
        vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
        tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        sep = ' [sep] '
        special_tokens = ['[sep]', '[bos]']

        model_dest = '../Data/semanticscholar/model/gpt2/wordpiece'
        from transformers import GPT2LMHeadModel
        model_dict['GPT'] = {}
        model_dict['GPT']['model'] = GPT2LMHeadModel.from_pretrained(model_dest)
        model_dict['GPT']['tok'] = tokenizer
        model_dict['GPT']['spl_tok'] = special_tokens
        model_dict['GPT']['sep'] = sep

        model_dest = '../Data/semanticscholar/model/ctrl'
        from transformers import CTRLLMHeadModel
        model_dict['CTRL'] = {}
        model_dict['CTRL']['model'] = CTRLLMHeadModel.from_pretrained(model_dest)
        model_dict['CTRL']['tok'] = tokenizer
        model_dict['CTRL']['spl_tok'] = special_tokens
        model_dict['CTRL']['sep'] = sep

    ft = load_ft_model()
    regret = {}
    avg_sim = {}
    avg_dst = {}
    src = get_data_source(dt)
    #for bandit in model_dict.keys(): 
    for bandit in experiment_bandit:
        log_file = Path('../Data/', src, 'logs',src+'_%s_%s.log' %(setting, bandit))
        logging.basicConfig(filename = log_file, format='%(asctime)s : %(message)s', level=logging.INFO)
        logging.info("Running %s algorithm trained from %s" %(bandit,setting))
        regret[bandit] = {}
        avg_sim[bandit] = {}
        avg_dst[bandit] = {}

        for anchor in anchor_ids:
            anchor_session_id = df.iloc[anchor]['session_id']
            true_ids = df.index[df['session_id'] == anchor_session_id].tolist()
            true_ids.sort() #just in case if
            seq_err, avg_sim[bandit][anchor], avg_dst[bandit][anchor] = policy_evaluation(bandit, setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)
            regret[bandit][anchor] = regret_calculation(seq_err)

        logger = logging.getLogger()
        for hdlr in logger.handlers[:]:
            hdlr.close()
            logger.removeHandler(hdlr)

        simv = sum([sum(x)/noof_anchors for x in zip(*avg_sim[bandit].values())])/n_rounds
        dstv = sum([sum(x)/noof_anchors for x in zip(*avg_dst[bandit].values())])/n_rounds
        #print(sum(zip(*regret[bandit].values())))
        print("average similarity of %s is: %f" %(bandit, simv))
        print("average distance of %s is: %f" %(bandit, dstv))
    
    import matplotlib.pyplot as plt
    from matplotlib import rc
    with plt.style.context(("seaborn-darkgrid",)):
        f = plt.figure()
        f.clear()
        plt.clf()
        plt.close(f)
        fig, ax = plt.subplots(frameon=False)
        rc('mathtext',default='regular')
        rc('text', usetex=True)
        col_list = ['b', 'r', 'k', 'c', 'm', 'g']
        #col = {experiment_bandit[i]:col_list[i] for i in range(len(experiment_bandit))}
        col = {'EXP3-SS':'b','EXP3':'r', 'GPT':'c', 'CTRL':'r', 'XL':'m'}
        sty = {'EXP3-SS':'b', 'EXP3':'-', 'GPT':':', 'CTRL':'--', 'XL':'-.'}
        labels = {'EXP3-SS':'EXP3-SS', 'EXP':'EXP', 'GPT':'GPT', 'CTRL':'CTRL', 'XL':'XL'}
        regret_file = '%s_cum_regret.txt' %(setting)
        with open(regret_file, "w") as regret_fd:
            for bandit in experiment_bandit:
                cum_regret = [sum(x)/noof_anchors for x in zip(*regret[bandit].values())]
                val = bandit+','+','.join([str(e) for e in cum_regret])
                print(val, file=regret_fd)
                ax.plot(range(n_rounds), cum_regret, c=col[bandit], ls=sty[bandit], label=labels[bandit])
                ax.set_xlabel('rounds')
                ax.set_ylabel('cumulative regret')
                ax.legend()
        fig.savefig('round_regret_%s.pdf' %(setting),format='pdf')
        f = plt.figure()
        f.clear()
        plt.close(f)
        
def run_xl(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd = Random()
    rnd.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))

    for t in range(n_rounds):
        curr_id = rnd.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        #next_query = get_next_query('XL', setting, curr_query)
        next_query = get_next_query(curr_query, model_dict, setting)
        score = get_recommendation_score(ground_queries, next_query)
        if score >= 0.5:
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        simv[t] = get_similarity(ft, curr_query, next_query)
        dstv[t] = get_distance(ft, curr_query, next_query)

    return seq_error, simv, dstv

def run_ctrl(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd = Random()
    rnd.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))

    if setting == 'pretrained':
        return seq_error, simv, dstv

    for t in range(n_rounds):
        curr_id = rnd.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        #next_query = get_next_query('CTRL', setting, curr_query)
        next_query = get_next_query(curr_query, model_dict, setting)
        score = get_recommendation_score(ground_queries, next_query)
        if score >= 0.5:
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        simv[t] = get_similarity(ft, curr_query, next_query)
        dstv[t] = get_distance(ft, curr_query, next_query)

    return seq_error, simv, dstv


def run_gpt(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd = Random()
    rnd.seed(42)
    seq_error = np.zeros(shape=(n_rounds,1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    for t in range(n_rounds):
        curr_id = rnd.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        next_query = get_next_query(curr_query, model_dict, setting)
        score = get_recommendation_score(ground_queries, next_query)
        if score >= 0.5:
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        simv[t] = get_similarity(ft, curr_query, next_query)
        dstv[t] = get_distance(ft, curr_query, next_query)

    return seq_error, simv, dstv


def run_exp3(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    cand_set_sz = 25
    from random import Random
    rnd1 = Random()
    rnd1.seed(42)
    rnd2 = Random()
    rnd2.seed(99)
    random.seed(42)
    eta = 1e-3
    seq_error = np.zeros(shape=(n_rounds, 1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    r_t = 1
    w_t = dict()
    curr_id = true_ids[0]   #for curr_id in true_ids[:-1]:  #p_t = list()
    curr_query = X[curr_id]
    cand = get_recommendations(curr_query, cand_set_sz, model_dict, setting)
    cand_sz = len(cand)
    logger.info("candidate set are: {}".format(','.join(map(str, cand))))
    S = [0 for i in range(cand_sz)]

    for t in range(n_rounds):
        p = [ exp(eta*S[i]) for i in range(cand_sz) ]
        p = p/sum(p)
        curr_id = rnd1.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        ind = rnd2.choices(range(len(p)), weights=p)[0]
        logger.info("getting recommendation scores")
        score = get_recommendation_score(ground_queries,list(cand)[ind])
        logger.info("recommendation score is: %f" %(score))
        
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            r_t = 0
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        S[ind] = S[ind] +  1 - (1-r_t)/p[ind]

        simv[t] = get_similarity(ft, curr_query, w_k[ind])
        dstv[t] = get_distance(ft, curr_query, w_k[ind])

    return seq_error, simv , dstv

def run_exp3_ss(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    from random import Random
    rnd1 = Random()
    rnd1.seed(42)
    rnd2 = Random()
    rnd2.seed(99)
    random.seed(42)
    eta = 1e-3
    seq_error = np.zeros(shape=(n_rounds, 1))
    simv = np.zeros(shape=(n_rounds, 1))
    dstv = np.zeros(shape=(n_rounds, 1))
    r_t = 1
    w_t = dict()
    cand = set()

    for t in range(n_rounds):
        curr_id = rnd1.choice(true_ids)   #for curr_id in true_ids[:-1]:  #p_t = list()
        curr_query = X[curr_id]
        logging.info("Running recommendations for id : %d" %(curr_id))
        logging.info("Corresponding query is : %s" %(curr_query))
        ground_actions = true_ids.copy()
        ground_actions.remove(curr_id)  #this is the possible set of actions that are correct
        ground_queries = X[ground_actions]
        cand_t = get_recommendations(curr_query, cand_set_sz, model_dict, setting)
        tsz = len(cand)
        cand_sz = 1 if tsz == 0 else tsz
        cand_t = cand_t.difference(cand)
        tsz = len(cand_t)
        cand_t_sz = 1 if tsz == 0 else tsz
        if (t == 0):
            for q in cand_t:
                w_t[q] = eta/((1-eta)*cand_t_sz)
        else:
            W = sum([w_t[q] for q in cand])
            for q in cand_t:
                w_t[q] = (eta*W)/((1-eta)*cand_t_sz)

        w_k = list(w_t.keys())
        s_w = sum(w_t.values())
        p_t = [ (1-eta)*w/s_w + eta/cand_sz for w in w_t.values() ]
        cand.update(cand_t)
        logger.info("candidate set are: {}".format(','.join(map(str, cand))))
        ind = rnd2.choices(range(len(p_t)), weights=p_t)[0]
        logger.info("getting recommendation scores")
        score = get_recommendation_score(ground_queries,w_k[ind])
        logger.info("recommendation score is: %f" %(score))
        if score >= 0.5:
            r_t = 1
            if (t > 0):
                seq_error[t] = seq_error[t-1]
        else:
            r_t = 0
            seq_error[t] = 1 if (t==0) else seq_error[t-1] + 1.0

        r_hat = r_t/p_t[ind]
        w_t[w_k[ind]] = w_t[w_k[ind]]*np.exp(eta*r_hat)

        simv[t] = get_similarity(ft, curr_query, w_k[ind])
        dstv[t] = get_distance(ft, curr_query, w_k[ind])
    return seq_error, simv, dstv


def policy_evaluation(bandit, setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft):
    if bandit == 'EXP3-SS':
        return run_exp3_ss(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'EXP3':
        return run_exp3(setting, model_dict, X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'GPT':
        return run_gpt(setting, model_dict['GPT'], X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'CTRL':
        return run_ctrl(setting, model_dict['CTRL'], X, true_ids, n_rounds, cand_set_sz, ft)
    if bandit == 'XL':
        return run_xl(setting, model_dict['XL'], X, true_ids, n_rounds, cand_set_sz, ft)

def regret_calculation(seq_error):
    t = len(seq_error)
    regret = [x / y for x, y in zip(seq_error, range(1, t + 1))]
    return regret 

def get_distance(ft, curr_query, pred_query):
    pred = ' '.join(list(set(pred_query.split())))
    curr = ' '.join(list(set(curr_query.split())))
    p_vec = get_query_embeddings(ft, pred)
    c_vec = get_query_embeddings(ft, curr)
    return euclidean_distances(p_vec.reshape(1,-1),c_vec.reshape(1,-1))[0][0]

def get_similarity(ft, curr_query, pred_query):
    pred = ' '.join(list(set(pred_query.split())))
    curr = ' '.join(list(set(curr_query.split())))
    p_vec = get_query_embeddings(ft, pred)
    c_vec = get_query_embeddings(ft, curr)
    return cosine_similarity(p_vec.reshape(1,-1),c_vec.reshape(1,-1))[0][0]
