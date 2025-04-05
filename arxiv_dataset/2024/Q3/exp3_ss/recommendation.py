import random
import logging
logger = logging.getLogger(__name__)


def get_recommendation_score(ground_truth,prediction):
    pred_set = set(prediction.split())
    rewards = [len(list(set(g.split()) & pred_set))/len(set(g.split())) for g in ground_truth]
    return max(rewards)


def get_recommendations(curr_query, cand_set_sz, model_dict, setting):
    import torch
    context_q_no = len(curr_query.split())
    cand = set()
    mlen = 2*context_q_no + 8

    if setting == 'scratch':
        #from transformers import BertTokenizer
        #vocab = '../Data/semanticscholar/tokenizer/wordpiece/vocab.txt'
        #tokenizer = BertTokenizer(vocab_file=vocab, unk_token='[unk]', cls_token='[bos]', sep_token='[sep]', bos_token='[bos]', eos_token='[eos]', pad_token='[pad]')
        for method in model_dict.keys():
            logging.info("getting recommendations for %s trained from %s" %(method,setting))
            tokenizer = model_dict[method]['tok']
            model = model_dict[method]['model']
            sep = model_dict[method]['sep']
            special_tokens = model_dict[method]['spl_tok']
            input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
            rmds = [ tokenizer.decode(outputs[i], skip_special_tokens=False).split(sep)[1] for i in range(cand_set_sz) ]
            for i in range(len(rmds)):
                for j in special_tokens:
                    rmds[i] = rmds[i].replace(j,'')
            cand.update(rmds)

    if setting == 'pretrained':
        for method in model_dict.keys():
            logging.info("getting recommendations for %s trained from %s" %(method,setting))
            tokenizer = model_dict[method]['tok']
            model = model_dict[method]['model']
            sep = model_dict[method]['sep']
            special_tokens = model_dict[method]['spl_tok']

            input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=cand_set_sz, max_length=mlen, do_sample=False, temperature=0.4)
            rmds = list()
            for i in range(cand_set_sz):
                ss = tokenizer.decode(outputs[i], skip_special_tokens=False)
                if sep in ss:
                    rmds.append(ss.split(sep)[1])
                else:
                    rmds.append(special_tokens[0])
                #rmds = [ tokenizer.decode(outputs[i], skip_special_tokens=False).split(sep)[1] for i in range(cand_set_sz) ]
            for i in range(len(rmds)):
                for j in special_tokens:
                    rmds[i] = rmds[i].replace(j,'')
            cand.update(rmds)

    return cand

def get_next_query(curr_query, model_dict, setting):
    import torch

    #pretrained_models = ['GPT','XL','CTRL','BERT','BART']
    #pretrained_models = ['GPT', 'XL', 'CTRL']
    #scratch_models = ['GPT', 'CTRL']
    #logger.info("running baseline query recommendation algorithms")
    context_q_no = len(curr_query.split())
    mlen = 2*context_q_no + 8
    #mlen = len(curr_query.split()) + max([len(q.split()) for q in curr_query]) + context_q_no + 4

    if setting == 'pretrained':
            sep = model_dict['sep']
            special_tokens = model_dict['spl_tok']
            tokenizer = model_dict['tok']
            model = model_dict['model']
            input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, max_length=mlen, do_sample=False, temperature=0.4)
            next_query = "" #tokenizer.decode(outputs[0], skip_special_tokens=False).split(sep)[1]
            ss = tokenizer.decode(outputs[0], skip_special_tokens=False)
            if sep in ss:
                next_query += ss.split(sep)[1]
            else:
                next_query += special_tokens[0]
            for tok in special_tokens:
                next_query = next_query.replace(tok, '')

            return next_query.strip()
    else:
            tokenizer = model_dict['tok']
            model = model_dict['model']
            sep = model_dict['sep']
            special_tokens = model_dict['spl_tok']
            input_ids = torch.tensor(tokenizer.encode(curr_query)).unsqueeze(0)
            outputs = model.generate(input_ids=input_ids, num_beams=20, num_return_sequences=1, max_length=mlen, do_sample=False, temperature=0.4)
            next_query = tokenizer.decode(outputs[0], skip_special_tokens=False).split(sep)[1]
            for tok in special_tokens:
                next_query = next_query.replace(tok, '')
            return next_query.strip()
