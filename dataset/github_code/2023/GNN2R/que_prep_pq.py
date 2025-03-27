import os
import torch
import random
from tqdm import tqdm
from datetime import datetime
from argparse import ArgumentParser
from transformers import BertTokenizer, BertModel
from collections import defaultdict
from utils import write_obj, read_obj


def load_kg():
    ent2id = {}
    num_ents = 0
    rel2id = {}
    num_rels = 0
    trp2id = {}
    num_trps = 0

    print('* loading the KG')

    if args.dataset in ['2-hop', '3-hop']:
        tmp_file = '{}H-kb.txt'.format(args.hop)
    else:
        tmp_file = 'PQL{}-KB.txt'.format(args.hop)

    with open(file=os.path.join(args.data_path, tmp_file), mode='r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            line = line.strip()
            h, r, t = line.split()
            if h not in ent2id:
                ent2id[h] = num_ents
                num_ents += 1
            if r not in rel2id:
                rel2id[r] = num_rels
                num_rels += 1
            if t not in ent2id:
                ent2id[t] = num_ents
                num_ents += 1
            trp = (ent2id[h], rel2id[r], ent2id[t])
            if trp not in trp2id:
                trp2id[trp] = num_trps
                num_trps += 1
    print('\t* #entity: {}, #relation: {}, #triples: {}'.format(num_ents, num_rels, num_trps))

    write_obj(obj=[ent2id, rel2id, trp2id], file_path=os.path.join(args.data_path,
                                                                   tmp_file.replace('txt', 'pickle')))


def load_que_subg():
    print('* preparing questions and subgraphs...')

    if args.dataset in ['2-hop', '3-hop']:
        kg_file = '{}H-kb.pickle'.format(args.hop)
        qa_file = 'PQ-{}H.txt'.format(args.hop)
    else:
        kg_file = 'PQL{}-KB.pickle'.format(args.hop)
        qa_file = 'PQL-{}H.txt'.format(args.hop)

    ent2id, rel2id, trp2id = read_obj(file_path=os.path.join(args.data_path, kg_file))

    glo_ents = list(range(len(ent2id)))
    edge_index = [[], []]
    edge_attr = []
    for trp in trp2id.keys():
        h, r, t = trp
        edge_index[0].append(h)
        edge_attr.append(r)
        edge_index[1].append(t)

    with open(file=os.path.join(args.data_path, qa_file), mode='r') as f:
        lines = f.readlines()
        qid2que = {}
        num_ques = 0
        for line in tqdm(lines):
            line = line.strip()
            que, ans, path = line.split('\t')

            first_ans = ans.replace('_(', '\t').split('(')[0].replace('\t', '_(')

            all_ans = [first_ans]
            for tmp_ans in ans.replace(first_ans, '', 1)[1:-1].split('/'):
                if tmp_ans not in all_ans and tmp_ans != '':
                    all_ans.append(tmp_ans)
            all_ans = [ent2id[_] for _ in all_ans]

            top_ent = ent2id[path.split('#')[0]]

            qid2que[num_ques] = {
                'question': que,
                'glo_ents': glo_ents,
                'edge_index': edge_index,
                'edge_attr': edge_attr,
                'top_ents': top_ent,
                'glo_ans': all_ans,
                'path': path,
            }

            num_ques += 1
        print('\t* #questions: {}'.format(num_ques))

        write_obj(obj=qid2que, file_path=os.path.join(args.data_path, qa_file.replace('txt', 'pickle')))


def que_rel_enc():
    print('\n* encoding masked questions and relations...')

    if args.dataset in ['2-hop', '3-hop']:
        kg_file = '{}H-kb.pickle'.format(args.hop)
        qa_file = 'PQ-{}H.pickle'.format(args.hop)
    else:
        kg_file = 'PQL{}-KB.pickle'.format(args.hop)
        qa_file = 'PQL-{}H.pickle'.format(args.hop)
    qid2que = read_obj(file_path=os.path.join(args.data_path, qa_file))

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').cuda()
    model.eval()

    qid2que_embeds = {}

    for qid, q_data in tqdm(qid2que.items()):
        masked_que = q_data['question'].replace(q_data['path'].split('#')[0], 'TOP_ENT').split(' ')
        qid2que_embeds[qid] = bert_enc(tokenizer=tokenizer, model=model, words=masked_que)

    write_obj(obj=qid2que_embeds, file_path=os.path.join(args.data_path, qa_file.replace('H.', 'H-qid2embeds.')))

    rel2embeds = {}
    rel2id = read_obj(file_path=os.path.join(args.data_path, kg_file))[1]

    if args.dataset in ['2-hop', '3-hop']:
        for rel, rel_id in rel2id.items():
            rel2embeds[rel_id] = bert_enc(tokenizer=tokenizer, model=model, words=rel.split('_'))
    else:
        for rel, rel_id in rel2id.items():
            rel2embeds[rel_id] = bert_enc(tokenizer=tokenizer, model=model, words=rel.split('__')[-1].split('_'))

    write_obj(obj=rel2embeds, file_path=os.path.join(args.out_path, 'rel2embeds.pickle'))


def bert_enc(tokenizer: BertTokenizer, model: BertModel, words: list):
    with torch.no_grad():
        filtered_words = []
        for word in words:
            if word != 'TOP_ENT':
                filtered_words.append(word)
        tokenized = tokenizer(' '.join(filtered_words), return_tensors='pt')
        bert_output = model(input_ids=tokenized['input_ids'].cuda(),
                            attention_mask=tokenized['attention_mask'].cuda(),
                            token_type_ids=tokenized['token_type_ids'].cuda())
    return bert_output['last_hidden_state'].squeeze(0).cpu()


def data_split():
    print('* split PQ data')

    if args.dataset in ['2-hop', '3-hop']:
        kg_file = '{}H-kb.pickle'.format(args.hop)
        qa_file = 'PQ-{}H.pickle'.format(args.hop)
    else:
        kg_file = 'PQL{}-KB.pickle'.format(args.hop)
        qa_file = 'PQL-{}H.pickle'.format(args.hop)
    qid2que = read_obj(file_path=os.path.join(args.data_path, qa_file))
    qid2que_embeds = read_obj(file_path=os.path.join(args.data_path, qa_file.replace('H.', 'H-qid2embeds.')))

    assert len(qid2que) == len(qid2que_embeds)
    train_ids, test_ids, dev_ids = [list(_) for _ in torch.utils.data.random_split(list(qid2que.keys()), args.split)]

    for mode, ids in zip(['train', 'dev', 'test'], [train_ids, dev_ids, test_ids]):
        print('\t* #{}: {}'.format(mode, len(ids)))

        tmp_que_count = 0
        tmp_qid2que = {}
        tmp_qid2embeds = {}

        for idx in ids:
            tmp_qid2que[tmp_que_count] = qid2que[idx]
            tmp_qid2embeds[tmp_que_count] = qid2que_embeds[idx]
            tmp_que_count += 1

        write_obj(obj=ids, file_path=os.path.join(args.out_path, '{}_qid2.pickle'.format(mode)))
        write_obj(obj=tmp_qid2que, file_path=os.path.join(args.out_path, '{}_qid2que.pickle'.format(mode)))
        write_obj(obj=tmp_qid2embeds, file_path=os.path.join(args.out_path, '{}_qid2embeds.pickle'.format(mode)))


def for_subg_reason():
    if args.dataset in ['2-hop', '3-hop']:
        tmp_file = '{}H-kb.txt'.format(args.hop)
    else:
        tmp_file = 'PQL{}-KB.txt'.format(args.hop)

    ent2id, rel2id, _ = read_obj(file_path=os.path.join(args.data_path, tmp_file.replace('txt', 'pickle')))

    write_obj(obj=[ent2id, rel2id], file_path=os.path.join(args.out_path, 'ent_rel2id.pickle'))

    ent2label = {_k: _k for _k, _ in ent2id.items()}

    write_obj(obj=ent2label, file_path=os.path.join(args.out_path, 'ent2label.pickle'))


def exp_subg_prep():
    if args.dataset in ['2-hop', '3-hop']:
        kg_file = '{}H-kb.pickle'.format(args.hop)
    else:
        kg_file = 'PQL{}-KB.pickle'.format(args.hop)

    ent2id, rel2id, trp2id = read_obj(file_path=os.path.join(args.data_path, kg_file))

    for mode in ['train', 'dev', 'test']:

        qid2que = read_obj(file_path=os.path.join(args.out_path, '{}_qid2que.pickle'.format(mode)))

        qid2exp_subg = defaultdict(list)

        for qid, q_data in qid2que.items():
            tmp_path = q_data['path']
            if args.dataset in ['2-hop', '3-hop']:
                tmp_path = tmp_path.split('#')[:-2]
            else:
                tmp_path = tmp_path.split('#')
            for idx in range(0, len(tmp_path) - 1, 2):
                h, r, t = tmp_path[idx], tmp_path[idx + 1], tmp_path[idx + 2]
                trp = (ent2id[h], rel2id[r], ent2id[t])
                qid2exp_subg[qid].append(trp)

        write_obj(obj=qid2exp_subg, file_path=os.path.join(args.out_path, '{}_qid2exp_subg.pickle'.format(mode)))


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M")

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='3-hop',
                            choices={'2-hop', '3-hop', 'l-2-hop', 'l-3-hop'})

    arg_parser.add_argument('--data_path', type=str, default='datasets/PQ/input/{}')
    arg_parser.add_argument('--out_path', type=str, default='datasets/{}/in_path')

    arg_parser.add_argument('--cutoff', type=int, default=3)
    arg_parser.add_argument('--hop', type=int, default=None)

    arg_parser.add_argument('--random_seed', type=int, default=0)
    arg_parser.add_argument('--split', type=float, default=[0.8, 0.1, 0.1], nargs='+')

    args = arg_parser.parse_args()

    args.data_path = args.data_path.format(args.dataset)

    dataset2hop = {
        '2-hop': 2,
        '3-hop': 3,
        'l-2-hop': 2,
        'l-3-hop': 3
    }

    args.hop = dataset2hop[args.dataset]

    tmp = {'2-hop': 'pq-2hop',
           '3-hop': 'pq-3hop',
           'l-2-hop': 'pql-2hop',
           'l-3-hop': 'pql-3hop'}

    args.out_path = args.out_path.format(tmp[args.dataset])

    print('## Data Preparation - {}'.format(timestamp))

    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    load_kg()

    load_que_subg()

    que_rel_enc()

    torch.manual_seed(args.random_seed)
    random.seed(args.random_seed)

    data_split()

    for_subg_reason()

    exp_subg_prep()

