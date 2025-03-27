import os
import json
import torch
import graph_tool
from tqdm import tqdm
from OpenNER import OpenNER
from datetime import datetime
from graph_tool import topology
from collections import defaultdict
from argparse import ArgumentParser, Namespace
from transformers import BertTokenizer, BertModel
from utils import DeviceAction, write_obj, read_obj


def load_que_subg(args: Namespace):
    print('\n* preparing questions and subgraphs...')
    ent2id = {}
    num_ents = 0
    rel2id = {}
    num_rels = 0
    with open(os.path.join(args.in_path, 'entities.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            assert line.strip() not in ent2id, 'found a duplicate entity {}'.format(line.strip())
            ent2id[line.strip()] = num_ents
            num_ents += 1
    print('\t* number of entities: {}'.format(num_ents))
    with open(os.path.join(args.in_path, 'relations.txt'), 'r') as f:
        lines = f.readlines()
        for line in lines:
            assert line.strip() not in rel2id, 'found a duplicate relation {}'.format(line.strip())
            rel2id[line.strip()] = num_rels
            num_rels += 1
    print('\t* number of relations: {}'.format(num_rels))
    write_obj(obj=[ent2id, rel2id], file_path=os.path.join(args.out_path, 'ent_rel2id.pickle'))

    for mode in ['dev', 'test', 'train']:
        with open(os.path.join(args.in_path, '{}_simple.json'.format(mode)), 'r') as f:
            lines = f.readlines()
            qid2que = {}
            num_valid_ques, num_bad_ques = 0, 0
            for line in tqdm(lines):
                line = json.loads(line.strip())
                question = line['question']
                glo_ents = line['subgraph']['entities']
                edge_index = [[], []]
                edge_attr = []
                for trp in line['subgraph']['tuples']:
                    edge_index[0].append(trp[0])
                    edge_attr.append(trp[1])
                    edge_index[1].append(trp[2])
                tmp_top_ents, top_ents = line['entities'], []
                for ent in tmp_top_ents:
                    if ent in glo_ents:
                        top_ents.append(ent)
                if args.dataset[:6] == 'metaqa':
                    tmp_glo_ans, glo_ans = [ent2id[an['text']] for an in line['answers']], []
                else:
                    tmp_glo_ans, glo_ans = [ent2id[an['kb_id']] for an in line['answers']], []
                for ent in tmp_glo_ans:
                    if ent in glo_ents:
                        glo_ans.append(ent)
                if len(top_ents) != 0 and len(glo_ans) != 0:
                    qid2que[num_valid_ques] = {
                        'question': question,
                        'glo_ents': glo_ents,
                        'edge_index': edge_index,
                        'edge_attr': edge_attr,
                        'top_ents': top_ents,
                        'glo_ans': glo_ans
                    }
                    num_valid_ques += 1
                else:
                    num_bad_ques += 1
                    if len(top_ents) == 0:
                        print(line)
            print('\t* {} - number of answerable and unanswerable questions: {}, {}'.format(mode,
                                                                                            num_valid_ques,
                                                                                            num_bad_ques))
            write_obj(obj=qid2que, file_path=os.path.join(args.out_path, '{}_qid2que.pickle'.format(mode)))


def subg_prep(args: Namespace):
    print('\n* preparing subgraphs...')
    for mode in ['dev', 'test', 'train']:
        qid2que = read_obj(file_path=os.path.join(args.out_path, '{}_qid2que.pickle'.format(mode)))
        qid2subg = {}
        for qid, q_data in tqdm(qid2que.items()):
            head_ents, tail_ents = q_data['edge_index']
            edge_attr = q_data['edge_attr']
            glo_ents = q_data['glo_ents']
            top_ents = q_data['top_ents']
            glo_ans = q_data['glo_ans']

            gt_graph = graph_tool.Graph(directed=False)
            eprop = gt_graph.new_edge_property('int')
            for h, r, t in zip(head_ents, edge_attr, tail_ents):
                eprop[gt_graph.add_edge(h, t)] = r
            gt_graph.edge_properties['edge_attr'] = eprop

            pos_subgs = {}
            for ans in glo_ans:
                pos_subgs[ans] = subg_con(top_ents=top_ents, ans=ans, cutoff=args.cutoff, gt_graph=gt_graph)
            neg_subgs = {}
            for neg_ans in glo_ents:
                if neg_ans not in glo_ans:
                    neg_subgs[neg_ans] = subg_con(top_ents=top_ents, ans=neg_ans,
                                                  cutoff=args.cutoff, gt_graph=gt_graph)
            qid2subg[qid] = [pos_subgs, neg_subgs]
        write_obj(obj=qid2subg, file_path=os.path.join(args.out_path, '{}_qid2subg.pickle'.format(mode)))


def subg_con(top_ents: list, ans: int, cutoff: int, gt_graph: graph_tool.Graph):
    uniq_trps = {}
    for top in top_ents:
        all_paths = topology.all_paths(g=gt_graph, source=gt_graph.vertex(top), target=gt_graph.vertex(ans),
                                       cutoff=cutoff, edges=True)
        for path in all_paths:
            for edge in path:
                head, rel, tail = int(edge.source()), gt_graph.edge_properties['edge_attr'][edge], int(edge.target())
                uniq_trps[(head, rel, tail)] = None
    tmp_edge_index = [[], []]
    tmp_edge_attr = []
    for uniq_trp in uniq_trps.keys():
        tmp_edge_index[0].append(uniq_trp[0])
        tmp_edge_attr.append(uniq_trp[1])
        tmp_edge_index[1].append(uniq_trp[2])
    return tmp_edge_index, tmp_edge_attr


def mask_que_prep(args: Namespace):
    print('\n* preparing masked questions...')
    tagger = OpenNER(bert_model='bert-large-cased', model_dir=args.OpenNER_path)
    for mode in ['dev', 'test', 'train']:
        qid2que = read_obj(file_path=os.path.join(args.out_path, '{}_qid2que.pickle'.format(mode)))
        qid2masked_que = defaultdict(list)
        for qid, q_data in tqdm(qid2que.items()):
            tags = tagger.predict(q_data['question'])
            for tag in tags:
                if tag.split(' ')[1] == 'O':
                    qid2masked_que[qid].append(tag.split(' ')[0])
                else:
                    if len(qid2masked_que[qid]) == 0:
                        qid2masked_que[qid].append('TOP_ENT')
                    else:
                        if qid2masked_que[qid][-1] != 'TOP_ENT':
                            qid2masked_que[qid].append('TOP_ENT')
        write_obj(obj=qid2masked_que,
                  file_path=os.path.join(args.out_path, '{}_qid2masked_que.pickle').format(mode))


def que_rel_enc(args: Namespace):
    print('\n* encoding masked questions and relations...')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(args.device)
    model.eval()
    for mode in ['dev', 'test', 'train']:
        qid2que_embeds = {}
        qid2masked_que = read_obj(file_path=os.path.join(args.out_path, '{}_qid2masked_que.pickle').format(mode))

        for qid, masked_que in tqdm(qid2masked_que.items()):
            qid2que_embeds[qid] = bert_enc(tokenizer=tokenizer, model=model, words=masked_que, device=args.device)

        write_obj(obj=qid2que_embeds, file_path=os.path.join(args.out_path, '{}_qid2embeds.pickle'.format(mode)))

    rel2embeds = {}
    rel2id = read_obj(file_path=os.path.join(args.out_path, 'ent_rel2id.pickle'))[1]

    for rel, rel_id in rel2id.items():
        rel2embeds[rel_id] = bert_enc(tokenizer=tokenizer, model=model,
                                      words=rel.split('.')[-1].split('_'), device=args.device)

    write_obj(obj=rel2embeds, file_path=os.path.join(args.out_path, 'rel2embeds.pickle'))


def bert_enc(tokenizer: BertTokenizer, model: BertModel, words: list, device: torch.device):
    with torch.no_grad():
        filtered_words = []
        for word in words:
            if word != 'TOP_ENT':
                filtered_words.append(word)
        tokenized = tokenizer(' '.join(filtered_words), return_tensors='pt')
        bert_output = model(input_ids=tokenized['input_ids'].to(device),
                            attention_mask=tokenized['attention_mask'].to(device),
                            token_type_ids=tokenized['token_type_ids'].to(device))
    return bert_output['last_hidden_state'].squeeze(0).cpu()


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M")

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='WebQuestionsSP',
                            help='specify which dataset to use',
                            choices={'CWQ', 'WebQuestionsSP'})
    arg_parser.add_argument('--in_path', type=str, default='datasets/{}/origin',
                            help='path of original QA data (please download according to README)')
    arg_parser.add_argument('--out_path', type=str, default='datasets/{}/in_path',
                            help='path of preprocessed data (output of this script)')
    arg_parser.add_argument('--OpenNER_path', type=str, default='datasets/OpenNER_large',
                            help='path to the pre-trained OpenNER model')
    arg_parser.add_argument('--device', type=str, default=torch.device('cuda:0'), action=DeviceAction,
                            help='id of the GPU to use')
    arg_parser.add_argument('--cutoff', type=int, default=2, help='cutoff to use when preparing subgraphs')

    arg_parser.add_argument('--load_que_subg', default=True, action='store_true',
                            help='step1 - load original questions and knowledge graphs')
    arg_parser.add_argument('--subg_prep', default=False, action='store_true',
                            help='step2 - prepare positive/negative subgraphs for training')
    arg_parser.add_argument('--mask_que_prep', default=False, action='store_true',
                            help='step3 - prepare questions with named entities masked')
    arg_parser.add_argument('--que_rel_enc', default=False, action='store_true',
                            help='step4 - pre-encode questions and relations')

    arguments = arg_parser.parse_args()

    arguments.in_path = arguments.in_path.format(arguments.dataset)
    arguments.out_path = arguments.out_path.format(arguments.dataset)

    print('## Data Preparation - {}'.format(timestamp))
    for k, v in vars(arguments).items():
        print('* {}: {}'.format(k, v))

    if arguments.load_que_subg:
        load_que_subg(args=arguments)
    if arguments.subg_prep:
        subg_prep(args=arguments)
    if arguments.mask_que_prep:
        mask_que_prep(args=arguments)
    if arguments.que_rel_enc:
        que_rel_enc(args=arguments)
