import os
import torch
import graph_tool
import transformers
import torch.nn as nn
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from models import AttenModel, ExpMatchModel
from utils import num_total_ques, read_obj, QAData, ExpData
from utils import collate_fn, write_obj, get_time, extract_subg_pq, rewrite_subg, cal_eval_metric

class SubgraphReasoning:
    def __init__(self):

        self.train_qid2que, self.valid_qid2que, self.test_qid2que = [
            read_obj(file_path=os.path.join(args.in_path, '{}_qid2que.pickle'.format(_))) for _ in [
                'train', 'dev', 'test']]
        self.train_qid2embeds, self.valid_qid2embeds, self.test_qid2embeds = [
            read_obj(file_path=os.path.join(args.in_path, '{}_qid2embeds.pickle'.format(_))) for _ in [
                'train', 'dev', 'test']]

        self.ent2id, self.rel2id = read_obj(file_path=os.path.join(args.in_path, 'ent_rel2id.pickle'))
        self.id2ent, self.id2rel = {v: k for k, v in self.ent2id.items()}, {v: k for k, v in self.rel2id.items()}

        self.ent2label = read_obj(file_path=os.path.join(args.in_path, 'ent2label.pickle'))

        self.rel2embeds = read_obj(file_path=os.path.join(args.in_path, 'rel2embeds.pickle'))

        self.dis_comp = nn.PairwiseDistance(p=args.norm)
        self.sigmoid = nn.Sigmoid()

        self.train_qid2subgs, self.valid_qid2subgs, self.test_qid2subgs = self.subg_extract()

        if args.dataset in ['pq-2hop', 'pq-3hop']:
            tmp_file = '{}H-kb.pickle'.format(args.dataset[-4])
        else:
            tmp_file = 'PQL{}-KB.pickle'.format(args.dataset[-4])
        _, _, self.trp2id = read_obj(file_path=os.path.join(args.in_path, tmp_file))

    def subg_extract(self):
        if not os.path.exists(args.subgraph_data_path) or args.prep_subg:
            if not os.path.exists(args.subgraph_data_path):
                os.makedirs(args.subgraph_data_path)
            print('### Extracting Subgraphs based on Candidate Answers')
            atten_model = AttenModel(rel2embeds=self.rel2embeds, in_dim=args.in_dim,
                                     hid_dim=args.align_hid_dim, num_layers=args.num_gcn_layers,
                                     dropout=0.).cuda()
            atten_model.load_state_dict(torch.load(os.path.join(args.align_model_path, 'best.tar'),
                                                   map_location=torch.device(torch.cuda.current_device()))['model_state'])
            rel_embeds = atten_model.rel_enc()
            for qid2que, qid2embeds, mode in zip(
                    [self.train_qid2que, self.valid_qid2que, self.test_qid2que],
                    [self.train_qid2embeds, self.valid_qid2embeds, self.test_qid2embeds],
                    ['train', 'valid', 'test']):
                print('* extracting and rewriting reasoning subgraphs for {} questions...'.format(mode))
                qid2subg_exp, corr_count = {}, 0
                tmp_top = args.train_top if mode == 'train' else args.valid_test_top
                qa_data = QAData(qid2que=qid2que, qid2embeds=qid2embeds, max_pos_neg_pairs=100)
                data_loader = DataLoader(dataset=qa_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
                for batch_id, batch_data in enumerate(tqdm(data_loader)):
                    # generate candidate answers
                    qid, num_subg_ents, edge_index, edge_attr, loc_tops, _, que_embeds, _, _ = batch_data[0]
                    x, fin_que_embed = atten_model(que_embeds=que_embeds, r=rel_embeds, num_subg_ents=num_subg_ents,
                                                   edge_index=edge_index, edge_attr=edge_attr, loc_tops=loc_tops)
                    all_dis = self.sigmoid(self.dis_comp(x, fin_que_embed))  # size: (num_subg_ents)
                    values, indices = torch.sort(all_dis)
                    values, indices = values.tolist(), indices.tolist()

                    glo_ents = torch.unique(torch.LongTensor(qid2que[qid]['glo_ents']))

                    can_ents = indices[:tmp_top] if len(indices) > tmp_top else indices
                    can_ents = [int(glo_ents[_]) for _ in can_ents]

                    all_ents = [int(glo_ents[_]) for _ in indices]
                    ent2dis = {ent: val for ent, val in zip(all_ents, values)}

                    # prepare directed KG for reasoning subgraph extraction
                    gt_graph = graph_tool.Graph(directed=True)
                    eprop = gt_graph.new_edge_property('int')
                    q_data = qid2que[qid]
                    edge_index = torch.LongTensor(q_data['edge_index'])
                    edge_attr = torch.LongTensor(q_data['edge_attr'])
                    edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)
                    edge_attr = torch.cat([edge_attr, edge_attr + len(self.rel2id)], dim=0)
                    for h, r, t in zip(edge_index[0].tolist(), edge_attr.tolist(), edge_index[1].tolist()):
                        eprop[gt_graph.add_edge(h, t)] = r
                    gt_graph.edge_properties['edge_attr'] = eprop

                    tail_rel2head = defaultdict(set)
                    for edge_idx in range(edge_index.size(1)):
                        head, rel, tail = [int(edge_index[0, edge_idx]), int(edge_attr[edge_idx]),
                                           int(edge_index[1, edge_idx])]
                        tail_rel2head[(tail, rel)].add(head)

                    # extract reasoning subgraphs
                    glo_ans, glo_tops = q_data['glo_ans'], q_data['top_ents']
                    all_subg_set, all_can_subgs, subg2ents, subg2explicit_subgs = set(), [], {}, {}

                    ent2consider = set(can_ents + glo_ans) if mode == 'train' else can_ents

                    for ent in ent2consider:
                        tmp_subgs, tmp_subg_ans, all_subgs_subgs = extract_subg_pq(ordered_tops=glo_tops, ans=ent, gt_graph=gt_graph,
                                                                                   cutoff=args.cutoff, num_rels=len(self.rel2id),
                                                                                   tail_rel2head=tail_rel2head)
                        if len(tmp_subgs) <= args.max_num_subgs_per_ent:
                            for subg, subg_ans, subgs_subgs in zip(tmp_subgs, tmp_subg_ans, all_subgs_subgs):
                                if subg not in all_subg_set:
                                    all_subg_set.add(subg)
                                    subg2ents[subg] = subg_ans
                                    subg2explicit_subgs[subg] = subgs_subgs
                                    num_pos_ans = len([_ for _ in subg_ans if _ in glo_ans])
                                    num_neg_ans = len(subg_ans) - num_pos_ans
                                    all_can_subgs.append([subg, num_pos_ans, num_neg_ans, num_pos_ans - num_neg_ans])

                    all_can_subg_exps = [rewrite_subg(subg=subg[0], que=q_data['question'], id2ent=self.id2ent,
                                                      ent2label=self.ent2label, id2rel=self.id2rel)
                                         for subg in all_can_subgs]
                    if len(all_can_subgs) == 0:
                        print('extracted no subgraph for q-{}: {}'.format(qid, q_data['question']))

                    # determine positive and negative reasoning subgraphs
                    pos_subgs, neg_subgs = [], []
                    pos_subg_exps, neg_subg_exps = [], []
                    sorted_all_subgs = sorted(zip(all_can_subgs, all_can_subg_exps),
                                              key=lambda _: _[0][3], reverse=True)
                    max_vote = sorted_all_subgs[0][0][3] if len(all_can_subgs) > 0 else 0
                    # min_pos_length = 999
                    # for _ in sorted_all_subgs:
                    #     if _[0][3] == max_vote and _[0][1] > 0:
                    #         if len(_[0][0]) < min_pos_length:
                    #             min_pos_length = len(_[0][0])
                    for _ in sorted_all_subgs:
                        if _[0][3] == max_vote and _[0][1] > 0: # and len(_[0][0]) == min_pos_length:
                            pos_subgs.append(_[0][0])
                            pos_subg_exps.append(_[1])
                        else:
                            neg_subgs.append(_[0][0])
                            neg_subg_exps.append(_[1])

                    # check extracted reasoning subgraphs
                    estimated_ans = set()
                    for subg in pos_subgs:
                        estimated_ans.update(subg2ents[subg])
                    final_ans, final_dis = None, 999.
                    for ans in estimated_ans:
                        if ent2dis[ans] < final_dis:
                            final_ans = ans
                            final_dis = ent2dis[ans]
                    if final_dis != 999.:
                        if final_ans in glo_ans:
                            corr_count += 1

                    qid2subg_exp[qid] = [all_can_subgs, all_can_subg_exps, subg2ents, ent2dis,
                                         pos_subgs, pos_subg_exps, neg_subgs, neg_subg_exps, subg2explicit_subgs]

                if mode == 'train':
                    num_ques = args.num_train
                elif mode == 'valid':
                    num_ques = args.num_valid
                else:
                    num_ques = args.num_test
                print('\t * {:.7f}: {} out of {} questions can be answered via positive reasoning subgraphs'.format(
                    corr_count/num_ques, corr_count, num_ques,))

                write_obj(obj=qid2subg_exp, file_path=os.path.join(args.subgraph_data_path,
                                                                   '{}_qid2subg_exp.pickle'.format(mode)))
        return [read_obj(os.path.join(args.subgraph_data_path,
                                      '{}_qid2subg_exp.pickle'.format(mode))) for mode in ['train', 'valid', 'test']]

    def train(self):
        print('### Training')
        exp_match_model = ExpMatchModel(lm_name=args.lm_name, norm=args.norm,
                                        reinit_n=args.reinit_n)
        exp_match_model.cuda()

        named_params, param_groups = list(exp_match_model.named_parameters()), []
        no_decay = ['bias', 'layer_norm', 'LayerNorm']  # no weight decay for bias and layer norm parameters
        lr = args.lr
        for layer in range(5, -1, -1):
            param_decay = [p for n, p in named_params if 'layer.{}'.format(layer) in n
                           and not any(nd in n for nd in no_decay)]
            param_no_decay = [p for n, p in named_params if 'layer.{}'.format(layer) in n
                              and any(nd in n for nd in no_decay)]
            param_groups.append({'params': param_decay, 'lr': lr, 'weight_decay': args.weight_decay})
            param_groups.append({'params': param_no_decay, 'lr': lr, 'weight_decay': 0.})
            lr = lr * args.lr_decay
        param_decay = [p for n, p in named_params if 'embeddings' in n and not any(nd in n for nd in no_decay)]
        param_no_decay = [p for n, p in named_params if 'embeddings' in n and any(nd in n for nd in no_decay)]
        param_groups.append({'params': param_decay, 'lr': lr, 'weight_decay': args.weight_decay})
        param_groups.append({'params': param_no_decay, 'lr': lr, 'weight_decay': 0.})

        num_params = 0
        for group in param_groups:
            num_params += len(group['params'])
        assert num_params == len(list(exp_match_model.parameters())), 'missing parameter error!'

        optimizer = transformers.AdamW(params=param_groups, lr=args.lr)
        criterion = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y:
                                                     1.0 - nn.functional.cosine_similarity(x, y),
                                                     margin=args.margin,
                                                     reduction=args.loss_red)

        train_data = ExpData(qid2que=self.train_qid2que, qid2subgs=self.train_qid2subgs,
                             neg_size=args.neg_size, mode='train')
        train_loader = DataLoader(dataset=train_data, collate_fn=collate_fn,
                                  batch_size=args.batch_size, shuffle=True)

        train_steps = args.num_epochs * len(train_loader)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer=optimizer,
                                                                 num_warmup_steps=args.num_warmup_steps,
                                                                 num_training_steps=train_steps)

        print('* before training')
        best_acc = self.eval(mode='valid', exp_match_model=exp_match_model)['hits@1']

        for epoch in range(args.num_epochs):
            print('* epoch {} - {}'.format(epoch, get_time()))
            epoch_loss = 0.
            exp_match_model.train()
            for batch_id, batch_data in enumerate(train_loader):
                batch_loss = torch.tensor(0.).cuda()
                exp_match_model.zero_grad()
                for single_data in batch_data:
                    (_, que, _, _, _, _, _, num_pos, _, pos_subg_exps,
                     num_neg, _, neg_subg_exps, subg2explicit_subgs) = single_data
                    if num_pos > 0 and num_neg > 0:
                        que_embed = exp_match_model.que_embed(que=que)  # size: (1, 768)
                        pos_embeds = exp_match_model(subg_exps=pos_subg_exps)  # size: (num_pos_exps, 768)
                        neg_embeds = exp_match_model(subg_exps=neg_subg_exps)  # size: (num_neg_exps, 768)

                        que_embed = que_embed.expand(num_pos * num_neg, -1)
                        pos_embeds = torch.flatten(pos_embeds.unsqueeze(1).expand(-1, num_neg, -1), end_dim=1)
                        neg_embeds = torch.flatten(neg_embeds.unsqueeze(0).expand(num_pos, -1, -1), end_dim=1)

                        batch_loss += criterion(que_embed, pos_embeds, neg_embeds)

                if batch_loss != 0.:
                    batch_loss.backward()
                    optimizer.step()
                    epoch_loss += batch_loss.item()
                    scheduler.step()

                if batch_id != 0 and batch_id % args.valid_batch_freq == 0:
                    acc = self.eval(mode='valid', exp_match_model=exp_match_model)['hits@1']
                    if acc > best_acc:
                        ckp = {
                            'epoch': epoch,
                            'timestamp': get_time(),
                            'model_state': exp_match_model.state_dict(),
                            'optimizer_state': optimizer.state_dict(),
                        }

                        if not os.path.exists(args.subgraph_model_path):
                            os.makedirs(args.subgraph_model_path)
                        torch.save(ckp, os.path.join(args.subgraph_model_path, 'best.tar'))
                        print('\t* hits@1 increased: {} -> {} at batch {} time {}'.format(best_acc, acc,
                                                                                          batch_id, get_time()))
                        best_acc = acc

            print('\t* loss: {}, lr: {}, time: {}'.format(epoch_loss, optimizer.param_groups[0]['lr'], get_time()))

            ckp = {
                'epoch': epoch,
                'timestamp': get_time(),
                'model_state': exp_match_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
            if not os.path.exists(args.subgraph_model_path):
                os.makedirs(args.subgraph_model_path)
            torch.save(ckp, os.path.join(args.subgraph_model_path, 'current.tar'))

            acc = self.eval(mode='valid', exp_match_model=exp_match_model)['hits@1']
            if acc > best_acc or epoch == 0:
                torch.save(ckp, os.path.join(args.subgraph_model_path, 'best.tar'))
                print('\t* hits@1 increased: {} -> {} at {}'.format(best_acc, acc, get_time()))
                best_acc = acc

    def eval(self, mode: str, exp_match_model: ExpMatchModel):
        assert mode in {'test', 'valid'}, 'invalid evaluation mode: {}!'.format(mode)

        exp_match_model.eval()
        with torch.no_grad():
            eval_data = ExpData(qid2que=self.valid_qid2que if mode == 'valid' else self.test_qid2que,
                                qid2subgs=self.valid_qid2subgs if mode == 'valid' else self.test_qid2subgs,
                                neg_size=args.neg_size, mode=mode)
            eval_loader = DataLoader(dataset=eval_data, collate_fn=collate_fn, batch_size=1, shuffle=False)

            subg_count, hits1_count = 0, 0
            all_precision, all_recall, all_f1, all_hits = [], [], [], []
            for batch_id, batch_data in enumerate(eval_loader):
                [qid, que, target_ans, all_can_subgs, all_can_subg_exps,
                 subg2ents, ent2dis, _, pos_subgs, _, _, _, _, _] = batch_data[0]

                if len(all_can_subg_exps) == 0:
                    continue

                all_can_subgs = [_[0] for _ in all_can_subgs]

                que_embeds = exp_match_model.que_embed(que=que)

                can_exp_batches = [all_can_subg_exps[_:_ + args.eval_enc_batch]
                                   for _ in range(0, len(all_can_subg_exps), args.eval_enc_batch)]
                all_embeds = [exp_match_model(subg_exps=exp_batch) for exp_batch in can_exp_batches]
                all_embeds = torch.cat(all_embeds, dim=0)

                all_scores = nn.functional.cosine_similarity(que_embeds, all_embeds)

                subg_exp_score_tuples = list(zip(all_can_subgs, all_can_subg_exps, all_scores.tolist()))
                subg_exp_score_tuples = sorted(subg_exp_score_tuples, key=lambda x: x[2], reverse=True)
                max_score = subg_exp_score_tuples[0][2]

                final_subgs = []

                flag = True
                derived_ans = set()
                for ans_tuple in subg_exp_score_tuples:
                    if ans_tuple[2] == max_score: # and len(ans_tuple[0]) == shortest_length:
                        subg = ans_tuple[0]
                        derived_ans.update(subg2ents[subg])
                        final_subgs.append(subg)
                        if subg in pos_subgs and flag:
                            subg_count += 1
                            flag = False

                final_ans, final_dis = None, 999.
                for ans in derived_ans:
                    if ent2dis[ans] < final_dis:
                        final_ans = ans
                        final_dis = ent2dis[ans]

                pred_ans = set()
                for ans in derived_ans:
                    if ent2dis[ans] <= args.threshold * final_dis:
                        pred_ans.add(ans)

                if final_ans in target_ans:
                    hits1_count += 1
                    # hits1_flag = True
                # else:
                #     print('\t\t * failed question id: {}'.format(qid))

                precision, recall, f1, hits = cal_eval_metric(best_pred=final_ans,
                                                              preds=list(pred_ans),
                                                              answers=target_ans)

                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
                all_hits.append(hits)

            num_eval_ques = args.num_valid if mode == 'valid' else args.num_test

            # hits w.r.t all questions
            results = {'hits@1': hits1_count / num_eval_ques,
                       'pos_ratio': subg_count / num_eval_ques,}

            print('\t* precision: {}, recall: {}, f1: {}'.format(
                sum(all_precision) / num_eval_ques,
                sum(all_recall) / num_eval_ques,
                sum(all_f1) / num_eval_ques,
                # sum(all_hits) / num_eval_ques
            ))

            print('\t* ' + ', '.join(['{}: {:.7f}'.format(key, value) for key, value in results.items()]))

            return results

    def test(self):
        print('### Test')
        exp_match_model = ExpMatchModel(lm_name=args.lm_name, norm=args.norm,
                                        reinit_n=args.reinit_n)
        exp_match_model.cuda()
        if args.num_epochs == 0 and args.pre_timestamp:
            ckp_path = os.path.join(args.subgraph_model_path.replace(args.timestamp,
                                                                          args.pre_timestamp), 'best.tar')
        else:
            ckp_path = os.path.join(args.subgraph_model_path, 'best.tar')
        ckp = torch.load(ckp_path, map_location=torch.device(torch.cuda.current_device()))
        print('* loaded the pre-trained model from {} which are cached at {}'.format(ckp_path, ckp['timestamp']))
        exp_match_model.load_state_dict(ckp['model_state'])

        self.eval(mode='test', exp_match_model=exp_match_model)


if __name__ == '__main__':
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='pql-2hop',
                            help='specify which dataset to use',
                            choices={'CWQ', 'WebQuestionsSP', 'pq-2hop', 'pq-3hop', 'pql-2hop', 'pql-3hop'})
    arg_parser.add_argument('--timestamp', type=str, default=timestamp)

    arg_parser.add_argument('--align_timestamp', type=str, default=None,
                            help='timestamp of pre-trained GNN module')
    arg_parser.add_argument('--pre_timestamp', type=str, default=None,
                            help='timestamp of pre-trained ranking module')

    arg_parser.add_argument('--lm_name', type=str, default='sentence-transformers/multi-qa-distilbert-cos-v1',
                            help='name of the pre-trained language model to use')

    arg_parser.add_argument('--train_top', type=int, default=None,
                            help='number of candidate answers for training question')
    arg_parser.add_argument('--valid_test_top', type=int, default=None,
                            help='number of candidate answers for validation/test question')

    arg_parser.add_argument('--prep_subg', action='store_true', default=False,
                            help='whether to extract reasoning subgraphs')

    arg_parser.add_argument('--cutoff', type=int, default=2,
                            help='the cutoff when preparing candidate sequences')

    arg_parser.add_argument('--in_path', type=str, default='datasets/{}/in_path',
                            help='path of preprocessed data (output of the que_prep.py script)')
    arg_parser.add_argument('--align_model_path', type=str, default='datasets/{}/out_path/{}',
                            help='path of the pre-trained alignment model')
    arg_parser.add_argument('--subgraph_data_path', type=str, default='datasets/{}/out_path/{}/{}',
                            help='path of extracted reasoning subgraphs')
    arg_parser.add_argument('--subgraph_model_path', type=str, default='datasets/{}/out_path/{}/{}/{}',
                            help='path of the explicit reasoning model')

    arg_parser.add_argument('--in_dim', type=int, default=768,
                            help='dimension of pre-encoded question/relation tokens')
    arg_parser.add_argument('--hid_dim', type=int, default=256,
                            help='dimension of embeddings')
    arg_parser.add_argument('--align_hid_dim', type=int, default=256,
                            help='dimension of embeddings in pre-trained alignment module')
    arg_parser.add_argument('--num_gcn_layers', type=int, default=3,
                            help='number of GCN layers in the graph encoder')

    arg_parser.add_argument('--norm', type=int, default=2,
                            help='order of norm when computing embedding distances')
    arg_parser.add_argument('--margin', type=float, default=0.1,
                            help='margin for triplet distance loss')
    arg_parser.add_argument('--eval_enc_batch', type=int, default=512,
                            help='the number of candidate reasoning subgraph to encode in batch')
    arg_parser.add_argument('--num_epochs', type=int, default=0,
                            help='number of training epochs')
    arg_parser.add_argument('--batch_size', type=int, default=12,
                            help='batch size for fine-tuning')
    arg_parser.add_argument('--neg_size', type=int, default=32,
                            help='positive and negative sample size for fine-tuning the LM')
    arg_parser.add_argument('--lr', type=float, default=8e-6,
                            help='learning rate')
    arg_parser.add_argument('--lr_decay', type=float, default=1.,
                            help='layer-wise learning rate decay')
    arg_parser.add_argument('--weight_decay', type=float, default=1e-3,
                            help='weight_decay')
    arg_parser.add_argument('--num_warmup_steps', type=int, default=500,
                            help='number of warmup steps in fine-tuning')
    arg_parser.add_argument('--reinit_n', type=int, default=0,
                            help='reinitialize linear and layer norm weights in top-n layers')
    arg_parser.add_argument('--valid_batch_freq', type=int, default=500,
                            help='do validation every x batches in each epoch')
    arg_parser.add_argument('--loss_red', type=str, default='mean',
                            help='reduction in loss computation')
    arg_parser.add_argument('--max_num_subgs_per_ent', type=int, default=10000,
                            help='ignore candidate answers with more than this number of subgraphs')
    arg_parser.add_argument('--threshold', type=float, default=1.5)

    arg_parser.add_argument('--num_train', type=int, default=0,
                            help='the total number of training questions for computing eval metrics'
                                 'questions unanswerable due to subgraph preparation errors are not included in the prepared data')
    arg_parser.add_argument('--num_valid', type=int, default=0,
                            help='the total number of validation questions for computing eval metrics'
                                 'questions unanswerable due to subgraph preparation errors are not included in the prepared data')
    arg_parser.add_argument('--num_test', type=int, default=0,
                            help='the total number of test questions for computing eval metrics'
                                 'questions unanswerable due to subgraph preparation errors are not included in the prepared data')

    args = arg_parser.parse_args()
    args.in_path = args.in_path.format(args.dataset)
    args.align_model_path = args.align_model_path.format(args.dataset, args.align_timestamp)
    args.subgraph_data_path = args.subgraph_data_path.format(args.dataset, args.align_timestamp,
                                                             '{}-{}'.format(args.train_top, args.valid_test_top))
    args.subgraph_model_path = args.subgraph_model_path.format(args.dataset, args.align_timestamp,
                                                               '{}-{}'.format(args.train_top, args.valid_test_top),
                                                               args.timestamp)

    args.num_train = num_total_ques[args.dataset]['train']
    args.num_valid = num_total_ques[args.dataset]['dev']
    args.num_test = num_total_ques[args.dataset]['test']

    print('\n## Subgraph Reasoning - {} - Top-{} - {}'.format(args.dataset, '{}-{}'.format(args.train_top,
                                                                                           args.valid_test_top),
                                                              args.timestamp))
    for k_, v_ in vars(args).items():
        print('* {}: {}'.format(k_, v_))

    subg_reasoning = SubgraphReasoning()
    if args.num_epochs > 0:
        subg_reasoning.train()
    subg_reasoning.test()
