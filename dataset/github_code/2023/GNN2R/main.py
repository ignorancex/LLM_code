import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import AttenModel
from datetime import datetime
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from utils import num_total_ques, get_time, read_obj, collate_fn, compute_hits, QAData, cal_eval_metric


class AttenTrain:
    def __init__(self):
        self.train_qid2que, self.valid_qid2que, self.test_qid2que = [
            read_obj(file_path=os.path.join(args.in_path, '{}_qid2que.pickle'.format(_))) for _ in [
                'train', 'dev', 'test']]

        self.train_qid2embeds, self.valid_qid2embeds, self.test_qid2embeds = [
            read_obj(file_path=os.path.join(args.in_path, '{}_qid2embeds.pickle'.format(_))) for _ in [
                'train', 'dev', 'test']]

        self.rel2embeds = read_obj(file_path=os.path.join(args.in_path, 'rel2embeds.pickle'))

    def train(self):
        print('### Training')
        atten_model = AttenModel(rel2embeds=self.rel2embeds, in_dim=args.in_dim,
                                 hid_dim=args.hid_dim, num_layers=args.num_gcn_layers,
                                 dropout=args.dropout)
        atten_model.cuda()

        optimizer = optim.RAdam(params=atten_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        criterion = nn.TripletMarginWithDistanceLoss(distance_function=nn.PairwiseDistance(p=args.norm),
                                                     margin=args.margin, reduction=args.loss_red)

        train_data = QAData(qid2que=self.train_qid2que, qid2embeds=self.train_qid2embeds,
                            max_pos_neg_pairs=args.max_pos_neg_pairs)
        train_loader = DataLoader(dataset=train_data, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True)

        if args.pre_timestamp:
            ckp_path = os.path.join(args.out_path.replace(args.timestamp, args.pre_timestamp), 'best.tar')
            ckp = torch.load(ckp_path, map_location=torch.device(torch.cuda.current_device()))
            atten_model.load_state_dict(ckp['model_state'])
            print('* loaded the pre-trained model from {} which was cached at {}'.format(ckp_path, ckp['timestamp']))

        print('* validation results before training')
        highest_acc = self.eval(mode='valid', model=atten_model)['hits@1']

        for epoch in range(args.num_epochs):
            print('* epoch {} - {}'.format(epoch, get_time()))
            epoch_loss = 0.
            atten_model.train()
            for batch_id, batch_data in enumerate(train_loader):
                atten_model.zero_grad()
                r = atten_model.rel_enc()
                batch_loss = torch.tensor(0.).cuda()
                for single_data in batch_data:
                    [_, num_subg_ents, edge_index, edge_attr, loc_tops,
                     _, que_embeds, pos_loc_ans, neg_loc_ans] = single_data
                    x, fin_que_embed = atten_model(que_embeds=que_embeds, r=r, num_subg_ents=num_subg_ents,
                                                   edge_index=edge_index, edge_attr=edge_attr, loc_tops=loc_tops)
                    anchor = fin_que_embed.expand(pos_loc_ans.size()[0], -1)
                    positive = x[pos_loc_ans]
                    negative = x[neg_loc_ans]

                    batch_loss += criterion(anchor, positive, negative)
                batch_loss.backward()
                optimizer.step()
                epoch_loss += batch_loss.item()
            print('\t* loss: {}, lr: {}, time: {}'.format(epoch_loss, optimizer.param_groups[0]['lr'], get_time()))

            ckp = {
                'epoch': epoch,
                'timestamp': get_time(),
                'model_state': atten_model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
            }
            if not os.path.exists(args.out_path):
                os.makedirs(args.out_path)
            torch.save(ckp, os.path.join(args.out_path, 'current.tar'))

            acc = self.eval(mode='valid', model=atten_model)['hits@1']
            if acc > highest_acc:
                torch.save(ckp, os.path.join(args.out_path, 'best.tar'))
                print('\t* accuracy improved: {} -> {} at {}'.format(highest_acc, acc, get_time()))
                highest_acc = acc

    def eval(self, mode: str, model: AttenModel):
        model.eval()
        sigmoid_func = nn.Sigmoid()
        dis_comp = nn.PairwiseDistance(p=args.norm)
        with torch.no_grad():
            eval_data = QAData(qid2que=self.valid_qid2que if mode == 'valid' else self.test_qid2que,
                               qid2embeds=self.valid_qid2embeds if mode == 'valid' else self.test_qid2embeds,
                               max_pos_neg_pairs=args.max_pos_neg_pairs)
            eval_loader = DataLoader(dataset=eval_data, collate_fn=collate_fn, batch_size=1, shuffle=False)
            all_ranks = []
            r = model.rel_enc()
            all_precision, all_recall, all_f1, all_hits = [], [], [], []
            for batch_id, batch_data in enumerate(eval_loader):
                _, num_subg_ents, edge_index, edge_attr, loc_tops, loc_ans, que_embeds, _, _ = batch_data[0]
                x, fin_que_embed = model(que_embeds=que_embeds, r=r, num_subg_ents=num_subg_ents,
                                         edge_index=edge_index, edge_attr=edge_attr, loc_tops=loc_tops)
                all_dis = torch.norm(x - fin_que_embed, p=args.norm, dim=1)  # size: (num_subg_ents)

                min_ans_dis = torch.min(all_dis[loc_ans])
                all_dis[loc_ans] = 999999.
                # noinspection PyTypeChecker
                all_ranks.append(torch.nonzero(all_dis <= min_ans_dis, as_tuple=True)[0].size()[0] + 1)

                # candidate answer selection
                all_dis = sigmoid_func(dis_comp(x, fin_que_embed))  # size: (num_subg_ents)
                ent2dis = {ent: val for ent, val in zip(list(range(all_dis.size(0))), all_dis)}

                final_ans, final_dis = None, 999.
                for ans in ent2dis.keys():
                    if ent2dis[ans] < final_dis:
                        final_ans = ans
                        final_dis = ent2dis[ans]

                pred_ans = set()
                for ans in ent2dis.keys():
                    if ent2dis[ans] <= args.threshold * final_dis:
                        pred_ans.add(ans)

                precision, recall, f1, hits = cal_eval_metric(best_pred=final_ans, preds=list(pred_ans), answers=loc_ans)
                all_precision.append(precision)
                all_recall.append(recall)
                all_f1.append(f1)
                all_hits.append(hits)

            all_ranks = torch.LongTensor(all_ranks)

            results = {'hits@{}'.format(_): compute_hits(all_ranks, _, args.num_valid if mode == 'valid' else args.num_test)
                       for _ in [1, 3, 10, 20, 50, 100, 500, 1000]}  # hits w.r.t all questions
            results['mr'] = torch.mean(all_ranks.float())  # the mean rank w.r.t only answerable questions

            print('\t* precision: {}, recall: {}, f1: {}'.format(
                sum(all_precision) / args.num_test,
                sum(all_recall) / args.num_test,
                sum(all_f1) / args.num_test,
                # sum(all_hits) / args.num_test
            ))

            print('\t* ' + ', '.join(['{}: {:.7f}'.format(key, value) for key, value in results.items()]))

            return results

    def test(self):
        print('### Testing')
        atten_model = AttenModel(rel2embeds=self.rel2embeds,
                                 in_dim=args.in_dim, hid_dim=args.hid_dim,
                                 num_layers=args.num_gcn_layers,
                                 dropout=args.dropout)
        atten_model.cuda()
        if args.num_epochs == 0 and args.pre_timestamp:
            ckp_path = os.path.join(args.out_path.replace(args.timestamp, args.pre_timestamp), 'best.tar')
        else:
            ckp_path = os.path.join(args.out_path, 'best.tar')
        ckp = torch.load(ckp_path, map_location=torch.device(torch.cuda.current_device()))
        atten_model.load_state_dict(ckp['model_state'])
        print('* loaded the pre-trained model from {} which was cached at {}'.format(ckp_path, ckp['timestamp']))
        self.eval(mode='test', model=atten_model)


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, default='WebQuestionsSP',
                            help='specify which dataset to use',
                            choices={'CWQ', 'WebQuestionsSP', 'pq-2hop', 'pq-3hop', 'pql-2hop', 'pql-3hop'})

    arg_parser.add_argument('--timestamp', type=str, default=datetime.now().strftime('%Y.%m.%d.%H.%M.%S'))
    arg_parser.add_argument('--pre_timestamp', type=str, default=None,
                            help='timestamp of pre-trained model')

    arg_parser.add_argument('--in_path', type=str, default='datasets/{}/in_path',
                            help='path of preprocessed data (output of the que_prep.py script)')
    arg_parser.add_argument('--out_path', type=str, default='datasets/{}/out_path',
                            help='path of trained models (output of this script)')

    arg_parser.add_argument('--num_epochs', type=int, default=0,
                            help='number of training epochs')
    arg_parser.add_argument('--num_gcn_layers', type=int, default=3,
                            help='number of GCN layers in the graph encoder')
    arg_parser.add_argument('--batch_size', type=int, default=16,
                            help='number of questions in each batch')
    arg_parser.add_argument('--lr', type=float, default=5e-4,
                            help='learning rate')
    arg_parser.add_argument('--weight_decay', type=float, default=1e-5,
                            help='weight_decay')
    arg_parser.add_argument('--dropout', type=float, default=0.,
                            help='dropout rate in the graph encoder')

    arg_parser.add_argument('--in_dim', type=int, default=768,
                            help='dimension of pre-encoded question/relation tokens')
    arg_parser.add_argument('--hid_dim', type=int, default=256,
                            help='dimension of embeddings')

    arg_parser.add_argument('--norm', type=int, default=2,
                            help='order of norm when computing embedding distances')
    arg_parser.add_argument('--margin', type=float, default=1.,
                            help='margin for triplet distance loss')
    arg_parser.add_argument('--loss_red', type=str, default='mean',
                            help='reduction in loss computation')
    arg_parser.add_argument('--max_pos_neg_pairs', type=int, default=50000,
                            help='maximum number of positive-negative answer pairs to consider in loss function')
    arg_parser.add_argument('--threshold', type=float, default=1.0)

    arg_parser.add_argument('--num_valid', type=int, default=0,
                            help='the total number of validation questions for computing eval metrics'
                                 'questions unanswerable due to subgraph preparation errors are not included in the prepared data')
    arg_parser.add_argument('--num_test', type=int, default=0,
                            help='the total number of test questions for computing eval metrics'
                                 'questions unanswerable due to subgraph preparation errors are not included in the prepared data')

    args = arg_parser.parse_args()

    args.in_path = args.in_path.format(args.dataset)
    args.out_path = os.path.join(args.out_path.format(args.dataset), args.timestamp)

    args.num_valid = num_total_ques[args.dataset]['dev']
    args.num_test = num_total_ques[args.dataset]['test']

    print('\n## GNN2R Step-I - {} - {}'.format(args.dataset, args.timestamp))
    for k, v in vars(args).items():
        print('* {}: {}'.format(k, v))

    atten_train = AttenTrain()
    if args.num_epochs > 0:
        atten_train.train()
    atten_train.test()