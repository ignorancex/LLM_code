import re
import torch
import pickle
import argparse
import graph_tool
from torch import LongTensor
from datetime import datetime
from graph_tool import topology
from torch_scatter import scatter
from torch.utils.data import Dataset
from itertools import product, chain


def cal_eval_metric(best_pred, preds, answers):
    """
    Code from GraftNet (https://github.com/haitian-sun/GraftNet).
    """
    correct, total = 0.0, 0.0
    for entity in preds:
        if entity in answers:
            correct += 1
        total += 1
    if len(answers) == 0:
        if total == 0:
            return 1.0, 1.0, 1.0, 1.0 # precision, recall, f1, hits
        else:
            return 0.0, 1.0, 0.0, 1.0 # precision, recall, f1, hits
    else:
        hits = float(best_pred in answers)
        if total == 0:
            return 1.0, 0.0, 0.0, hits # precision, recall, f1, hits
        else:
            precision, recall = correct / total, correct / len(answers)
            f1 = 2.0 / (1.0 / precision + 1.0 / recall) if precision != 0 and recall != 0 else 0.0
            return precision, recall, f1, hits


def get_time() -> str:
    return datetime.now().strftime("%H:%M:%S %Y-%m-%d")


def write_obj(obj: object, file_path: str) -> None:
    print("\t* dumping to {} at {} ...".format(file_path, get_time()))
    with open(file=file_path, mode="wb") as f:
        pickle.dump(obj=obj, file=f, protocol=4)


def read_obj(file_path: str) -> dict:
    print("\t* loading from {} at {} ...".format(file_path, get_time()))
    with open(file=file_path, mode="rb") as f:
        obj = pickle.load(file=f)
    return obj


def compute_hits(ranks: LongTensor, hit: int, total_num: int) -> float:
    # noinspection PyTypeChecker
    return torch.nonzero(ranks <= hit, as_tuple=True)[0].size()[0] / total_num


def collate_fn(batch_data):
    return batch_data


class QAData(Dataset):
    def __init__(self, qid2que: dict, qid2embeds: dict, max_pos_neg_pairs: int):
        super(QAData, self).__init__()
        self.qid2que = qid2que
        self.qid2embeds = qid2embeds
        self.max_pos_neg_pairs = max_pos_neg_pairs

    def __len__(self):
        return len(self.qid2que)

    def __getitem__(self, item):
        q_data = self.qid2que[item]
        que_embeds = self.qid2embeds[item].cuda()

        # create a small local graph for each question
        glo_ents = torch.unique(torch.LongTensor(q_data['glo_ents']).cuda())
        num_ents = glo_ents.size()[0]
        loc_ents = torch.arange(num_ents).cuda()
        glo2loc = scatter(src=loc_ents, index=glo_ents, reduce='mean', dim=0)

        edge_index = glo2loc[torch.LongTensor(q_data['edge_index']).cuda()]
        edge_attr = torch.LongTensor(q_data['edge_attr']).cuda()

        # add inverse edges
        edge_index = torch.cat([edge_index, torch.stack([edge_index[1], edge_index[0]], dim=0)], dim=1)
        edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

        # create local topic entities
        if not isinstance(q_data['top_ents'], list):
            q_data['top_ents'] = [q_data['top_ents']]
        top_ents = torch.LongTensor(q_data['top_ents']).cuda()
        loc_tops = glo2loc[top_ents]

        # create positive and negative pairs
        loc_ans = glo2loc[torch.LongTensor(q_data['glo_ans']).cuda()]  # size: (num_ans,)

        # collect non-answer entities
        loc_ents[loc_ans] = num_ents
        neg_loc_ans = loc_ents[loc_ents != num_ents]
        num_ans, num_neg = loc_ans.size()[0], neg_loc_ans.size()[0]
        assert num_ans > 0 and num_neg > 0, print('#positive answer: {}, #negative answer: {}'.format(num_ans, num_neg))

        # do random sampling if the number of possible positive-negative answer pairs is greater than a number
        if num_ans * num_neg >= self.max_pos_neg_pairs:
            sam_num_neg = self.max_pos_neg_pairs // num_ans
            neg_loc_ans = neg_loc_ans[torch.randperm(num_neg).cuda()[:sam_num_neg]]
            num_neg = sam_num_neg

        # collate positive-negative answer pairs
        pos_loc_ans = torch.flatten(loc_ans.unsqueeze(1).expand(-1, num_neg))
        neg_loc_ans = torch.flatten(neg_loc_ans.expand(num_ans, -1))

        return item, num_ents, edge_index, edge_attr, loc_tops, loc_ans, que_embeds, pos_loc_ans, neg_loc_ans


def rewrite_subg(subg: tuple, que: str, id2ent: dict, ent2label: dict, id2rel: dict):
    # determine the w-word to use for the given question
    num_rels = len(id2rel)
    exp_chunks = []
    match_obj = re.search(r'(what)|(who)|(where)|(when)|(which)|(whom)|(whose)', que)
    if match_obj:
        exp_chunks.append(match_obj.group(0))
    else:
        exp_chunks.append('what')
    # rewrite the given subgraph into a natural language expression
    path_end_flag = False
    int_ent_flag = False
    for element in subg:
        if path_end_flag:
            exp_chunks.append('and')
            path_end_flag = False
            int_ent_flag = False

        if element < 3 * num_rels:
            if int_ent_flag:
                exp_chunks.append('that')
            if element < num_rels:
                exp_chunks.append('has the {}'.format(id2rel[element].split('.')[-1].replace('_', ' ')))
            else:
                exp_chunks.append('is the {} of'.format(id2rel[element - num_rels].split('.')[-1].replace('_', ' ')))
            exp_chunks.append('an entity')
            int_ent_flag = True
        else:
            exp_chunks = exp_chunks[:-1]
            exp_chunks.append(ent2label[id2ent[element - 3 * num_rels]])
            path_end_flag = True
    subg_exp = ' '.join(exp_chunks)
    return subg_exp


def extract_subg_pq(ordered_tops: list, ans: int, gt_graph: graph_tool.Graph, cutoff: int, num_rels: int,
                 tail_rel2head: dict):

    # extract reasoning subgraphs given a candidate answer and topic entities
    all_subgs = []  # unique relation paths for all topic entities
    all_subgs_ans = []  # corresponding answers of all extracted subgraphs
    all_subgs_subgs = []  # corresponding explicit subgraphs of relation paths

    for top in ordered_tops:

        top_reasoning_paths = []  # all unique reasoning paths for each topic entity
        top_reasoning_path_ans = []
        top_reasoning_path_subgs = []

        for edge_path in topology.all_paths(g=gt_graph, source=gt_graph.vertex(ans),
                                            target=gt_graph.vertex(top), cutoff=cutoff, edges=True):

            tmp_rel_path = []  # one reasoning path between a topic entity and the given answer
            tmp_rel_path_ans = set()  # corresponding answers of reasoning paths
            tmp_rel_path_subgs = set()  # corresponding triples of reasoning paths

            ans_flag = True
            for edge in edge_path:

                head = int(edge.source())

                rel = gt_graph.edge_properties['edge_attr'][edge]

                tail = int(edge.target())

                tmp_rel_path.append(rel)

                if rel < num_rels:
                    tmp_rel_path_subgs.update([(head, rel, tail)])
                else:
                    tmp_rel_path_subgs.update([(tail, rel - num_rels, head)])

                if ans_flag:
                    tmp_rel_path_ans.update(tail_rel2head[(tail, rel)])
                    ans_flag = False

            tmp_rel_path.append(top + 3 * num_rels)
            tmp_rel_path = tuple(tmp_rel_path)  # each candidate path is a tuple of relations and topic entity + 3 * num_rels

            if tmp_rel_path not in top_reasoning_paths:
                top_reasoning_paths.append(tmp_rel_path)
                top_reasoning_path_ans.append(tmp_rel_path_ans)
                top_reasoning_path_subgs.append(tmp_rel_path_subgs)
            else:
                top_reasoning_path_ans[top_reasoning_paths.index(tmp_rel_path)].update(tmp_rel_path_ans)
                top_reasoning_path_subgs[top_reasoning_paths.index(tmp_rel_path)].update(tmp_rel_path_subgs)

        if len(top_reasoning_paths) > 0:
            all_subgs.append(top_reasoning_paths)
            all_subgs_ans.append(top_reasoning_path_ans)
            all_subgs_subgs.append(top_reasoning_path_subgs)

    if len(all_subgs) > 1:
        all_subgs = list(product(*all_subgs))  # combinations of paths to different topic entities
        all_subgs = [tuple(chain(*_)) for _ in all_subgs]
        all_subgs_ans = list(product(*all_subgs_ans))
        all_subgs_ans = [set.intersection(*_) for _ in all_subgs_ans]
        all_subgs_subgs = list(product(*all_subgs_subgs))
        all_subgs_subgs = [set.intersection(*_) for _ in all_subgs_subgs]
    elif len(all_subgs) == 1:
        all_subgs = all_subgs[0]
        all_subgs_ans = all_subgs_ans[0]
        all_subgs_subgs = all_subgs_subgs[0]

    return all_subgs, all_subgs_ans, all_subgs_subgs


def extract_subg(ordered_tops: list, ans: int, gt_graph: graph_tool.Graph, cutoff: int, num_rels: int,
                 tail_rel2head: dict):
    # extract reasoning subgraphs given a candidate answer and topic entities
    all_subgs = []  # unique relation paths for all topic entities
    all_subgs_ans = []  # corresponding answers of all extracted subgraphs

    for top in ordered_tops:
        top_reasoning_paths = []  # all unique reasoning paths for each topic entity
        top_reasoning_path_ans = []

        for edge_path in topology.all_paths(g=gt_graph, source=gt_graph.vertex(ans),
                                            target=gt_graph.vertex(top), cutoff=cutoff, edges=True):
            tmp_rel_path = []  # one reasoning path between a topic entity and the given answer
            tmp_rel_path_ans = set()  # corresponding answers of reasoning paths

            ans_flag = True
            for edge in edge_path:
                tmp_rel_path.append(gt_graph.edge_properties['edge_attr'][edge])
                # tmp_rel_path.append(int(edge.target()))
                if ans_flag:
                    first_tail, first_rel = int(edge.target()), gt_graph.edge_properties['edge_attr'][edge]
                    tmp_rel_path_ans.update(tail_rel2head[(first_tail, first_rel)])
                    ans_flag = False

            tmp_rel_path.append(top + 3 * num_rels)
            tmp_rel_path = tuple(tmp_rel_path)  # each candidate path is a tuple of relations + topic entity

            if tmp_rel_path not in top_reasoning_paths:
                top_reasoning_paths.append(tmp_rel_path)
                top_reasoning_path_ans.append(tmp_rel_path_ans)
            else:
                top_reasoning_path_ans[top_reasoning_paths.index(tmp_rel_path)].update(tmp_rel_path_ans)

        if len(top_reasoning_paths) > 0:
            all_subgs.append(top_reasoning_paths)
            all_subgs_ans.append(top_reasoning_path_ans)

    if len(all_subgs) > 1:
        all_subgs = list(product(*all_subgs))  # combinations of paths to different topic entities
        all_subgs = [tuple(chain(*_)) for _ in all_subgs]
        all_subgs_ans = list(product(*all_subgs_ans))
        all_subgs_ans = [set.intersection(*_) for _ in all_subgs_ans]
    elif len(all_subgs) == 1:
        all_subgs = all_subgs[0]
        all_subgs_ans = all_subgs_ans[0]

    return all_subgs, all_subgs_ans


class ExpData(Dataset):
    def __init__(self, qid2que: dict, qid2subgs: dict, neg_size: int, mode: str):
        super(ExpData, self).__init__()
        self.qid2que = qid2que
        self.qid2subgs = qid2subgs
        self.neg_size = neg_size
        self.mode = mode

    def __len__(self):
        return len(self.qid2subgs)

    def __getitem__(self, item):
        q_data = self.qid2que[item]
        que, target_ans = q_data['question'], q_data['glo_ans']

        if len(self.qid2subgs[item]) == 8:
            [all_can_subgs, all_can_subg_exps, subg2ents, ent2dis,
            pos_subgs, pos_subg_exps, neg_subgs, neg_subg_exps] = self.qid2subgs[item]
            subg2explicit_subgs = None
        else:
            [all_can_subgs, all_can_subg_exps, subg2ents, ent2dis,
             pos_subgs, pos_subg_exps, neg_subgs, neg_subg_exps, subg2explicit_subgs] = self.qid2subgs[item]
        num_pos, num_neg = len(pos_subg_exps), len(neg_subg_exps)
        if num_pos > self.neg_size and self.mode == 'train':
            sample_ids = torch.randperm(n=num_pos)[:self.neg_size].tolist()
            pos_subg_exps = [pos_subg_exps[_] for _ in sample_ids]
            num_pos = self.neg_size
        if num_neg > self.neg_size and self.mode == 'train':
            sample_ids = torch.randperm(n=num_neg)[:self.neg_size].tolist()
            neg_subg_exps = [neg_subg_exps[_] for _ in sample_ids]
            num_neg = self.neg_size

        return [item, que, target_ans, all_can_subgs, all_can_subg_exps, subg2ents, ent2dis,
                num_pos, pos_subgs, pos_subg_exps, num_neg, neg_subgs, neg_subg_exps, subg2explicit_subgs]


class DeviceAction(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        if values == 'cpu':
            setattr(args, self.dest, torch.device('cpu'))
        else:
            setattr(args, self.dest, torch.device('cuda:{}'.format(values)))


num_total_ques = {
    # number of answerable questions + number of unanswerable questions
    'CWQ': {'dev': 2838 + 681, 'test': 2611 + 920, 'train': 22060 + 5579},
    'WebQuestionsSP': {'dev': 235 + 15, 'test': 1551 + 88, 'train': 2708 + 140},
    'pq-2hop': {'dev': 190, 'test': 191, 'train': 1527},
    'pq-3hop': {'dev': 519, 'test': 520, 'train': 4159},
    'pql-2hop': {'dev': 159, 'test': 159, 'train': 1276},
    'pql-3hop': {'dev': 103, 'test': 103, 'train': 825}
    }
