#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Extract timestamp_ms, user posting, hashtag, and user mentioned.
In the process, correct timestamp_ms in rate limit message, remove duplicate tweets, and sort tweets chronologically.

Usage: python build_retweet_network.py
Input data files: ../data/[app_name]_out/complete_user_[app_name].txt, ../data/[app_name]_out/user_[app_name]_all.txt
Output data files: ../data/[sample|complete]_hashtag_user_stats.p, ../data/[sample|complete]_user_hashtag_stats.p
Time: ~30M
"""

import sys, os, logging, pickle
from collections import defaultdict
from tarjan import tarjan
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def main():
    timer = Timer()
    timer.start()

    app_name = 'cyberbullying'

    complete_uid_embed_dict = {}
    with open('../data/{0}_out/{0}_embed_user.txt'.format(app_name), 'r') as fin:
        for line in fin:
            embed, uid = line.rstrip().split(',')
            complete_uid_embed_dict[uid] = int(embed[1:])

    load_data_from_pickle = True
    for date_type in ['sample', 'complete']:
        if not load_data_from_pickle:
            if date_type == 'sample':
                user_datefile = open('../data/{0}_out/user_{0}_all.txt'.format(app_name), 'r')
            else:
                user_datefile = open('../data/{0}_out/complete_user_{0}.txt'.format(app_name), 'r')

            weighted_retweet_network = defaultdict(lambda: defaultdict(int))
            in_links_for_node = defaultdict(set)
            out_links_for_node = defaultdict(set)
            retweeted_uid_set = set()
            num_retweets_total = 0
            for line in user_datefile:
                tid, root_uid, reply_uid, retweeted_uid, quoted_uid = line.rstrip().split(',')
                if reply_uid == 'N':
                    if retweeted_uid != 'N':
                        weighted_retweet_network[complete_uid_embed_dict[root_uid]][complete_uid_embed_dict[retweeted_uid]] += 1
                        out_links_for_node[complete_uid_embed_dict[root_uid]].add(complete_uid_embed_dict[retweeted_uid])
                        in_links_for_node[complete_uid_embed_dict[retweeted_uid]].add(complete_uid_embed_dict[root_uid])
                        retweeted_uid_set.add(complete_uid_embed_dict[retweeted_uid])
                        num_retweets_total += 1
                    elif retweeted_uid == 'N' and quoted_uid != 'N':
                        weighted_retweet_network[complete_uid_embed_dict[root_uid]][complete_uid_embed_dict[quoted_uid]] += 1
                        out_links_for_node[complete_uid_embed_dict[root_uid]].add(complete_uid_embed_dict[quoted_uid])
                        in_links_for_node[complete_uid_embed_dict[quoted_uid]].add(complete_uid_embed_dict[root_uid])
                        retweeted_uid_set.add(complete_uid_embed_dict[quoted_uid])
                        num_retweets_total += 1

            unweighted_retweet_network = {uid: list(weighted_retweet_network[uid].keys()) for uid in weighted_retweet_network}
            pickle.dump(dict(weighted_retweet_network), open('../data/{0}_out/{1}_weighted_retweet_network.p'.format(app_name, date_type), 'wb'))
            pickle.dump(dict(unweighted_retweet_network), open('../data/{0}_out/{1}_unweighted_retweet_network.p'.format(app_name, date_type), 'wb'))
            pickle.dump(dict(in_links_for_node), open('../data/{0}_out/{1}_in_links_for_node.p'.format(app_name, date_type), 'wb'))
            pickle.dump(dict(out_links_for_node), open('../data/{0}_out/{1}_out_links_for_node.p'.format(app_name, date_type), 'wb'))
        else:
            weighted_retweet_network = pickle.load(open('../data/{0}_out/{1}_weighted_retweet_network.p'.format(app_name, date_type), 'rb'))
            unweighted_retweet_network = pickle.load(open('../data/{0}_out/{1}_unweighted_retweet_network.p'.format(app_name, date_type), 'rb'))
            in_links_for_node = pickle.load(open('../data/{0}_out/{1}_in_links_for_node.p'.format(app_name, date_type), 'rb'))
            out_links_for_node = pickle.load(open('../data/{0}_out/{1}_out_links_for_node.p'.format(app_name, date_type), 'rb'))

        print('>>> in {0} set'.format(date_type))
        print('{0:,} users have retweeted others'.format(len(unweighted_retweet_network)))
        retweeted_uid_set = set()
        for uid in unweighted_retweet_network:
            retweeted_uid_set.update(unweighted_retweet_network[uid])
        print('{0:,} users have been retweeted by others'.format(len(retweeted_uid_set)))
        total_uid_set = set(unweighted_retweet_network.keys())
        total_uid_set.update(retweeted_uid_set)
        num_total_user = len(total_uid_set)
        print('overall {0:,} users appear in the network'.format(num_total_user))
        num_retweets_total = sum([sum(weighted_retweet_network[uid].values()) for uid in weighted_retweet_network])
        print('they have produced {0:,} retweets\n'.format(num_retweets_total))

        print('time of loading data')
        timer.stop()

        # == == == == == == Part 4: Extract bow-tie structure == == == == == == #
        scc_content = tarjan(unweighted_retweet_network)
        scc_content = sorted(scc_content, key=lambda x: len(x), reverse=True)
        print('we have found {0} scc'.format(len(scc_content)))
        print('size of top 10 scc', [len(scc) for scc in scc_content[:10]])

        # create in nodes and out nodes set for scc
        scc_in_nodes = []
        scc_out_nodes = []
        for scc in scc_content:
            scc_in = set()
            scc_out = set()

            for uid in scc:
                if uid in in_links_for_node:
                    scc_in.update(set(in_links_for_node[uid]))
                if uid in out_links_for_node:
                    scc_out.update(set(out_links_for_node[uid]))
            scc_in_nodes.append(scc_in - set(scc))
            scc_out_nodes.append(scc_out - set(scc))

        fout1 = open('{0}_bowtie_users.txt'.format(date_type), 'w')
        # largest SCC
        largest_scc = scc_content.pop(0)
        largest_scc = set(largest_scc)
        lscc_in_nodes = scc_in_nodes.pop(0)
        lscc_out_nodes = scc_out_nodes.pop(0)
        print('>>> {0} {1:.3f}% users in the largest SCC'.format(len(largest_scc), len(largest_scc) / num_total_user * 100))
        print('>>> {0} users (outside LSCC) retweet LSCC, LSCC retweet {1} users'.format(len(lscc_in_nodes), len(lscc_out_nodes)))
        fout1.write('{0}\n'.format(','.join(map(str, largest_scc))))

        print('time of getting the LSCC component')
        timer.stop()

        # find IN component
        in_component = set()
        in_component_in = set()
        in_component_out = set()
        num_scc_in = 0
        to_visit_scc = []
        to_visit_in = []
        to_visit_out = []
        for scc_idx, scc in enumerate(scc_content):
            if len(scc_out_nodes[scc_idx].intersection(largest_scc)) > 0:
                in_component.update(scc)
                in_component_in.update(scc_in_nodes[scc_idx])
                in_component_out.update(scc_out_nodes[scc_idx])
                num_scc_in += 1
            else:
                to_visit_scc.append(scc)
                to_visit_in.append(scc_in_nodes[scc_idx])
                to_visit_out.append(scc_out_nodes[scc_idx])
        in_component_in = in_component_in - in_component
        in_component_out = in_component_out - in_component
        print('>>> {0} {1:.3f}% users in the IN component'.format(len(in_component), len(in_component) / num_total_user * 100))
        print('    {0} scc in the IN component'.format(num_scc_in))
        print('>>> {0} users (outside IN) retweet IN, IN retweet {1} users'.format(len(in_component_in), len(in_component_out)))
        fout1.write('{0}\n'.format(','.join(map(str, in_component))))

        print('time of getting the IN component')
        timer.stop()

        # find OUT component
        out_component = set()
        out_component_in = set()
        out_component_out = set()
        num_scc_out = 0
        to_visit_scc2 = []
        to_visit_in2 = []
        to_visit_out2 = []
        for scc_idx, scc in enumerate(to_visit_scc):
            if len(scc_in_nodes[scc_idx].intersection(largest_scc)) > 0:
                out_component.update(scc)
                out_component_in.update(to_visit_in[scc_idx])
                out_component_out.update(to_visit_out[scc_idx])
                num_scc_out += 1
            else:
                to_visit_scc2.append(scc)
                to_visit_in2.append(to_visit_in[scc_idx])
                to_visit_out2.append(to_visit_out[scc_idx])
        out_component_in = out_component_in - out_component
        out_component_out = out_component_out - out_component
        print('>>> {0} {1:.3f}% users in the OUT component'.format(len(out_component), len(out_component) / num_total_user * 100))
        print('    {0} scc in the OUT component'.format(num_scc_out))
        print('>>> {0} users (outside OUT) retweet OUT, OUT retweet {1} users'.format(len(out_component_in), len(out_component_out)))
        fout1.write('{0}\n'.format(','.join(map(str, out_component))))

        print('time of getting the OUT component')
        timer.stop()

        # find TUBE component
        tube_component = set()
        tube_component_in = set()
        tube_component_out = set()
        num_scc_tube = 0
        to_visit_scc3 = []
        to_visit_in3 = []
        to_visit_out3 = []
        for scc_idx, scc in enumerate(to_visit_scc2):
            if len(scc_in_nodes[scc_idx].intersection(in_component)) > 0 and len(scc_out_nodes[scc_idx].intersection(out_component)) > 0:
                tube_component.update(scc)
                tube_component_in.update(to_visit_in2[scc_idx])
                tube_component_out.update(to_visit_out2[scc_idx])
                num_scc_tube += 1
            else:
                to_visit_scc3.append(scc)
                to_visit_in3.append(to_visit_in2[scc_idx])
                to_visit_out3.append(to_visit_out2[scc_idx])
        tube_component_in = tube_component_in - tube_component
        tube_component_out = tube_component_out - tube_component
        print('>>> {0} {1:.3f}% users in the TUBE component'.format(len(tube_component), len(tube_component) / num_total_user * 100))
        print('    {0} scc in the TUBE component'.format(num_scc_tube))
        print('>>> {0} users (outside OUT) retweet TUBE, TUBE retweet {1} users'.format(len(tube_component_in), len(tube_component_out)))
        fout1.write('{0}\n'.format(','.join(map(str, tube_component))))

        print('time of getting the TUBE component')
        timer.stop()

        # find Tendrils component
        tendrils_component = set()
        tendrils_component_in = set()
        tendrils_component_out = set()
        num_scc_tendrils = 0
        to_visit_scc4 = []
        to_visit_in4 = []
        to_visit_out4 = []
        for scc_idx, scc in enumerate(to_visit_scc3):
            if len(scc_in_nodes[scc_idx].intersection(in_component)) > 0 or len(scc_out_nodes[scc_idx].intersection(out_component)) > 0:
                tendrils_component.update(scc)
                tendrils_component_in.update(to_visit_in3[scc_idx])
                tendrils_component_out.update(to_visit_out3[scc_idx])
                num_scc_tendrils += 1
            else:
                to_visit_scc4.append(scc)
                to_visit_in4.append(to_visit_in3[scc_idx])
                to_visit_out4.append(to_visit_out3[scc_idx])
        tendrils_component_in = tendrils_component_in - tendrils_component
        tendrils_component_out = tendrils_component_out - tendrils_component
        print('>>> {0} {1:.3f}% users in the Tendrils component'.format(len(tendrils_component), len(tendrils_component) / num_total_user * 100))
        print('    {0} scc in the Tendrils component'.format(num_scc_tendrils))
        print('>>> {0} users (outside OUT) retweet Tendrils, Tendrils retweet {1} users'.format(len(tendrils_component_in), len(tendrils_component_out)))
        fout1.write('{0}\n'.format(','.join(map(str, tendrils_component))))

        print('time of getting the Tendrils component')
        timer.stop()

        num_disconnected = len(total_uid_set) - len(largest_scc) - len(in_component) - len(out_component) - len(tendrils_component) - len(tube_component)
        disconnected = total_uid_set - set(largest_scc) - set(in_component) - set(out_component) - set(tendrils_component) - set(tube_component)
        print('>>> {0} {1:.3f}% users in the disconnected'.format(num_disconnected, num_disconnected / num_total_user * 100))
        print('>>> {0} users in the disconnected'.format(len(disconnected)))
        print('above 2 numbers should match')
        fout1.write('{0}\n'.format(','.join(map(str, disconnected))))

        fout1.close()

        # get the weight matrix
        uid_label_dict = {}
        with open('../data/{0}_out/{1}_user_bowtie_label.p'.format(app_name, date_type), 'w') as fout:
            for label, component in zip(range(6), [largest_scc, in_component, out_component, tube_component, tendrils_component, disconnected]):
                for uid in component:
                    uid_label_dict[uid] = label
                    fout.write('u{0},{1}\n'.format(uid, label))

        weight_matrix = np.zeros(shape=(6, 6))
        for root_uid in weighted_retweet_network:
            for retweeted_uid in weighted_retweet_network[root_uid]:
                weight = weighted_retweet_network[root_uid][retweeted_uid]
                weight_matrix[uid_label_dict[root_uid], uid_label_dict[retweeted_uid]] += weight
        print(weight_matrix)

        print('time of getting the weight matrix')
        timer.stop()


if __name__ == '__main__':
    main()
