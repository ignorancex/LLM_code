import sys, os, pickle
import numpy as np
from scipy import sparse
from tarjan import tarjan
from sknetwork.clustering import BiLouvain, BiSpectralClustering

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from utils.helper import Timer


def main():
    timer = Timer()
    timer.start()

    n_cluster = 6

    for date_type in ['sample', 'complete']:
        uid_hid_stats = pickle.load(open('./{0}_uid_hid_stats.p'.format(date_type), 'rb'))
        hid_uid_stats = pickle.load(open('./{0}_hid_uid_stats.p'.format(date_type), 'rb'))

        num_users = len(uid_hid_stats)
        num_hashtags = len(hid_uid_stats)
        print('in {0} set, {1} users, {2} hashtags'.format(date_type, num_users, num_hashtags))

        all_graph = {uid: [x[0] for x in lst[1:]] for uid, lst in uid_hid_stats.items()}
        rev_all_graph = {hid: [x[0] for x in lst[1:]] for hid, lst in hid_uid_stats.items()}
        all_graph.update(rev_all_graph)

        all_bipartites = tarjan(all_graph)
        all_bipartites = sorted(all_bipartites, key=lambda x: len(x), reverse=True)
        print('number of bipartites: {0}'.format(len(all_bipartites)))

        largest_bipartite = all_bipartites[0]
        largest_bipartite_users = [x for x in largest_bipartite if x.startswith('u')]
        largest_bipartite_hashtags = [x for x in largest_bipartite if x.startswith('h')]
        largest_bipartite_num_users = len(largest_bipartite_users)
        largest_bipartite_num_hashtags = len(largest_bipartite_hashtags)
        print('components of largest bipartite: {0} users; {1} hashtags'.format(largest_bipartite_num_users, largest_bipartite_num_hashtags))

        # B = nx.Graph()
        # # Add edges only between nodes of opposite node sets
        # bipartite_edges = []
        # for uid in largest_bipartite_users:
        #     for hid, cnt in uid_hid_stats[uid]:
        #         bipartite_edges.append((uid, hid, {'weight': cnt}))
        # B.add_edges_from(bipartite_edges)

        # re-embed
        new_user_embed = {uid: embed for embed, uid in enumerate(sorted(largest_bipartite_users))}
        new_embed_user = {v: k for k, v in new_user_embed.items()}
        new_hashtag_embed = {hid: embed for embed, hid in enumerate(sorted(largest_bipartite_hashtags))}
        new_embed_hashtag = {v: k for k, v in new_hashtag_embed.items()}

        bipartite_edges = {}
        for uid in largest_bipartite_users:
            bipartite_edges[new_user_embed[uid]] = []
            for hid, _ in uid_hid_stats[uid][1:]:
                bipartite_edges[new_user_embed[uid]].append(new_hashtag_embed[hid])
        row, col = [], []
        for key, item in bipartite_edges.items():
            row += [key] * len(item)
            col += item
        biadjacency = sparse.csr_matrix((np.ones(len(row), dtype=int), (row, col)))

        print('built the biadjacency')

        bispectral = BiSpectralClustering(n_clusters=n_cluster)
        print('running BiSpectralClustering...')
        bispectral.fit(biadjacency)
        print('completed BiSpectralClustering...')
        row_labels = bispectral.row_labels_
        col_labels = bispectral.col_labels_
        clusters = [[] for _ in range(n_cluster)]
        for user_idx, label in enumerate(row_labels):
            clusters[label].append(new_embed_user[user_idx])
        for hashtag_idx, label in enumerate(col_labels):
            clusters[label].append(new_embed_hashtag[hashtag_idx])
        for i in range(n_cluster):
            print('cluster {0}, size: {1}, num_user: {2}, num_hashtag: {3}'
                  .format(i, len(clusters[i]),
                          len([x for x in clusters[i] if x.startswith('u')]),
                          len([x for x in clusters[i] if x.startswith('h')])))
            with open('./{0}_cluster{1}.txt'.format(date_type, i), 'w') as fout:
                fout.write(','.join(clusters[i]))

        # bilouvain = BiLouvain()
        # print('running BiLouvain...')
        # bilouvain.fit(biadjacency)
        # print('completed BiLouvain...')
        # row_labels = bilouvain.row_labels_
        # col_labels = bilouvain.col_labels_
        # clusters = [[] for _ in range(n_cluster)]
        # for user_idx, label in enumerate(row_labels):
        #     clusters[label].append(new_embed_user[user_idx])
        # for hashtag_idx, label in enumerate(col_labels):
        #     clusters[label].append(new_embed_hashtag[hashtag_idx])
        # for i in range(n_cluster):
        #     print('cluster {0}, size: {1}, num_user: {2}, num_hashtag: {3}'
        #           .format(i, len(clusters[i]),
        #                   len([x for x in clusters[i] if x.startswith('u')]),
        #                   len([x for x in clusters[i] if x.startswith('h')])))


if __name__ == '__main__':
    main()
