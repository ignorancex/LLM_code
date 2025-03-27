import utils
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from scipy.optimize import linear_sum_assignment


class CloudCluster:
    def __init__(self, cpoint, nbs, ids, k=0, sim2c=None):
        self.k = k
        self.point_num = len(ids)
        self.points = ids
        self.center = cpoint
        self.c_neighbors = nbs
        self.sim2c = sim2c


def embed_filter(static_embedding):
    special_tokens = ["[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    number_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    index_voc, voc_index, embeddings = static_embedding
    filter_out = ''


def has_num(w):
    number_tokens = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    for num in number_tokens:
        if num in w:
            return True
    return False


def clear_sim_words(keyword, embed, simsize, static_embeds, language=None):
    index_voc, voc_index, embeddings = static_embeds
    special_tokens = [keyword, "[PAD]", "[CLS]", "[SEP]", "[UNK]", "[MASK]"]
    if language is None:
        stop_words = utils.stop_words(language='en')
    else:
        stop_words = utils.stop_words(language=language)
    sizegap = 20
    words = utils.most_sim_words(embed, simsize + sizegap, voc_index, embeddings, willprint=False)
    poly_sims = []
    for i, w in enumerate(words):
        if w[0] in special_tokens:
            continue
        if w[0] in stop_words:
            continue
        if len(w[0]) < 2:
            continue
        if has_num(w[0]):
            continue
        if len(poly_sims) < simsize:
            poly_sims.append(w[0])

    return poly_sims


def eva_polysim_linear_assignment(polyset1, polyset2, static_embeds):
    index_voc, voc_index, embeddings = static_embeds
    weights = np.zeros((len(polyset1), len(polyset2)))
    for i, w_row in enumerate(polyset1):
        for j, w_col in enumerate(polyset2):
            weight = utils.word_cosine_similarity(w_row, w_col, voc_index, embeddings)
            weights[i, j] = weight
    row_ind, col_ind = linear_sum_assignment(weights, maximize=True)
    mean = weights[row_ind, col_ind].mean()

    return mean


def neighbor_similarity(smset1, smset2, static_embeds):
    polytree_sim = eva_polysim_linear_assignment(smset1, smset2, static_embeds)
    return polytree_sim


def cloud_distance(cloud, static_embeds, keyword, k=12):
    nbs_cloud = []
    sim_matrix = np.ones((len(cloud), len(cloud)), dtype='float32')
    for i, point in enumerate(cloud):
        nbs = clear_sim_words(keyword, point, k, static_embeds)
        for j, nbs_ in enumerate(nbs_cloud):
            sim = neighbor_similarity(nbs, nbs_, static_embeds)
            sim_matrix[i, j] = sim
            sim_matrix[j, i] = sim
        nbs_cloud.append(nbs)
    return sim_matrix, nbs_cloud


def cloud_distance_mp_unit(pairs, nbs_cloud, static_embeds, pid, return_dict):
    sim_cloud = {}
    for pair in tqdm(pairs):
        nbs = nbs_cloud[pair[0]]
        nbs_ = nbs_cloud[pair[1]]
        sim = neighbor_similarity(nbs, nbs_, static_embeds)
        sim_cloud[pair] = sim
    return_dict[pid] = sim_cloud


def cloud_distance_mp(cloud, static_embeds, keyword, k=12, p_number=8, language=None):
    nbs_cloud = []
    pairs = []
    sim_matrix = np.ones((len(cloud), len(cloud)), dtype='float32')
    clouds = []
    for i, point in enumerate(cloud):
        clouds.append((i, point))
    for point in tqdm(clouds):
        i, emb = point
        nbs = clear_sim_words(keyword, emb, k, static_embeds, language=language)
        for j, nbs_ in enumerate(nbs_cloud):
            pairs.append((i, j))
        nbs_cloud.append(nbs)

    manager = mp.Manager()
    return_dict = manager.dict()

    procs = []
    step = len(pairs) // p_number
    for pid in range(p_number):
        sub_pairs = pairs[pid*step: (pid+1)*step] if pid < p_number-1 else pairs[pid*step:]
        p = mp.Process(target=cloud_distance_mp_unit, args=(
            sub_pairs, nbs_cloud, static_embeds, pid, return_dict))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    for pid in range(p_number):
        pairs_dict = return_dict[pid]
        for pair in pairs_dict:
            sim_matrix[pair[0], pair[1]] = pairs_dict[pair]
            sim_matrix[pair[1], pair[0]] = pairs_dict[pair]
    return sim_matrix, nbs_cloud


def gen_cloud_distance(cloud, static_embeds, keyword, language, time, k=12, save_nbs=True, usecache=True):
    matrix_path = './.cache/{}/words/{}/cloud_sim_matrix/k{}_cloud_matrix_t{}.npy'.format(language, keyword, k, time)
    nbs_path = './.cache/{}/words/{}/cloud_sim_matrix/k{}_cloud_nbs_t{}.pkl'.format(language, keyword, k, time)
    if usecache and utils.exists(matrix_path):
        return load_cloud_prior(matrix_path, nbs_path, load_nbs=save_nbs)

    sim_matrix, nbs_cloud = cloud_distance_mp(cloud, static_embeds, keyword, k=k, language=language)
    utils.create_filepath_dir(matrix_path)
    np.save(matrix_path, sim_matrix)
    if save_nbs:
        utils.save_to_disk(nbs_path, nbs_cloud)
    return sim_matrix, nbs_cloud


def load_cloud_prior(matrix_path, nbs_path=None, load_nbs=True):
    utils.create_filepath_dir(matrix_path)
    sim_matrix = np.load(matrix_path)

    nbs_cloud = None
    if load_nbs:
        nbs_cloud = utils.load_from_disk(nbs_path)

    return sim_matrix, nbs_cloud


def points_set_distances(clusters1, clusters2, sim_matrix, static_embeds=None):
    if sim_matrix is None:
        return neighbor_similarity(clusters1.c_neighbors, clusters2.c_neighbors, static_embeds)
    points1 = clusters1.points
    points2 = clusters2.points
    if len(points1)*len(points2) < 1000 or static_embeds is None:
        sims = np.zeros((len(points1), len(points2)), dtype='float32')
        for m, i in enumerate(points1):
            for n, j in enumerate(points2):
                sims[m, n] = sim_matrix[i, j]
        return sims.flatten().mean()
    else:
        return neighbor_similarity(clusters1.c_neighbors, clusters2.c_neighbors, static_embeds)


def clusters_distances(clusters, sim_matrix, rip_dist=None, static_embeds=None):
    cluster_num = len(clusters)
    cluster_dist = np.zeros((cluster_num, cluster_num), dtype='float32')
    if rip_dist is not None:
        cluster_dist[:rip_dist.shape[0], :rip_dist.shape[1]] = rip_dist
        i = cluster_num - 1
        for j in range(cluster_num-1):
            csim = points_set_distances(clusters[i], clusters[j], sim_matrix, static_embeds)
            cluster_dist[i, j] = csim
            cluster_dist[j, i] = csim
        return cluster_dist
    else:
        for i in range(cluster_num):
            for j in range(i + 1, cluster_num):
                csim = points_set_distances(clusters[i], clusters[j], sim_matrix, static_embeds)
                cluster_dist[i, j] = csim
                cluster_dist[j, i] = csim
    return cluster_dist


def merge_cluster(cluster1: CloudCluster, cluster2: CloudCluster, cloud, static_embeds, keyword, language=None):
    k = cluster1.k
    ids = cluster1.points + cluster2.points
    intersection = set(cluster1.points) & set(cluster2.points)
    assert len(intersection) == 0
    assert cluster1.k == cluster2.k
    assert len(ids) == len(cluster1.points) + len(cluster2.points)
    points_embed = [cloud[i] for i in ids]
    cpoint = np.mean(points_embed, axis=0)
    nbs = clear_sim_words(keyword, cpoint, k, static_embeds, language=language)
    sim2c = [utils.cosine_similarity(cloud[i], cpoint) for i in ids]
    new_cluster = CloudCluster(cpoint, nbs, ids, k, sim2c)
    return new_cluster


def merge_cluster_once(clusters, cluster_simdist, cloud, static_embeds, keyword, drop=False, language=None):
    arg = np.argmax(cluster_simdist)
    arg = np.unravel_index(arg, cluster_simdist.shape)
    drop_st = []
    if not drop:
        m_cluster = merge_cluster(clusters[arg[0]], clusters[arg[1]],
                                  cloud, static_embeds, keyword, language=language)
    else:
        clu0 = clusters[arg[0]]
        clu1 = clusters[arg[1]]
        if clu0.point_num > clu1.point_num and clu0.point_num/clu1.point_num > 2:
            m_cluster = clusters[arg[0]]
            drop_st += [clusters[arg[1]]]
        elif clu1.point_num > clu0.point_num and clu1.point_num/clu0.point_num > 2:
            m_cluster = clusters[arg[1]]
            drop_st += [clusters[arg[0]]]
        else:
            m_cluster = merge_cluster(clusters[arg[0]], clusters[arg[1]],
                                      cloud, static_embeds, keyword, language=language)

    new_clusters = []
    for i, clu in enumerate(clusters):
        if i == arg[0] or i == arg[1]:
            continue
        new_clusters.append(clu)
    new_clusters.append(m_cluster)
    rip_dist = cluster_simdist
    rip_dist = np.delete(rip_dist, [arg[0], arg[1]], axis=0)
    rip_dist = np.delete(rip_dist, [arg[0], arg[1]], axis=1)
    return new_clusters, rip_dist, drop_st


def cloud_clusters_reduce(clusters, cluster_simdist, sim_matrix, cloud, static_embeds, keyword,
                          rt=0.9, drop=False, language=None):
    drop_semantic = []
    while np.max(cluster_simdist) > rt:
        clusters, rip_dist, drop_st =\
            merge_cluster_once(clusters, cluster_simdist, cloud, static_embeds, keyword, drop=drop, language=language)
        cluster_simdist = clusters_distances(clusters, sim_matrix, rip_dist, static_embeds)
        drop_semantic += drop_st

    return clusters, cluster_simdist, drop_semantic


def subsmantic_from_clusters(clusters, cluster_simdist, sim_matrix,
                             cloud, static_embeds, keyword, sct=0.6, drop=True):
    hf_clusters = []
    lf_semantic = []

    hf_clusters = clusters

    hf_cluster_simdist = clusters_distances(hf_clusters, sim_matrix, None, static_embeds)

    hf_semantic, polysemy_simdist, drop_semantic = \
        cloud_clusters_reduce(hf_clusters, hf_cluster_simdist, sim_matrix,
                              cloud, static_embeds, keyword, rt=sct, drop=drop)
    lf_semantic += drop_semantic
    return hf_semantic, lf_semantic


def sub_semantic_similarity_matrix(subsm_set1, subsm_set2, static_embeds):
    sim_matrix = np.zeros((len(subsm_set1), len(subsm_set2)))
    for m, subsm1 in enumerate(subsm_set1):
        for n, subsm2 in enumerate(subsm_set2):
            sim = neighbor_similarity(subsm1, subsm2, static_embeds)
            sim_matrix[m, n] = sim
    return sim_matrix


def sim_matrix(hf_semantic1, hf_semantic2, static_embeds):
    hf_sms1 = [hf.c_neighbors for hf in hf_semantic1]
    hf_sms2 = [hf.c_neighbors for hf in hf_semantic2]
    hf_sims = sub_semantic_similarity_matrix(hf_sms1, hf_sms2, static_embeds)
    return hf_sims


def cloud_cluster_rt(sim_matrix, nbs_cloud, cloud, static_embeds, keyword, rt, k, time, language, usecache=True):
    rt_clusters_pth = './.cache/{}/words/{}/cloud_sim_matrix/rt{}/k{}_rt_clusters_t{}.pkl'. \
        format(language, keyword, rt, k, time)
    rt_cluster_simdist_pth = './.cache/{}/words/{}/cloud_sim_matrix/rt{}/k{}_rt_cluster_simdist_t{}.pkl'. \
        format(language, keyword, rt, k, time)
    clusters = utils.load_from_disk(rt_clusters_pth)
    cluster_simdist = utils.load_from_disk(rt_cluster_simdist_pth)
    if usecache and clusters is not None and cluster_simdist is not None:
        return clusters, cluster_simdist

    clusters = []
    for i, point in enumerate(cloud):
        clusters.append(CloudCluster(point, nbs_cloud[i], [i], k, [1]))

    cluster_simdist = sim_matrix.copy()
    row, col = np.diag_indices_from(cluster_simdist)
    cluster_simdist[row, col] = 0

    clusters, cluster_simdist, _ = \
        cloud_clusters_reduce(clusters, cluster_simdist, sim_matrix,
                              cloud, static_embeds, keyword, rt=rt, language=language)

    utils.save_to_disk(rt_clusters_pth, clusters)
    utils.save_to_disk(rt_cluster_simdist_pth, cluster_simdist)
    return clusters, cluster_simdist


def cloud_cluster(cloud, static_embeds, keyword, language, k,
                  usecache=False, subsample=True, t=0.8, cache_time=None):
    rt = t
    sct = t

    subsample_length = 400
    if subsample:
        cloud = cloud[:subsample_length]

    sim_matrix, nbs_cloud = \
        gen_cloud_distance(cloud, static_embeds, keyword, language, time=cache_time, k=k, usecache=usecache)

    clusters, cluster_simdist =\
        cloud_cluster_rt(sim_matrix, nbs_cloud, cloud, static_embeds, keyword, rt, k, cache_time, language, usecache=usecache)

    hf_semantic, lf_semantic = \
        subsmantic_from_clusters(clusters, cluster_simdist, sim_matrix,
                                 cloud, static_embeds, keyword, sct=sct)

    return hf_semantic, lf_semantic


