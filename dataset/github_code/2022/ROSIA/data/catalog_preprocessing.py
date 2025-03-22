import numpy as np
import sys
from numba import jit

def catalog_reading(catalog_path):
    ''' importing catalog and normalising it '''
    with open(catalog_path) as f:
        lines = f.read().splitlines()

    star_vec = np.zeros([len(lines), 3])
    star_mag = np.zeros([len(lines)])
    hip_id = np.zeros([len(lines)])
    for i in range(len(lines)):
        curr_cat_star = lines[i].split(' ')
        for j in range(3):
            star_vec[i, j] = float(curr_cat_star[j])

        star_mag[i] = float(curr_cat_star[3]) - 1.54
        hip_id[i] = curr_cat_star[4]

    return star_vec, star_mag, hip_id

@jit(nopython=True, nogil=True, cache=True)
def compute_k_nearest_distance(catalog_stars, k=2):
    # k = 2 for the triplet constraints
    angular_distance = np.ones((catalog_stars.shape[0], catalog_stars.shape[0])) * 1000 # any arbitrary large value for the diagonal components
    k_nearest_ang_dist = np.zeros((catalog_stars.shape[0], k))
    k_nearest_id = np.zeros((catalog_stars.shape[0], k))
    # first normalise them
    normed_catalog_stars = np.zeros((catalog_stars.shape[0], 3))
    for i in range(catalog_stars.shape[0]):
        normed_catalog_stars[i, :] = catalog_stars[i, :] / np.linalg.norm(catalog_stars[i, :])

    for i in range(catalog_stars.shape[0]):
        for j in range(catalog_stars.shape[0]):
            if i == j:
                continue # avoid the diagonal elements

            angular_distance[i, j] = np.arccos(np.dot(catalog_stars[i, :], catalog_stars[j, :]))

        k_nearest_ang_dist[i, :] = np.sort(angular_distance[i, :])[0:k]
        k_nearest_id[i, :] = np.argsort(angular_distance[i, :])[0:k]

    return k_nearest_ang_dist, k_nearest_id

def close_stars_removal(star_mag, knn_val, knn_id, ang_thres):
    # first retrieve knn_val less than thres
    removal_candidates = []
    for i in range(knn_val.shape[0]):
        for j in range(len(knn_val[i, :])):
            if knn_val[i, j] > ang_thres:
                continue # if the closest one is not smaller than ang_thres, the subsequent ones wont
            else:
                removal_candidates.append([i, knn_id[i, j], knn_val[i, j], star_mag[i], star_mag[int(knn_id[i, j])]])

    removal_candidates = np.array(removal_candidates)
    id_0 = removal_candidates[:, 0]
    id_1 = removal_candidates[:, 1]

    to_be_removed_entries = []
    for curr_idx in range(len(id_0)):
        if np.isin(curr_idx, to_be_removed_entries):
            continue

        curr_id0 = id_0[curr_idx]
        matched_ids = np.where(curr_id0 == id_1)[0]
        for m_id in matched_ids:
            if id_1[curr_idx] == id_0[m_id]:
                to_be_removed_entries.append(m_id)

    to_be_removed_entries = np.array(to_be_removed_entries, dtype=np.int64)
    removal_candidates = np.delete(removal_candidates, to_be_removed_entries, 0) # removing duplicated entries

    to_be_r_star_ids = []
    for i in range(removal_candidates.shape[0]):
        curr_cand = removal_candidates[i, :]
        if curr_cand[3] > curr_cand[4]: # remove fainter stars
            to_be_r_star_ids.append(curr_cand[1])
        else:
            to_be_r_star_ids.append(curr_cand[0])

    return to_be_r_star_ids




if __name__ == "__main__":
    catalog_path = sys.argv[1]
    blacklist_path = sys.argv[2]
    prepro_cat_path = sys.argv[3]
    mag_thres = float(sys.argv[4])
    dist_thres = float(sys.argv[5])
    k = int(sys.argv[6])
    # catalog_path = '/data/HIP_full_cat.txt'
    # blacklist_path = '/data/blacklist.txt'
    # prepro_cat_path = '/data/prepro_catalog.txt'

    # mag_thres = 6
    dist_thres = np.deg2rad(dist_thres) # to remove stars that are too close together (degree)
    # read catalog
    star_vec, star_mag, hip_id = catalog_reading(catalog_path)

    # filter based on mag threshold
    star_vec = star_vec[star_mag <= mag_thres, :]
    hip_id = hip_id[star_mag <= mag_thres]
    star_mag = star_mag[star_mag <= mag_thres]

    # compute closest angular distances
    k_nearest_ang_dist, k_nearest_id = compute_k_nearest_distance(star_vec, k=k)

    # filter based on closest distance
    blacklist = close_stars_removal(star_mag, k_nearest_ang_dist, k_nearest_id, dist_thres)
    blacklist = np.unique(blacklist)
    with open(blacklist_path, 'w') as f:
        for bl_id in blacklist:
            f.write('{}\n'.format(int(hip_id[int(bl_id)])))

    blacklist_hip_id = hip_id[np.int64(blacklist)]
    # iter through hip_id
    to_be_removed_idx = []
    for i in range(len(hip_id)):
        curr_hip_id = hip_id[i]
        if np.isin(curr_hip_id, blacklist_hip_id):
            to_be_removed_idx.append(i)

    to_be_removed_idx = np.array(to_be_removed_idx)

    star_vec = np.delete(star_vec, to_be_removed_idx, 0)
    star_mag = np.delete(star_mag, to_be_removed_idx, 0)
    hip_id = np.delete(hip_id, to_be_removed_idx, 0)
    k_nearest_ang_dist = np.delete(k_nearest_ang_dist, to_be_removed_idx, 0)

    # write the preprocessed catalog
    with open(prepro_cat_path, 'w') as f:
        for i in range(star_vec.shape[0]):
            f.write(str(star_vec[i, 0])+ " " + str(star_vec[i, 1]) + " " +  str(star_vec[i, 2]) + " " +  str(star_mag[i]) + " " + str(int(hip_id[i]))
                    + " " +  str(k_nearest_ang_dist[i, 0]) + " " + str(k_nearest_ang_dist[i, 1]) + "\n")




