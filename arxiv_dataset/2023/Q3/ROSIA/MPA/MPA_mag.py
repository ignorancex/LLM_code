import numpy as np
import sys
from numba import jit


def angular_distance(star_i, star_j):
    return np.real(np.arccos(np.dot(star_i, star_j)))

def create_feature_set(cat_dir, FOV):
    # new feature set has five elements
    feature_set = []
    catalog = []

    with open(cat_dir, "r") as f:
        for i, line in enumerate(f):
            line = list(map(float, line.split()))
            curr_vec = line[0:3]
            curr_mag = line[3]
            curr_hip = line[4]
            catalog.append((curr_hip, curr_vec, curr_mag))
    catalog = np.array(catalog, dtype=[('hip_id', object), ('position', object), ('magnitude', object)])

    for i in range(len(catalog)):
        for j in range(i + 1, len(catalog)):
            theta = angular_distance(catalog['position'][i], catalog['position'][j])
            if theta <= FOV:
                feature_set.append((catalog['hip_id'][i], catalog['hip_id'][j], theta, catalog['magnitude'][i], catalog['magnitude'][j]))

    return feature_set


def input_reading(input_dir):
    star_image = []
    ids_image = []
    mag_image = []

    with open(input_dir, 'r') as file:
        for line in file:
            line = list(map(float, line.split()))
            point = line[0:3]
            mag = line[3]
            id = line[4]

            star_image.append(list(point))
            ids_image.append(int(id))
            mag_image.append(float(mag))

    return star_image, ids_image, mag_image


# returns list of star position, pole position, angular distance
def img_feature_set(stars, magnitude, pole, pole_mag):
    img_features = []
    for star_idx, star in enumerate(stars):
        if star == pole:
            continue
        img_features.append([star, pole, angular_distance(star, pole), magnitude[star_idx], pole_mag])
    return img_features

# @jit(nopython=True, nogil=True, cache=True)
def pair_list(cat_features, img_features, eps_pi, eps_m):
    # check magnitude here
    # now features and distances are both len 5 vectors
    # img_feature = [star_pos1, star_pos2, star_theta, star_mag_pos1, star_mag_pos2]
    # cat_feature = [cat_id1, cat_id2, cat_theta, cat_mag_id1, cat_mag_id2]
    # pairs = [[star_pos1, star_pos2], [cat_id1, cat_id2], [star_mag_pos1, star_mag_pos2], [cat_theta]]
    # pairs = np.empty(len(img_features))
    pairs = []
    for img_feat_id, img_feature in enumerate(img_features):
        pairs.append([])
        pairs_img_pos_1, pairs_img_pos_2, pairs_cat_id_1, pairs_cat_id_2, pairs_star_pos_1, pairs_star_pos_2, pairs_cat_theta = \
            iter_catalog(img_feature[0], img_feature[1], img_feature[2], np.float64(img_feature[3]), np.float64(img_feature[4]), cat_features, eps_pi, eps_m)

        for i in range(len(pairs_img_pos_1)):
            pairs[-1].append(((pairs_img_pos_1[i], pairs_img_pos_2[i]), (int(pairs_cat_id_1[i]), int(pairs_cat_id_2[i])), (pairs_star_pos_1[i], pairs_star_pos_2[i]), (pairs_cat_theta[i])))

    return pairs

@jit(nopython=True, nogil=True, cache=True)
def iter_catalog(img_pos_1, img_pos_2, img_theta, img_mag_1, img_mag_2, cat_features, eps_pi, eps_m):
    # check magnitude here
    # now features and distances are both len 5 vectors
    # img_feature = [star_pos1, star_pos2, star_theta, star_mag_pos1, star_mag_pos2]
    # cat_feature = [cat_id1, cat_id2, cat_theta, cat_mag_id1, cat_mag_id2]
    # pairs = [[star_pos1, star_pos2], [cat_id1, cat_id2], [star_mag_pos1, star_mag_pos2], [star_mag_pos1, star_mag_pos2], [cat_theta]]
    pairs_img_pos_1 = []
    pairs_img_pos_2 = []
    pairs_cat_id_1 = []
    pairs_cat_id_2 = []
    pairs_theta = []
    pairs_star_mag_1 = []
    pairs_star_mag_2 = []
    for i in range(cat_features.shape[0]):
        cat_feature = cat_features[i, :]
        if (cat_feature[2] > img_theta + eps_pi): # because the catalog is sorted based on angular distance
            break
        if (abs(cat_feature[2] - img_theta) <= eps_pi):
            if ((abs(cat_feature[3] - img_mag_1) <= eps_m) and (abs(cat_feature[4] - img_mag_2) <= eps_m)):
                pairs_img_pos_1.append(img_pos_1)
                pairs_img_pos_2.append(img_pos_2)
                pairs_cat_id_1.append(cat_feature[0])
                pairs_cat_id_2.append(cat_feature[1])
                pairs_star_mag_1.append(img_mag_1)
                pairs_star_mag_2.append(img_mag_2)
                pairs_theta.append(cat_feature[2])
            elif ((abs(cat_feature[3] - img_mag_2) <= eps_m) and (abs(cat_feature[4] - img_mag_1) <= eps_m)):
                pairs_img_pos_1.append(img_pos_1)
                pairs_img_pos_2.append(img_pos_2)
                pairs_cat_id_1.append(cat_feature[1])
                pairs_cat_id_2.append(cat_feature[0])
                pairs_star_mag_1.append(img_mag_1)
                pairs_star_mag_2.append(img_mag_2)
                pairs_theta.append(cat_feature[2])
    return pairs_img_pos_1, pairs_img_pos_2, pairs_cat_id_1, pairs_cat_id_2, pairs_star_mag_1, pairs_star_mag_2, pairs_theta

def check_list(ls, entry):
    for i, item in enumerate(ls):
        if item[0] == entry:
            return i
    return -1


def sort_occurence(entry):
    return entry[1]

def find_pole_in_pairs(pairs):
    index_counts = []
    counts = {}
    max_count_ids = []

    for distance in pairs:
        index_counts.append([])
        for pair in distance:
            for star_id in pair[1]:
                if star_id not in index_counts[-1]:
                    index_counts[-1].append(star_id)
                    if star_id in list(counts.keys()):
                        counts[star_id] += 1
                    else:
                        counts[star_id] = 1
    if len(counts) > 1:
        max_count = max(list(counts.values()))
    else:
        max_count = 0

    for i, count in enumerate(list(counts.values())):
        if count == max_count:
            max_count_ids.append(list(counts.keys())[i])
    if len(max_count_ids) > 1 or max_count < 3:
        return -1

    return max_count_ids[0]

def neighbour_set(pair_lists, pole_id, pole_position):
    neighbours = []
    for pair_list_id, pair_list in enumerate(pair_lists): #for each pair
        tmp_neighbours = []
        tmp_dists = []
        for pair in pair_list: #for each candidate in each pair
            if pole_id in pair[1]: #if pole exists
                exists = False
                for i in range(len(pair[1])): # check which one is the neighbour
                    if pair[1][i] != pole_id:
                        neighbour_id = pair[1][i]
                    if pair[0][i] != pole_position:
                        neighbour_position = pair[0][i]
                        neighbour_mag = pair[2][i]

                for neighbour in neighbours: # prevent repetition
                    if neighbour_id in neighbour:
                        exists = True
                if not exists:
                    tmp_neighbours.append([[neighbour_position, neighbour_id, neighbour_mag]])
                    tmp_dists.append(np.abs(angular_distance(neighbour_position, pole_position) - pair[3]))

        if (len(tmp_dists) > 1):
            tmp_neighbours = [tmp_neighbours[np.argmin(np.array(tmp_dists))]]

        if len(tmp_neighbours) == 1: # check if there is any neighbour
            neighbours.append(tmp_neighbours[0][0])

    return neighbours


def intersection(neighbours_i, neighbours_j):
    intersection = []
    for neighbour_i in neighbours_i:
        for neighbour_j in neighbours_j:
            if neighbour_i[1] == neighbour_j[1]:
                intersection.append(neighbour_i)
    return intersection

def confirm_pair_list(features, distance, eps_pi, gt_id):
    pairs = []
    for feature in features:
        if np.abs(feature[2] - distance) <= eps_pi:
            pairs.append((int(feature[0]), int(feature[1])))
    return pairs

def label_false_stars(confirmed, stars):
    false_stars_added = []
    for star in stars:
        for c in confirmed:
            false_star = True
            if star == c[0]:
                false_star = False
                false_stars_added.append(c[1])
                break
        if false_star:
            false_stars_added.append(-1)
    return false_stars_added

def remove_false_stars(set):
    return [s for s in set if s != -1]

def get_results(ids, hip_id, chain_set, out_dir, id_log_dir):
    no_result = isinstance(chain_set, int)

    if ((no_result) or (len(chain_set) < 3)):
        # write 0 to id rate
        result_str = '0\n'
        with open(out_dir, 'a') as f:
            f.write(result_str)

        # write 0 to id
        with open(id_log_dir, 'a') as f:
            f.write('0\n')

    else:
        id_set = []
        error_set = []
        for link in chain_set:
            if link in ids:
                id_set.append(link)
            else:
                error_set.append(link)
        id_rate = len(id_set) / (len(ids) - ids.count(-1))
        error_rate = len(error_set) / len(ids)

        if ((len(id_set) >= 3) and (len(error_set) == 0)):
            result_str = str(1) + '\n'
        else:
            result_str = str(-1) + '\n'

        # result_str = str(id_rate) + ', ' + str(error_rate) + '\n'
        with open(out_dir, 'a') as f:
            f.write(result_str)

        if (len(chain_set) > 0):
            with open(id_log_dir, 'a') as f:
                for id in chain_set:
                    f.write(str(hip_id[id]) + ' ')
                    # f.write(str(id) + ' ')
                f.write('\n')
    return

# error keys:
# 1: incorrect first pole
# 2: incorrect second pole
# 3: exceeded verification loop limit
# 4: number of stars < threshold
# 5: first pole == -1
# 6: second pole == -1
def write_error_log(file_path, log):
    f = open(file_path, 'a')
    f.write(log)
    return

def acceptance_phase(stars, mags, catalog_features, pole_position, pole_mag, eps_pi, eps_m):
    img_features = img_feature_set(stars, mags, pole_position, pole_mag)
    pairs = pair_list(catalog_features, img_features, eps_pi, eps_m)
    pole = find_pole_in_pairs(pairs)

    if pole > -1:
        neighbours = neighbour_set(pairs, pole, pole_position)
    else:
        neighbours = []
    return [pole_position, pole], neighbours

def verification_phase(pole_i, pole_j, neighbours_i, neighbours_j):
    u_1 = False
    u_2 = False
    for neighbour_i in neighbours_i:
        if pole_j[1] == neighbour_i[1]:
            u_1 = True
    for neighbour_j in neighbours_j:
        if pole_i[1] == neighbour_j[1]:
            u_2 = True
    if not u_1 or not u_2:
        return -1
    verified = intersection(neighbours_i, neighbours_j)
    if len(verified) < th:
        return -1
    return verified

def confirmation_phase(stars, features, verified, confirmed, eps_pi):
    if len(verified) == 0:
        confirmed = label_false_stars(confirmed, stars)
        return confirmed
    #print(f'verifying {verified[0][1]}')
    confirmed_features = []
    for feature in features:
        if confirmed[-1][1] in feature:
            confirmed_features.append(feature)
            #print(feature[0], feature[1])
    distance = angular_distance(verified[0][0], confirmed[-1][0])
    confirmed_pairs = confirm_pair_list(confirmed_features, distance, eps_pi, verified[0][1])
    pairs = [pair for pair in confirmed_pairs if verified[0][1] in pair]
    if len(pairs) > 0:
        confirmed.append(verified[0])

    return confirmation_phase(stars, features, verified[1:], confirmed, eps_pi)

def recursive_mpa(stars, mags, cat_features, pole_i, neighbours_i, j, Rp, rep_count, eps_pi, eps_m, ids):
    # two termination conditions: run out of neighbours and hits max Rp
    new_pole_i_flag = -1
    if rep_count >= Rp:
        write_error_log(curr_error_log_dir, "3 ")
        chain_set = -1
        pole_j = [[0, 0, 0], -1]
        # new_pole_i_flag = -1
        return chain_set, rep_count, pole_j, new_pole_i_flag
    elif j > len(neighbours_i) - 1: # if runs out of neighbours, change a new pole
        chain_set = -1
        pole_j = [[0, 0, 0], -1]
        new_pole_i_flag = 1
        return chain_set, rep_count, pole_j, new_pole_i_flag

    # pole_j selected from neighbours_i
    pole_j, neighbours_j = acceptance_phase(stars, mags, cat_features, neighbours_i[j][0], neighbours_i[j][2], eps_pi, eps_m)

    if pole_j[1] == -1:
        return recursive_mpa(stars, mags, cat_features, pole_i, neighbours_i, j + 1, Rp, rep_count, eps_pi, eps_m, ids)

    verified = verification_phase(pole_i, pole_j, neighbours_i, neighbours_j)
    if verified == -1: # change pole_j if it's not verified. go through every neighbour
        return recursive_mpa(stars, mags, cat_features, pole_i, neighbours_i, j + 1, Rp, rep_count + 1, eps_pi, eps_m, ids)

    verified.append(pole_j)
    chain_set = confirmation_phase(stars, cat_features, verified, [pole_i], eps_pi)

    return chain_set, rep_count, pole_j, new_pole_i_flag

def output_write(chain_set, pole_i, neighbour_i,
                 star_image, output_dir):
    # process pole first
    dist_from_pole_i = []
    for i in range(len(star_image)):
        curr_dist = np.arccos(np.dot(star_image[i], pole_i[0]))
        dist_from_pole_i.append(curr_dist)

    pole_point = pole_i[0]

    with open(output_dir, 'a') as f:
        f.write(str(pole_point[0]) + ' ' + str(pole_point[1]) + ' ' + str(pole_point[2]) + ' '
                + str(pole_i[1]) + '\n')

    # now proceed with each neighbour
    for n_id in range(len(neighbour_i)):
        curr_chain_set_id = chain_set[n_id + 1]
        if (curr_chain_set_id == -1):
            continue
        curr_neighbour = neighbour_i[n_id]

        dist_from_n_i = []
        for i in range(len(star_image)):
            curr_dist = np.arccos(np.minimum(np.dot(star_image[i] / np.linalg.norm(star_image[i]), curr_neighbour[0] / np.linalg.norm(curr_neighbour[0])), 1))
            dist_from_n_i.append(curr_dist)

        n_point = curr_neighbour[0]

        with open(output_dir, 'a') as f:
            f.write(str(n_point[0]) + ' ' + str(n_point[1]) + ' ' + str(n_point[2]) + ' '
                    + str(curr_neighbour[1]) +'\n')
    pass


if __name__ == "__main__":
    prepro_cat_path = sys.argv[1]
    prepro_cat_npy_path = sys.argv[2]
    dist_thres = float(sys.argv[3])
    mag_thres = float(sys.argv[4])
    max_num_trials = int(sys.argv[5])
    min_num_intersections = int(sys.argv[6])

    input_dir = sys.argv[7]
    output_dir = sys.argv[8]
    id = int(sys.argv[9])

    curr_input_dir = input_dir + str(id) + '.txt'
    curr_out_dir = output_dir + str(id) + '.txt'
    curr_error_log_dir = output_dir + str(id) + '_error_log.txt'
    eps_pi = dist_thres
    eps_m = mag_thres

    FOV = np.deg2rad(14)
    Rp = max_num_trials
    th = min_num_intersections

    ## comment this out if a new onboard catalog with different FOV is desired
    # features = create_feature_set(prepro_cat_path, FOV)
    # np.save(prepro_cat_npy_path, features)

    features = np.load(prepro_cat_npy_path)
    feature_sorted_id = np.argsort(features[:, 2])
    features = features[feature_sorted_id, :]
    #
    stars, ids, mags = input_reading(curr_input_dir)
    stars_sorted_id = np.argsort(mags)
    stars = [stars[i] for i in stars_sorted_id]
    ids = [ids[i] for i in stars_sorted_id]
    mags = [mags[i] for i in stars_sorted_id]


    if len(stars) < th:
        write_error_log(curr_error_log_dir, "4 ")

    rep_count = 0
    chain_set = -1
    i = -1
    new_pole_i_flag = 1 # initialise to 1
    pole_i = [[0, 0, 0], -1]
    pole_j = [[0, 0, 0], -1]
    while chain_set == -1 and rep_count < Rp and i < len(stars) - 1:
        # pole = [position, id]      neighbours = [[position_i, id_i], [position_j, id_j]]
        if (new_pole_i_flag):
            pole_i = [[0, 0, 0], -1]
            while pole_i[1] == -1 and i < len(stars) - 1:
                i += 1
                pole_i, neighbours_i = acceptance_phase(stars, mags, features, stars[i], mags[i], eps_pi, eps_m)

        if pole_i[1] != -1:
            chain_set, rep_count, pole_j, new_pole_i_flag = recursive_mpa(stars, mags, features, pole_i, neighbours_i, 0, Rp, rep_count,
                                                         eps_pi, eps_m, ids)
    #
    # # convert returned_pole into actual catalog HIP ID
    pred_HIP_id_pole_i = pole_i[1]
    pred_HIP_id_pole_j = pole_j[1]
    #
    # write result anyway
    if pred_HIP_id_pole_i not in ids:
        write_error_log(curr_error_log_dir, "1 ")
    if pred_HIP_id_pole_j not in ids:
        write_error_log(curr_error_log_dir, "2 ")
    if (pole_i[1] == -1):
        write_error_log(curr_error_log_dir, "5 ")
    if (pole_j[1] == -1):
        write_error_log(curr_error_log_dir, "6 ")

    output_write(chain_set, pole_i, neighbours_i,
                 stars, curr_out_dir)
    #
    if not isinstance(chain_set, int):
        chain_set = remove_false_stars(chain_set)
        ids = remove_false_stars(ids)



