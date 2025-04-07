import json
import sys
import csv
import time

from collections import defaultdict
from math import sqrt

import numpy as np
import theano.tensor as T

from user_info_rbm import CFRBM
from experiments import read_experiment
from utils import chunker, revert_expected_value, expand, iteration_str
from user_info_dataset import load_dataset
from sklearn.metrics import precision_recall_fscore_support


def run(name, dataset, user_info, config, all_users, all_movies, all_occupations, all_sex, all_ages, tests, initial_v, sep):
    config_name = config['name']
    number_hidden = config['number_hidden']
    epochs = config['epochs']
    ks = config['ks']
    momentums = config['momentums']
    l_w = config['l_w']
    l_v = config['l_v']
    l_h = config['l_h']
    decay = config['decay']
    batch_size = config['batch_size']

    config_result = config.copy()
    config_result['results'] = []

    vis_x = T.matrix()
    vis_o = T.matrix()
    vis_s = T.matrix()
    vis_a = T.matrix()
    vmasks_x = T.matrix()
    vmasks_o = T.matrix()
    vmasks_s = T.matrix()
    vmasks_a = T.matrix()

    rbm = CFRBM(len(all_movies) * 5, len(all_occupations), 1, len(all_ages), number_hidden)

    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            profiles[uid].append((mid, float(rat)))

    print("Users and ratings loaded")

    user_occ = defaultdict(list)
    user_sex = defaultdict(list)
    user_age = defaultdict(list)

    r = csv.reader(open(user_info, 'rb'), delimiter='|')
    for row in r:
        user_age[row[0]] = [int(x) for x in row[1:7]]
        user_sex[row[0]] = [int(row[7])]
        user_occ[row[0]] = [int(x) for x in row[8:]]

    print("User info loaded")

    for j in range(epochs):
        def get_index(col):
            if j/(epochs/len(col)) < len(col):
                return j/(epochs/len(col))
            else:
                return -1

        index = get_index(ks)
        mindex = get_index(momentums)
        icurrent_l_w = get_index(l_w)
        icurrent_l_v = get_index(l_v)
        icurrent_l_h = get_index(l_h)

        k = ks[index]
        momentum = momentums[mindex]
        current_l_w = l_w[icurrent_l_w]
        current_l_v = l_v[icurrent_l_v]
        current_l_h = l_h[icurrent_l_h]

        train = rbm.cdk_fun(vis_x,
                            vis_o,
                            vis_s,
                            vis_a,
                            vmasks_x,
                            vmasks_o,
                            vmasks_s,
                            vmasks_a,
                            k=k,
                            w_lr=current_l_w,
                            v_lr=current_l_v,
                            h_lr=current_l_h,
                            decay=decay,
                            momentum=momentum)
        predict = rbm.predict(vis_x, vis_o, vis_s, vis_a)

        start_time = time.time()

        for batch_i, batch in enumerate(chunker(profiles.keys(),
                                                batch_size)):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            occ_profiles = {}
            sex_profiles = {}
            age_profiles = {}
            masks_x = {}
            masks_o = {}
            masks_s = {}
            masks_a = {}
            for userid in batch:
                user_profile = [0.] * len(all_movies)
                occ_profile = [0.] * len(all_occupations)
                sex_profile = [0.] * 1
                age_profile = [0.] * len(all_ages)
                mask_x = [0] * (len(all_movies) * 5)
                mask_o = [1] * (len(all_occupations))
                mask_s = [1] * (1)
                mask_a = [1] * (len(all_ages))

                for movie_id, rat in profiles[userid]:
                    user_profile[all_movies.index(movie_id)] = rat
                    for _i in range(5):
                        mask_x[5 * all_movies.index(movie_id) + _i] = 1

                mask_o = [1] * len(all_occupations)
                mask_s = [1] * 1
                mask_a = [1] * len(all_ages)

                example_x = expand(np.array([user_profile])).astype('float32')
                example_o = expand(np.array([occ_profile]), k=1).astype('float32')
                example_s = expand(np.array([sex_profile]), k=1).astype('float32')
                example_a = expand(np.array([age_profile]), k=1).astype('float32')
                bin_profiles[userid] = example_x
                occ_profiles[userid] = example_o
                sex_profiles[userid] = example_s
                age_profiles[userid] = example_a
                masks_x[userid] = mask_x
                masks_o[userid] = mask_o
                masks_s[userid] = mask_s
                masks_a[userid] = mask_a

            profile_batch = [bin_profiles[id] for id in batch]
            occ_batch = [occ_profiles[id] for id in batch]
            sex_batch = [sex_profiles[id] for id in batch]
            age_batch = [age_profiles[id] for id in batch]
            masks_x_batch = [masks_x[id] for id in batch]
            masks_o_batch = [masks_o[id] for id in batch]
            masks_s_batch = [masks_s[id] for id in batch]
            masks_a_batch = [masks_a[id] for id in batch]
            train_batch_x = np.array(profile_batch).reshape(size,
                                                          len(all_movies) * 5)
            train_batch_o = np.array(occ_batch).reshape(size,
                                                         len(all_occupations))
            train_batch_s = np.array(sex_batch).reshape(size,
                                                         1)
            train_batch_a = np.array(age_batch).reshape(size,
                                                         len(all_ages))
            train_masks_x = np.array(masks_x_batch).reshape(size,
                                                        len(all_movies) * 5)
            train_masks_o = np.array(masks_o_batch).reshape(size,
                                                        len(all_occupations))
            train_masks_s = np.array(masks_s_batch).reshape(size,
                                                        1)
            train_masks_a = np.array(masks_a_batch).reshape(size,
                                                        len(all_ages))
            train_masks_x = train_masks_x.astype('float32')
            train_masks_o = train_masks_o.astype('float32')
            train_masks_s = train_masks_s.astype('float32')
            train_masks_a = train_masks_a.astype('float32')
            train(train_batch_x, train_batch_o, train_batch_s, train_batch_a, train_masks_x, train_masks_o, train_masks_s, train_masks_a)
            sys.stdout.write('.')
            sys.stdout.flush()

        end_time = time.time()

        train_time = end_time - start_time

        ratings = []
        predictions = []

        start_time = time.time()

        for batch in chunker(tests.keys(), batch_size):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            occ_profiles = {}
            sex_profiles = {}
            age_profiles = {}
            masks_x = {}
            masks_o = {}
            masks_s = {}
            masks_a = {}
            for userid in batch:
                user_profile = [0.] * len(all_movies)
                occ_profile = [0.] * len(all_occupations)
                sex_profile = [0.] * 1
                age_profile = [0.] * len(all_ages)
                mask_x = [0] * (len(all_movies) * 5)
                mask_o = [1] * (len(all_occupations))
                mask_s = [1] * (1)
                mask_a = [1] * (len(all_ages))

                for movie_id, rat in profiles[userid]:
                    user_profile[all_movies.index(movie_id)] = rat
                    for _i in range(5):
                        mask_x[5 * all_movies.index(movie_id) + _i] = 1

                mask_o = [1] * len(all_occupations)
                mask_s = [1] * 1
                mask_a = [1] * len(all_ages)

                example_x = expand(np.array([user_profile])).astype('float32')
                example_o = expand(np.array([occ_profile]), k=1).astype('float32')
                example_s = expand(np.array([sex_profile]), k=1).astype('float32')
                example_a = expand(np.array([age_profile]), k=1).astype('float32')
                bin_profiles[userid] = example_x
                occ_profiles[userid] = example_o
                sex_profiles[userid] = example_s
                age_profiles[userid] = example_a
                masks_x[userid] = mask_x
                masks_o[userid] = mask_o
                masks_s[userid] = mask_s
                masks_a[userid] = mask_a

            positions = {profile_id: pos for pos, profile_id
                         in enumerate(batch)}
            profile_batch = [bin_profiles[el] for el in batch]
            occ_batch = [occ_profiles[el] for el in batch]
            sex_batch = [sex_profiles[el] for el in batch]
            age_batch = [age_profiles[el] for el in batch]
            test_batch_x = np.array(profile_batch).reshape(size,
                                                         len(all_movies) * 5)
            test_batch_o = np.array(occ_batch).reshape(size,
                                                        len(all_occupations))
            test_batch_s = np.array(sex_batch).reshape(size,
                                                        1)
            test_batch_a = np.array(age_batch).reshape(size,
                                                        len(all_ages))
            user_preds = revert_expected_value(predict(test_batch_x, test_batch_o, test_batch_s, test_batch_a))
            for profile_id in batch:
                test_movies = tests[profile_id]
                try:
                    for movie, rating in test_movies:
                        current_profile = user_preds[positions[profile_id]]
                        predicted = current_profile[all_movies.index(movie)]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        end_time = time.time()

        test_time = end_time - start_time

        true_rat = np.array(ratings, dtype=np.uint8)
        pred_rat = np.array(predictions, dtype=np.uint8)

        #print true_rat < 3, true_rat
        prec_rec = precision_recall_fscore_support(true_rat < 3,pred_rat < 3, average='binary')
        print prec_rec

        vabs = np.vectorize(abs)
        distances = np.array(ratings) - np.array(predictions)

        mae = vabs(distances).mean()
        rmse = sqrt((distances ** 2).mean())

        iteration_result = {
            'iteration': j,
            'k': k,
            'momentum': momentum,
            'mae': mae,
            'rmse': rmse,
            'lrate': current_l_w,
            'train_time': train_time,
            'test_time': test_time,
            'prec_rec': prec_rec
        }

        config_result['results'].append(iteration_result)

        print(iteration_str.format(j, k, current_l_w, momentum, mae, rmse))

        with open('experiments/{}_{}.json'.format(config_name, name), 'wt') as res_output:
            res_output.write(json.dumps(config_result, indent=4))

if __name__ == "__main__":
    experiments = read_experiment(sys.argv[1])

    for experiment in experiments:
        name = experiment['name']
        train_path = experiment['train_path']
        user_path = experiment['user_path']
        test_path = experiment['test_path']
        sep = experiment['sep']
        configs = experiment['configs']

        all_users, all_movies, all_occupations, all_sex, all_ages, tests = load_dataset(train_path, test_path, user_path,
                                                    sep, user_based=True)

        for config in configs:
            run(name, train_path, user_path, config, all_users, all_movies, all_occupations, all_sex, all_ages, tests,
                None, sep)
