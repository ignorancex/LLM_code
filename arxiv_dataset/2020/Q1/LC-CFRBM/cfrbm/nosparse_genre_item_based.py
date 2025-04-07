import json
import sys
import csv

from collections import defaultdict
from math import sqrt

import numpy as np
import theano.tensor as T

import utils

from genre_rbm import CFRBM
from experiments import read_experiment
from utils import revert_expected_value, expand, iteration_str
from genre_dataset import load_dataset
from sklearn.metrics import precision_recall_fscore_support


def run(name, dataset, movie_info, config, all_users, all_movies, all_genres, tests, initial_v, sep):
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
    vis_g = T.matrix()
    vmasks_x = T.matrix()
    vmasks_g = T.matrix()

    rbm = CFRBM(len(all_users) * 5, len(all_genres), number_hidden)

    profiles = defaultdict(list)

    with open(dataset, 'rt') as data:
        for i, line in enumerate(data):
            uid, mid, rat, timstamp = line.strip().split(sep)
            profiles[mid].append((uid, float(rat)))

    print("Users and ratings loaded")

    movie_genres = defaultdict(list)
    
    r = csv.reader(open(movie_info, 'rb'), delimiter='|')
    for row in r:
        movie_genres[row[0]] = [int(x) for x in row[5:]]

    print("Movie genres loaded")

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
                            vis_g,
                            vmasks_x,
                            vmasks_g,
                            k=k,
                            w_lr=current_l_w,
                            v_lr=current_l_v,
                            h_lr=current_l_h,
                            decay=decay,
                            momentum=momentum)
        predict = rbm.predict(vis_x,vis_g)

        start_time = time.time()
        for batch_i, batch in enumerate(utils.chunker(profiles.keys(),
                                                      batch_size)):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            gen_profiles = {}
            masks_x = {}
            masks_g = {}
            for movieid in batch:
                movie_profile = [0.] * len(all_users)
                genre_profile = [0.] * len(all_genres)
                mask_x = [0] * (len(all_users) * 5)
                mask_g = [0] * (len(all_genres))

                for user_id, rat in profiles[movieid]:
                    movie_profile[all_users.index(user_id)] = rat
                    for _i in range(5):
                        mask_x[5 * all_users.index(user_id) + _i] = 1
                
                mask_g = [1] * len(all_genres)

                example_x = expand(np.array([movie_profile])).astype('float32')
                example_g = expand(np.array([genre_profile]), k=1).astype('float32')
                bin_profiles[movieid] = example_x
                gen_profiles[movieid] = example_g
                masks_x[movieid] = mask_x
                masks_g[movieid] = mask_g

            movies_batch = [bin_profiles[id] for id in batch]
            #print len(movies_batch),len(movies_batch[0][0]),size,len(all_users)
            genres_batch = [gen_profiles[id] for id in batch]
            #print len(genres_batch),len(genres_batch[0][0]),size,len(all_genres)
            masks_x_batch = [masks_x[id] for id in batch]
            masks_g_batch = [masks_g[id] for id in batch]

            train_batch_x = np.array(movies_batch).reshape(size,
                                                         len(all_users) * 5)
            train_batch_g = np.array(genres_batch).reshape(size,
                                                         len(all_genres))
            train_masks_x = np.array(masks_x_batch).reshape(size,
                                                        len(all_users) * 5)
            train_masks_g = np.array(masks_g_batch).reshape(size,
                                                        len(all_genres))
            train_masks_x = train_masks_x.astype('float32')
            train_masks_g = train_masks_g.astype('float32')
            train(train_batch_x, train_batch_g, train_masks_x, train_masks_g)
            sys.stdout.write('.')
            sys.stdout.flush()

        end_time = time.time()
        train_time = end_time - start_time

        ratings = []
        predictions = []

        start_time = time.time()

        for batch in utils.chunker(tests.keys(), batch_size):
            size = min(len(batch), batch_size)

            # create needed binary vectors
            bin_profiles = {}
            gen_profiles = {}
            masks_x = {}
            masks_g = {}
            for movieid in batch:
                movie_profile = [0.] * len(all_users)
                genre_profile = [0.] * len(all_genres)
                mask_x = [0] * (len(all_users) * 5)
                mask_g = [0] * (len(all_genres))

                for userid, rat in profiles[movieid]:
                    movie_profile[all_users.index(userid)] = rat
                    for _i in range(5):
                        mask_x[5 * all_users.index(userid) + _i] = 1

                mask_g = [1] * len(all_genres)

                example_x = expand(np.array([movie_profile])).astype('float32')
                example_g = expand(np.array([genre_profile]), k=1).astype('float32')
                bin_profiles[movieid] = example_x
                gen_profiles[movieid] = example_g
                masks_x[movieid] = mask_x
                masks_g[movieid] = mask_g


            positions = {movie_id: pos for pos, movie_id in enumerate(batch)}
            movies_batch = [bin_profiles[el] for el in batch]
            genres_batch = [gen_profiles[el] for el in batch]
            test_batch_x = np.array(movies_batch).reshape(size,
                                                        len(all_users) * 5)
            test_batch_g = np.array(genres_batch).reshape(size,
                                                        len(all_genres))
            movie_predictions = revert_expected_value(predict(test_batch_x, test_batch_g))

            for movie_id in batch:
                test_users = tests[movie_id]
                try:
                    for user, rating in test_users:
                        current_movie = movie_predictions[positions[movie_id]]
                        predicted = current_movie[all_users.index(user)]
                        rating = float(rating)
                        ratings.append(rating)
                        predictions.append(predicted)
                except Exception:
                    pass

        end_time = time.time()
        test_time = end_time - start_time

        vabs = np.vectorize(abs)
        distances = np.array(ratings) - np.array(predictions)

        true_rat = np.array(ratings, dtype=np.uint8)
        pred_rat = np.array(predictions, dtype=np.uint8)

        #print true_rat < 3, true_rat
        prec_rec = precision_recall_fscore_support(true_rat < 3,pred_rat < 3, average='binary')
        print prec_rec

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
        item_path = experiment['item_path']
        test_path = experiment['test_path']
        sep = experiment['sep']

        all_users, all_movies, all_genres,tests = load_dataset(train_path, test_path, item_path, sep,
                                                user_based=False)

        for config in experiment['configs']:
            run(name, train_path, item_path, config, all_users, all_movies, all_genres, tests, None, sep)
