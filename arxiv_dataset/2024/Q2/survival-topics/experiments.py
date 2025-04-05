'''
Survival topic models experiment script

Authors: Lexie Li, George H. Chen
'''
import gc
import itertools
import sys, os, argparse, json, pickle
os.environ['OPENBLAS_MAIN_FREE'] = '1'
import numpy as np
import torch
from lifelines.utils import concordance_index
from progressbar import ProgressBar
from pycox.evaluation import EvalSurv
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split


parser = argparse.ArgumentParser(
    description='Configure experiments as follows:')
parser.add_argument('dataset', type=str)
parser.add_argument('model', type=str)
parser.add_argument('n_outer_iter', type=int,
                    help='Number of random train/test split')
parser.add_argument('tuning_scheme', type=str,
                    help="One of 'grid' or 'bayesian'")
parser.add_argument('tuning_config', type=str,
                    help='Path suffix to tuning instructions')
parser.add_argument('experiment_id', type=str,
                    help='Unique identifier for experiment')
parser.add_argument('saved_experiment_id', type=str,
                    help='Unique identifier for saved experiment outputs')
parser.add_argument('readme_msg', type=str,
                    help='Experiment notes')
parser.add_argument('preset_dir', type=str,
                    help='Path suffix to preset parameters (for testing)')
parser.add_argument('manual_train_test', type=int,
                    help='Whether to use manually specified train/test splits')
# optional parameter
parser.add_argument('--data_dir', default='dataset', type=str)
parser.add_argument('--log_dir', default='logs', type=str,
                    help='Folder name to store logs')
parser.add_argument('--frac_to_use_as_training', default=0.8, type=float)
parser.add_argument('--seed', default=47, type=int)
parser.add_argument('--verbosity', default=1, type=int,
                    help='verbosity = {1,2,3}')
parser.add_argument('--run_unit_tests', default=True, type=bool)


def load_data(path, dataset, suffix, VERBOSE=False, TEST=False,
              TRANSCRIPT=None):
    X = np.load(os.path.join(path, dataset, 'X{}.npy'.format(suffix)))
    Y = np.load(os.path.join(path, dataset, 'Y{}.npy'.format(suffix)))
    F = []
    with open(os.path.join(path, dataset, 'F{}.txt'.format(suffix)), 'r') \
            as feature_names:
        for line in feature_names.readlines():
            F.append(line.strip())

    if VERBOSE:
        print('Loading {} dataset...'.format(dataset))
        print('  Total samples={}'.format(X.shape[0]))
        print('  Total features={}'.format(X.shape[1]))
    if TRANSCRIPT is not None:
        print('Loading {} dataset...'.format(dataset), file=TRANSCRIPT)
        print('  Total samples={}'.format(X.shape[0]), file=TRANSCRIPT)
        print('  Total features={}'.format(X.shape[1]), file=TRANSCRIPT)
    if TEST:
        assert(Y.shape[1] == 2)
        assert(set(np.unique(Y[:,1])).issubset({0.0, 1.0}))
        assert(X.shape[0] == Y.shape[0])
        assert(X.shape[1] == len(F))
        print('  Test passed!')

    return X, Y, F


def load_config(tuning_config_path):
    with open(tuning_config_path) as f:
        param_ranges = json.load(f)
    return param_ranges


def load_preset_params(preset_dir):
    if not preset_dir.endswith('None.json'):
        with open(preset_dir) as f:
            preset_params = json.load(f)
    else:
        preset_params = None
    return preset_params


def get_tuned_params_grid(model_name, config_dict, train_data, outer_iter_i,
                          experiment_dir):
    param_settings = config_dict['params']
    param_names = sorted(param_settings.keys())
    max_cindex = -np.inf
    arg_max = None
    rng = np.random.default_rng(seed=SEED)
    for param_setting in itertools.product(
            *[param_settings[param_name] for param_name in param_names]):
        param_setting_dict = dict(zip(param_names, param_setting))

        # different random seeds for different hyperparameter settings
        if model_name in {'rsf', 'deepsurv', 'deephit',
                          'scholar_ldacox', 'scholar_ldadraft',
                          'scholar_sagecox', 'scholar_sagedraft',
                          'sparse_scholar_sagecox', 'sparse_scholar_sagedraft',
                          'naive_ldacox'}:
            seed = rng.integers(2**32)
            param_setting_dict['random_state'] = seed

        f = get_validated_cindex_fn(model_name, train_data, outer_iter_i,
                                    experiment_dir)
        cindex, model = f(**param_setting_dict)
        if cindex > max_cindex:
            max_cindex = cindex
            if arg_max is not None:
                del arg_max
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            arg_max = (param_setting_dict, model)
        else:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return arg_max


def get_validated_cindex_fn(model_name, train_data, outer_iter_i,
                            experiment_dir):
    model_fn = get_model_fn(model_name)
    metric_fn = get_metric_fn(model_name)
    experiment_val_metrics_path = \
        os.path.join(experiment_dir, 'val_metrics.npy')

    # for saving intermediate models NOT trained on entire training
    experiment_val_model_path = \
        os.path.join(experiment_dir, 'saved_models', '{}_{}_{}.pickle')

    def validated_cindex(**kwargs):
        train_x, val_x, train_y, val_y = \
            train_test_split(train_data['x'], train_data['y'], test_size=0.2,
                             stratify=train_data['y'][:, 1], random_state=SEED)
        metric_table = np.zeros((1, 6)) # 1 outer index and 5 metrics
        observed_time_list = list(np.sort(np.unique(train_y[:, 0])))
        model = model_fn(**kwargs)
        model.fit(train_x, train_y, val_x, val_y, train_data['f'])

        if model_name == 'coxph':
            log_hazards = model.predict(val_x)
            metrics = metric_fn(-log_hazards, None, val_y)
            metric_table[0] = [outer_iter_i,
                               metrics['concordance_antolini'],
                               metrics['concordance_median'],
                               metrics['integrated_brier'],
                               metrics['rmse'], metrics['mae']]
        elif model_name in {'scholar_ldacox', 'scholar_sagecox',
                            'sparse_scholar_sagecox', 'naive_ldacox'}: 
            # predict_lazy only computes partial log hazard scores;
            # for models using Cox regression, c-index based on negative
            # partial log hazards is identical to time-dependent c-index
            theta, log_hazards = model.predict_lazy(val_x)
            metrics = metric_fn(-log_hazards, None, val_y)
            metric_table[0] = [outer_iter_i,
                               metrics['concordance_antolini'],
                               metrics['concordance_median'],
                               metrics['integrated_brier'],
                               metrics['rmse'], metrics['mae']]

        elif model_name in {'scholar_ldadraft',
                            'scholar_sagedraft',
                            'sparse_scholar_sagedraft'}:
            theta, predicted_test_times, predicted_survival_functions = \
                model.predict(val_x)
            metrics = metric_fn(predicted_test_times,
                                predicted_survival_functions,
                                val_y)
            metric_table[0] = [outer_iter_i,
                               metrics['concordance_antolini'],
                               metrics['concordance_median'],
                               metrics['integrated_brier'],
                               metrics['rmse'], metrics['mae']]

        else:
            predicted_test_times, predicted_survival_functions = \
                model.predict(val_x)
            metrics = metric_fn(predicted_test_times,
                                predicted_survival_functions,
                                val_y)
            metric_table[0] = [outer_iter_i,
                               metrics['concordance_antolini'],
                               metrics['concordance_median'],
                               metrics['integrated_brier'],
                               metrics['rmse'], metrics['mae']]

        # # TODO: fix optional step
        # # Optional step, uncomment to save these validation models
        # if model_name not in {'deepsurv', 'deephit',
        #                       'scholar_ldacox', 'naive_ldacox'}:
        #     with open(experiment_val_model_path.format(
        #             outer_iter_i, str(kwargs), inner_iter_i), 'wb') \
        #                 as model_write:
        #         pickle.dump(model, model_write)

        if not os.path.exists(experiment_val_metrics_path):
            np.save(experiment_val_metrics_path, metric_table)
        else:
            curr_table = np.load(experiment_val_metrics_path)
            np.save(experiment_val_metrics_path,
                    np.append(curr_table, metric_table, axis=0))

        print('>>>> Params: {} Validation C-index: {}'.format(
            kwargs, np.mean(metric_table[:, 1])))
        return metric_table[0, 1], model

    return validated_cindex

def get_model_fn(model_name):

    if model_name == 'coxph':
        from baselines.cox import CoxWrapper
        return CoxWrapper

    elif model_name == 'knnkm':
        from baselines.knnkm import KNNKaplanMeierWrapper
        return KNNKaplanMeierWrapper

    elif model_name == 'rsf':
        from baselines.rsf import RandomSurvivalForestWrapper
        return RandomSurvivalForestWrapper

    elif model_name == 'deepsurv':
        from baselines.deepsurv import DeepSurvWrapper
        return DeepSurvWrapper

    elif model_name == 'deephit':
        from baselines.deephit import DeepHitWrapper
        return DeepHitWrapper

    elif model_name == 'naive_ldacox':
        from supervised_topic_models.naive_ldacox import NaiveLDACox
        return NaiveLDACox

    elif model_name == 'scholar_ldacox':
        from supervised_topic_models.scholar import ScholarLDACoxWrapper
        return ScholarLDACoxWrapper

    elif model_name == 'scholar_ldadraft':
        from supervised_topic_models.scholar import ScholarLDADRAFTWrapper
        return ScholarLDADRAFTWrapper

    elif model_name == 'scholar_sagecox':
        from supervised_topic_models.scholar import ScholarSAGECoxWrapper
        return ScholarSAGECoxWrapper

    elif model_name == 'scholar_sagedraft':
        from supervised_topic_models.scholar import ScholarSAGEDRAFTWrapper
        return ScholarSAGEDRAFTWrapper

    elif model_name == 'sparse_scholar_sagecox':
        from supervised_topic_models.sparse_scholar import ScholarSAGECoxWrapper
        return ScholarSAGECoxWrapper

    elif model_name == 'sparse_scholar_sagedraft':
        from supervised_topic_models.sparse_scholar \
            import ScholarSAGEDRAFTWrapper
        return ScholarSAGEDRAFTWrapper

    raise NotImplementedError

def get_metric_fn(model_name):

    def standard_metric_fn(predicted_test_times,
                           predicted_survival_functions,
                           observed_y):
        # predicted survival function dim: n_train * n_test
        obs_test_times = observed_y[:, 0].astype(np.float32)
        obs_test_events = observed_y[:, 1].astype(np.float32)
        results = dict()

        ev = EvalSurv(predicted_survival_functions,
                      obs_test_times,
                      obs_test_events,
                      censor_surv='km')
        results['concordance_antolini'] = ev.concordance_td('antolini')
        results['concordance_median'] = \
            concordance_index(obs_test_times,
                              predicted_test_times,
                              obs_test_events.astype(np.bool))

        # ignore brier scores at highest test times as they become unstable
        time_grid = \
            np.linspace(obs_test_times.min(), obs_test_times.max(), 100)[:80]
        results['integrated_brier'] = ev.integrated_brier_score(time_grid)

        if sum(obs_test_events) > 0:
            # only noncensored samples are used for rmse/mae calculation
            pred_obs_differences = \
                predicted_test_times[obs_test_events.astype(np.bool)] \
                - obs_test_times[obs_test_events.astype(np.bool)]
            results['rmse'] = np.sqrt(np.mean((pred_obs_differences)**2))
            results['mae'] = np.mean(np.abs(pred_obs_differences))
        else:
            print('[WARNING] All samples are censored.')
            results['rmse'] = 0
            results['mae'] = 0

        return results

    def standard_metric_fn_lazy(predicted_neg_hazards,
                                predicted_survival_functions,
                                observed_y):
        # predicted survival function dim: n_train * n_test
        obs_test_times = observed_y[:, 0].astype(np.float32)
        obs_test_events = observed_y[:, 1].astype(np.float32)
        results = dict()

        assert(predicted_survival_functions is None)
        try:
            lifeline_cindex = \
                concordance_index(obs_test_times,
                                  predicted_neg_hazards,
                                  obs_test_events.astype(np.bool))
        except:
            print('Lifelines detected NaNs in input...')
            lifeline_cindex = 0.0

        results['concordance_antolini'] = lifeline_cindex
        results['concordance_median'] = np.nan
        results['integrated_brier'] = np.nan
        results['rmse'] = np.nan
        results['mae'] = np.nan

        return results

    def topic_metric_fn(preds, test_y):
        raise NotImplementedError

    if model_name in {'coxph',
                      'scholar_ldacox',
                      'scholar_sagecox',
                      'sparse_scholar_sagecox',
                      'naive_ldacox'}:
        return standard_metric_fn_lazy
    else:
        return standard_metric_fn


def metric_fn_par(args):
    predicted_test_times, predicted_survival_functions, observed_y = args

    # predicted survival function dim: n_train * n_test
    obs_test_times = observed_y[:, 0].astype(np.float32)
    obs_test_events = observed_y[:, 1].astype(np.float32)
    results = [0, 0, 0, 0, 0]

    ev = EvalSurv(predicted_survival_functions, obs_test_times,
                  obs_test_events, censor_surv='km')
    results[0] = ev.concordance_td('antolini') # concordance_antolini

    # concordance_median
    results[1] = concordance_index(obs_test_times,
                                   predicted_test_times,
                                   obs_test_events.astype(np.bool))

    # ignore brier scores at highest test times as they become unstable
    time_grid = \
        np.linspace(obs_test_times.min(), obs_test_times.max(), 100)[:80]
    results[2] = ev.integrated_brier_score(time_grid) # integrated_brier

    if sum(obs_test_events) > 0:
        # only noncensored samples are used for rmse/mae calculation
        pred_obs_differences = \
            predicted_test_times[obs_test_events.astype(np.bool)] \
            - obs_test_times[obs_test_events.astype(np.bool)]
        results[3] = np.sqrt(np.mean((pred_obs_differences)**2)) # rmse
        results[4] = np.mean(np.abs(pred_obs_differences)) # mae
    else:
        print('[WARNING] All samples are censored.')
        results[3] = 0
        results[4]  = 0

    return results


def metric_fn_par_lazy(args):
    predicted_test_times, predicted_survival_functions, observed_y = args
    # predicted_test_times is actually predicted_neg_hazards here

    # predicted survival function dim: n_train * n_test
    obs_test_times = observed_y[:, 0].astype(np.float32)
    obs_test_events = observed_y[:, 1].astype(np.float32)

    assert(predicted_survival_functions is None) 
    try:
        lifeline_cindex = concordance_index(obs_test_times,
                                            predicted_test_times,
                                            obs_test_events.astype(np.bool))
    except:
        print('Lifelines detected NaNs in inputs...')
        lifeline_cindex = 0.0

    result = lifeline_cindex
    results = [result, np.nan, np.nan, np.nan, np.nan]

    return results


if __name__ == '__main__':
    args = parser.parse_args()
    topic_models = {'scholar_ldacox', 'scholar_ldadraft',
                    'scholar_sagecox', 'scholar_sagedraft',
                    'sparse_scholar_sagecox', 'sparse_scholar_sagedraft',
                    'naive_ldacox'}
    model_name = args.model
    dataset = args.dataset
    log_dir = args.log_dir
    data_dir = args.data_dir
    experiment_id = args.experiment_id
    n_outer_iter = args.n_outer_iter
    frac_to_use_as_training = args.frac_to_use_as_training
    tuning_scheme = args.tuning_scheme
    tuning_config = args.tuning_config # configuration's version number
    preset_dir = args.preset_dir
    manual_train_test = args.manual_train_test > 0
    saved_experiment_id = args.saved_experiment_id
    readme_msg = args.readme_msg
    
    experiment_dir = os.path.join(log_dir, dataset, model_name, experiment_id)
    experiment_log_path = os.path.join(experiment_dir, 'transcript.txt')
    experiment_model_path = \
        os.path.join(experiment_dir, 'saved_model_{}.pickle')
    experiment_metrics_path = os.path.join(experiment_dir, 'metrics.npy')

    saved_experiment_dir = \
        os.path.join(log_dir, dataset, model_name, saved_experiment_id)

    if readme_msg != 'None':
        readme_msg_file = open(os.path.join(experiment_dir, 'exp_readme.txt'), 'w')
        readme_msg_file.write(readme_msg)
        readme_msg_file.close()

    global VERBOSE, TEST, SEED
    VERBOSE = args.verbosity
    TEST = args.run_unit_tests
    TRANSCRIPT = open(experiment_log_path, 'w')
    SEED = args.seed

    if model_name == 'coxph':
        dataset_suffix = '_cox'
    elif model_name in topic_models:
        dataset_suffix = '_discretized'
    else:
        dataset_suffix = ''

    X, Y, F = load_data(data_dir, dataset, suffix=dataset_suffix,
                        VERBOSE=VERBOSE, TEST=TEST, TRANSCRIPT=TRANSCRIPT)
    config_dict = \
        load_config(
            os.path.join(
                'configurations',
                '{}-{}.json'.format(model_name, tuning_config)))
    preset_params = \
        load_preset_params(
            os.path.join(
                'configurations',
                '{}-{}-{}-{}.json'.format(
                    dataset, model_name, tuning_config, preset_dir)))

    if VERBOSE:
        print('Configuring experiments...')
        print('  Model={}'.format(model_name))
        print('  Params={}'.format(config_dict))
        print('  Tuning scheme={}'.format(tuning_scheme))
        print('  Train/Test Split Repeats={}'.format(n_outer_iter))
        print('  Training fraction={}'.format(frac_to_use_as_training))
        print('  Random seed={}'.format(SEED))

    print('Configuring experiments...', file=TRANSCRIPT)
    print('  Model={}'.format(model_name), file=TRANSCRIPT)
    print('  Params={}'.format(config_dict), file=TRANSCRIPT)
    print('  Tuning scheme={}'.format(tuning_scheme), file=TRANSCRIPT)
    print('  Train/Test Split Repeats={}'.format(n_outer_iter), file=TRANSCRIPT)
    print('  Training fraction={}'.format(frac_to_use_as_training),
          file=TRANSCRIPT)
    print('  Random seed={}'.format(SEED), file=TRANSCRIPT)

    if manual_train_test:
        assert n_outer_iter == 1
        if dataset == 'SUPPORT_ARF_MOSF' \
                or dataset == 'SUPPORT_COPD_CHF_Cirrhosis' \
                or dataset == 'SUPPORT_Cancer' \
                or dataset == 'SUPPORT_Coma':
            train_index = np.loadtxt(
                os.path.join(data_dir, dataset,
                             'train_indices.txt')).astype(int)
            test_index = np.loadtxt(
                os.path.join(data_dir, dataset,
                             'test_indices.txt')).astype(int)
            train_test_splits = [(train_index, test_index)]
        else:
            raise NotImplementedError
    else:
        train_test_random_splitter = \
            StratifiedShuffleSplit(n_splits=n_outer_iter, random_state=SEED,
                                   test_size=0.2)
        train_test_splits = train_test_random_splitter.split(X, Y[:, 1])
    model_fn = get_model_fn(model_name)
    metric_fn = get_metric_fn(model_name)

    # we log 5 different metric:
    # concordance_antolini, concordance_median, integrated_brier, rmse, mae
    metric_table = np.zeros((n_outer_iter, 5)) 
    outer_iter_i = 0

    # preserve censor rate between train and test
    for train_index, test_index in train_test_splits:

        # For producing bootstrapped test outputs: only look at iter 0
        if outer_iter_i != 0:
            outer_iter_i += 1
            continue

        print('  >> Iter {} in progress...'.format(outer_iter_i))

        train_data = {'x': X[train_index], 'y': Y[train_index], 'f': F}
        test_data = {'x': X[test_index], 'y': Y[test_index], 'f': F}
        train_x, train_y = (train_data['x'], train_data['y'])
        test_x, test_y = (test_data['x'], test_data['y'])
        observed_time_list = list(np.sort(np.unique(train_y[:, 0])))

        if saved_experiment_id != 'None':
            # load presaved models
            if model_name.startswith('scholar'):
                with open(os.path.join(saved_experiment_dir,
                               'saved_best_params_{}.pickle').format(
                    outer_iter_i), 'rb') as model_read:
                    best_params = pickle.load(model_read)
                print('  >> Iter {} best params: '.format(outer_iter_i),
                      best_params)
                print('  >> Iter {} best params: '.format(outer_iter_i),
                      best_params, file=TRANSCRIPT)

                model = model_fn(
                    saved_model=os.path.join(saved_experiment_dir,
                                             'saved_model_{}').format(outer_iter_i),
                    **best_params)
                train_x, val_x, train_y, val_y = \
                    train_test_split(train_data['x'], train_data['y'], test_size=0.2,
                             stratify=train_data['y'][:, 1], random_state=SEED)
                model.fit(train_x, train_y, val_x, val_y, train_data['f'])

            elif model_name in {'deepsurv', 'deephit'}:
                with open(os.path.join(saved_experiment_dir,
                               'saved_best_params_{}.pickle').format(
                    outer_iter_i), 'rb') as model_read:
                    best_params = pickle.load(model_read)
                print('  >> Iter {} best params: '.format(outer_iter_i),
                      best_params)
                print('  >> Iter {} best params: '.format(outer_iter_i),
                      best_params, file=TRANSCRIPT)

                model = model_fn(
                    saved_model=os.path.join(saved_experiment_dir,
                                             'saved_model_{}.pt').format(outer_iter_i),
                    **best_params)
                train_x, val_x, train_y, val_y = \
                    train_test_split(train_data['x'], train_data['y'], test_size=0.2,
                             stratify=train_data['y'][:, 1], random_state=SEED)
                model.fit(train_x, train_y, val_x, val_y, train_data['f'])

            elif model_name == 'coxph':
                with open(os.path.join(saved_experiment_dir,
                               'saved_best_params_{}.pickle').format(
                    outer_iter_i), 'rb') as model_read:
                    best_params = pickle.load(model_read)
                print('  >> Iter {} best params: '.format(outer_iter_i),
                      best_params)
                print('  >> Iter {} best params: '.format(outer_iter_i),
                      best_params, file=TRANSCRIPT)

                model = model_fn(
                    saved_model=os.path.join(saved_experiment_dir,
                                             'saved_model_{}.pickle').format(outer_iter_i),
                    **best_params)
                train_x, val_x, train_y, val_y = \
                    train_test_split(train_data['x'], train_data['y'], test_size=0.2,
                             stratify=train_data['y'][:, 1], random_state=SEED)
                model.fit(train_x, train_y, val_x, val_y, train_data['f'])

            else:
                with open(os.path.join(saved_experiment_dir,
                               'saved_model_{}.pickle').format(
                    outer_iter_i), 'rb') as model_read:
                    model = pickle.load(model_read)

        else:
            # hyperparam sweeping
            if preset_params is not None:
                best_params = preset_params
                train_x, val_x, train_y, val_y = \
                    train_test_split(train_data['x'], train_data['y'],
                                     test_size=0.2,
                                     stratify=train_data['y'][:, 1],
                                     random_state=SEED)
                model = model_fn(**best_params)
                model.fit(train_x, train_y, val_x, val_y, train_data['f'])
            else:
                best_params, model = \
                    get_tuned_params_grid(model_name, config_dict, train_data,
                                          outer_iter_i, experiment_dir)

            # save best params
            with open(os.path.join(experiment_dir,
                                   'saved_best_params_{}.pickle').format(
                          outer_iter_i), 'wb') as model_write:
                pickle.dump(best_params, model_write)

            print('  >> Iter {} best params: '.format(outer_iter_i),
                  best_params)
            print('  >> Iter {} best params: '.format(outer_iter_i),
                  best_params, file=TRANSCRIPT)

        if 'explain' in experiment_id:

            if model_name == 'coxph':
                # here explain means beta coefficients

                getcoxbeta_fname_pickle = \
                    os.path.join(experiment_dir,
                                 '{}_{}_{}_{}_beta.pickle'.format(
                        model_name, dataset, experiment_id, outer_iter_i))
                getcoxbeta_fname_text = \
                    os.path.join(experiment_dir,
                                 '{}_{}_{}_{}_beta.txt'.format(
                        model_name, dataset, experiment_id, outer_iter_i))

                sorted_beta_indices = np.argsort(-abs(model.beta))
                sorted_beta = model.beta[sorted_beta_indices]
                sorted_feature_names = \
                    np.array(model.feature_names)[sorted_beta_indices]
                n_nonzero_betas = sum(sorted_beta != 0)

                result_strs = [
                    '>>> Cox PH regression (regularized) beta coefficients '
                    + 'sorted by absolute value : {}'.format(dataset),
                    '>>> Total number of features : {}'.format(len(model.beta)),
                    '>>> Number of features with nonzero betas: {}\n'.format(
                        n_nonzero_betas)]
                for b_i in range(n_nonzero_betas):
                    curr_beta = sorted_beta[b_i]
                    curr_feature = sorted_feature_names[b_i]
                    curr_str = '{} : ({})'.format(curr_feature, curr_beta)
                    result_strs.append(curr_str)

                all_str = '\n'.join(result_strs)
                
                with open(getcoxbeta_fname_text, 'w') as coxfile:
                    coxfile.write(all_str)
    
                cox_beta_explain = dict()
                cox_beta_explain['beta'] = model.beta
                cox_beta_explain['features'] = model.feature_names
                with open(getcoxbeta_fname_pickle, 'wb') as coxfile:
                    pickle.dump(cox_beta_explain, coxfile)

            elif model_name in topic_models:
                model.beta_explain(feature_names=train_data['f'], \
                    save_path=os.path.join(experiment_dir,
                                           '{}_{}_{}_{}_beta.pickle'.format(
                        model_name, dataset, experiment_id, outer_iter_i)))

        if 'bootstrap_predictions' in experiment_id:

            if model_name == 'coxph':
                log_hazards = model.predict(test_x)
                predicted_test_times = None
                metrics = metric_fn(-log_hazards, None, test_y)
                metric_table[outer_iter_i] = [metrics['concordance_antolini'],
                                              metrics['concordance_median'],
                                              metrics['integrated_brier'],
                                              metrics['rmse'],
                                              metrics['mae']]
            elif model_name in {'scholar_ldacox', 'scholar_sagecox',
                                'sparse_scholar_sagecox', 'naive_ldacox'}:
                theta, log_hazards = model.predict_lazy(test_x)
                predicted_test_times = None
                metrics = metric_fn(-log_hazards, None, test_y)
                metric_table[outer_iter_i] = [metrics['concordance_antolini'],
                                              metrics['concordance_median'],
                                              metrics['integrated_brier'],
                                              metrics['rmse'],
                                              metrics['mae']]

            elif model_name in {'scholar_ldadraft', 'scholar_sagedraft',
                                'sparse_scholar_sagedraft'}:
                theta, predicted_test_times, predicted_survival_functions = \
                    model.predict(test_x)
                metrics = metric_fn(predicted_test_times,
                                    predicted_survival_functions,
                                    test_y)
                metric_table[outer_iter_i] = [metrics['concordance_antolini'],
                                              metrics['concordance_median'],
                                              metrics['integrated_brier'],
                                              metrics['rmse'],
                                              metrics['mae']]

            else:
                predicted_test_times, predicted_survival_functions \
                    = model.predict(test_x)
                metrics = metric_fn(predicted_test_times,
                                    predicted_survival_functions,
                                    test_y)
                metric_table[outer_iter_i] = [metrics['concordance_antolini'],
                                              metrics['concordance_median'],
                                              metrics['integrated_brier'],
                                              metrics['rmse'],
                                              metrics['mae']]

            # save and update most recent metrics to prevent crashing
            np.save(experiment_metrics_path, metric_table)

            print('  >> Iter {} metrics: '.format(outer_iter_i), metrics)
            print('  >> Iter {} metrics: '.format(outer_iter_i), metrics,
                  file=TRANSCRIPT)

            print('  >> Entering prediction bootstrapping...')
            bootstrap_B = 1000
            
            if predicted_test_times is None:
                # in this case we only predicted negative hazard scores
                predicted_neg_hazards = -log_hazards
                n_test = predicted_neg_hazards.shape[0]
                np.random.seed(SEED)
                bootstrap_pool_metrics = []
                pbar = ProgressBar()
                for B_i in pbar(list(range(bootstrap_B))):
                    Bsample_indices = \
                        np.random.choice(n_test, size=n_test, replace=True)
                    predicted_neg_hazards_Bsamples = \
                        predicted_neg_hazards[Bsample_indices]
                    test_y_Bsamples = test_y[Bsample_indices]
                    bootstrap_pool_metrics.append(
                        metric_fn_par_lazy((predicted_neg_hazards_Bsamples,
                                            None, test_y_Bsamples)))
            else:
                n_test = predicted_test_times.shape[0]
                # bootstrapped_predictions_inputs = []
                bootstrap_rng = np.random.RandomState(SEED)
                bootstrap_pool_metrics = []
                pbar = ProgressBar()
                for B_i in pbar(list(range(bootstrap_B))):
                    predicted_survival_functions_Bsamples = \
                        predicted_survival_functions.sample(
                            n=n_test, replace=True, random_state=bootstrap_rng,
                            axis=1)
                    Bsample_indices = \
                        np.array(predicted_survival_functions_Bsamples.columns,
                                 dtype=int)
                    predicted_test_times_Bsamples = \
                        predicted_test_times[Bsample_indices]
                    test_y_Bsamples = test_y[Bsample_indices]
                    bootstrap_pool_metrics.append(
                        metric_fn_par((predicted_test_times_Bsamples,
                                       predicted_survival_functions_Bsamples,
                                       test_y_Bsamples)))
            
            bootstrap_pool_metrics = np.array(bootstrap_pool_metrics)
            np.save(os.path.join(experiment_dir, 'metrics_bootstrapped.npy'),
                    bootstrap_pool_metrics)

            print('  >> Iter {} bootstrapped : MEAN'.format(outer_iter_i),
                  np.mean(bootstrap_pool_metrics, axis=0))
            print('  >> Iter {} bootstrapped : MEDIAN'.format(outer_iter_i),
                  np.median(bootstrap_pool_metrics, axis=0))
            
            print('  >> Iter {} bootstrapped : Q=0.025'.format(outer_iter_i),
                  np.quantile(bootstrap_pool_metrics, q=0.025, axis=0,
                              interpolation='lower'))
            print('  >> Iter {} bootstrapped : Q=0.975'.format(outer_iter_i),
                  np.quantile(bootstrap_pool_metrics, q=1-0.025, axis=0,
                              interpolation='higher'))
            
            print('  >> Iter {} bootstrapped : MEAN'.format(outer_iter_i),
                  np.mean(bootstrap_pool_metrics, axis=0), file=TRANSCRIPT)
            print('  >> Iter {} bootstrapped : MEDIAN'.format(outer_iter_i),
                  np.median(bootstrap_pool_metrics, axis=0), file=TRANSCRIPT)
            
            print('  >> Iter {} bootstrapped : Q=0.025'.format(outer_iter_i),
                  np.quantile(bootstrap_pool_metrics, q=0.025, axis=0,
                              interpolation='lower'),
                  file=TRANSCRIPT)
            print('  >> Iter {} bootstrapped : Q=0.975'.format(outer_iter_i),
                  np.quantile(bootstrap_pool_metrics, q=1-0.025, axis=0,
                              interpolation='higher'),
                  file=TRANSCRIPT)

        else: # no bootstrapping

            predicted_test_times, predicted_survival_functions = \
                model.predict(test_x)
            metrics = \
                metric_fn(predicted_test_times,
                          predicted_survival_functions,
                          test_y)

            metric_table[outer_iter_i] = [metrics['concordance_antolini'],
                                          metrics['concordance_median'],
                                          metrics['integrated_brier'],
                                          metrics['rmse'],
                                          metrics['mae']]

            # save and update most recent metrics to prevent crashing
            np.save(experiment_metrics_path, metric_table)

            print('  >> Iter {} metrics: '.format(outer_iter_i), metrics)
            print('  >> Iter {} metrics: '.format(outer_iter_i), metrics,
                  file=TRANSCRIPT)

        if model_name in {'scholar_ldacox', 'scholar_ldadraft',
                          'scholar_sagecox', 'scholar_sagedraft',
                          'sparse_scholar_sagecox', 'sparse_scholar_sagedraft'}:
            # optional, save the topic model's theta matrix
            theta_path = \
                os.path.join(experiment_dir, '{}_{}_{}_{}_theta.npy'.format(
                    model_name, dataset, experiment_id, outer_iter_i))
            y_path = \
                os.path.join(experiment_dir, '{}_{}_{}_{}_testy.npy'.format(
                    model_name, dataset, experiment_id, outer_iter_i))
            np.save(theta_path, theta)
            np.save(y_path, test_y)
            print('Theta saved!')

        # if model_name not in {'deepsurv', 'deephit', 'scholar_ldacox'}:
        #     with open(experiment_model_path.format(outer_iter_i), 'wb') \
        #             as model_write:
        #         pickle.dump(model, model_write)
        if model_name.startswith('scholar'):
            model.save_to_disk(
                os.path.join(experiment_dir,
                             'saved_model_{}').format(outer_iter_i))
        elif model_name == 'deepsurv' or model_name == 'deephit':
            model.save_to_disk(
                os.path.join(experiment_dir,
                             'saved_model_{}.pt').format(outer_iter_i))
        elif model_name == 'coxph':
            model.save_to_disk(
                os.path.join(experiment_dir,
                             'saved_model_{}.pickle').format(outer_iter_i))
        elif model_name == 'rsf' or model_name == 'naive_ldacox':
            with open(experiment_model_path.format(outer_iter_i), 'wb') \
                    as model_write:
                pickle.dump(model, model_write)

        outer_iter_i += 1

    if 'bootstrap_predictions' in experiment_id:
        print('Finished!')
        print('Finished!', file=TRANSCRIPT)
    else:
        print('Finished! Test metrics mean and std:',
              np.mean(metric_table, axis=0),
              np.std(metric_table, axis=0))
        print('Finished! Test metrics mean and std:',
              np.mean(metric_table, axis=0),
              np.std(metric_table, axis=0), file=TRANSCRIPT)
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print(metric_table)
        print(metric_table, file=TRANSCRIPT)

    TRANSCRIPT.close()
