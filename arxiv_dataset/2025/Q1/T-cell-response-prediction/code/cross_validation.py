import pandas as pd
from pathlib import Path
import os
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import code.params as cfg


class CrossValidation(object):

    def __init__(self, config_name, experiment_name, data_dir, start_id, test_experiment_name):
        self.config_name = config_name
        self.experiment_name = experiment_name
        self.test_experiment_name = test_experiment_name
        self.data_dir = data_dir
        self.start_id = start_id

    def nested_cv_configurations(self):
        for outer_fold in range(cfg.outer_k):
            yield from self.inner_cv_configurations(outer_fold)

    def inner_cv_configurations(self, outer_fold):
        for inner_fold in range(cfg.inner_k):
            yield from self.single_train_valid_split(outer_fold, inner_fold)

    def single_train_valid_split(self, outer_fold, inner_fold):
        yield from cfg.config_generator(self, outer_fold, inner_fold)

    def save_params_results(self, params, results):
        Path(params['valid_results_dir']).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([list(params.values()) + list(results.values())], columns=list(params.keys()) + list(results.keys()))
        df.to_pickle(params['valid_results_file'])

    def save_test_results(self, params, results):
        Path(params['test_results_dir']).mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([list(params.values()) + list(results.values())], columns=list(params.keys()) + list(results.keys()))
        df.to_pickle(params['test_results_file'])
        # allele_evaluation_df.to_pickle(params['test_allele_results_file'])

    def merge_test_results(self):
        config = cfg.get_combined_config(self)
        df_list = []
        for result_file in os.listdir(config['test_results_dir']):
            if 'allele' not in result_file:
                df_list.append(pd.read_pickle(config['test_results_dir'] + '/' + result_file))
        results_df = pd.concat(df_list)
        results_df.to_csv(config['combined_test_results_file'])
        return results_df

    def read_valid_results(self, config, outer_fold=None):
        df_list = []
        for result_file in os.listdir(config['valid_results_dir']):
            if '.pkl' in result_file:
                inner_fold = result_file[-5]
                df = pd.read_pickle(config['valid_results_dir'] + '/' + result_file)
                df['inner_fold'] = inner_fold
                df_list.append(df)
        results_df = pd.concat(df_list)
        if outer_fold is not None:
            results_df = results_df.loc[results_df['outer_fold'] == outer_fold]
        return results_df

    def get_best_hyperparams(self, selection_criteria, outer_k=cfg.outer_k, multiple_random_seeds=True):
        for outer_fold in range(outer_k):
            file_paths_config = cfg.get_combined_config(self, outer_fold, test=True)
            fold_results_df = self.read_valid_results(file_paths_config, outer_fold)
            for key, value in selection_criteria:
                fold_results_df = fold_results_df.loc[fold_results_df[key] == value]
            # take the mean over all inner folds with the same configuration
            selected_metrics = [metric for metric in set(fold_results_df.dropna(axis=1, how='all').columns) if 'Selected metric' in metric]
            if len(selected_metrics) != 1:
                print('selected metric undefined', outer_fold, selected_metrics)
                return False
            selected_metric = selected_metrics[0]

            param_keys = cfg.get_param_keys(self.config_name)
            param_keys.remove('random_seed')
            param_keys.remove('batch_size')
            if 'eval_interval' in param_keys:
                param_keys.remove('eval_interval')
            mean_results = fold_results_df[param_keys + ['early_stop_epochs', 'config_id', selected_metric]].groupby(param_keys).mean()

            # the index specifies the parameter values of the best configuration
            best_index = mean_results.idxmax()[selected_metric]
            best_params = dict(zip(param_keys, list(best_index)))

            # config_id will only be used for displaying it in tensorboard
            config_id = int(mean_results.loc[best_index, 'config_id'])

            if multiple_random_seeds:
                random_seeds = list(range(1, 11))
            else:
                random_seeds = [1]

            for random_seed in random_seeds:
                # add other parameters/file paths that are not considered for finding the best configuration
                best_params['random_seed'] = random_seed
                config = cfg.get_combined_config(self, outer_fold, test=True, config_id=config_id, params=best_params)
                params_combined = {key: best_params[key] if key in best_params else value for key, value in config.items()}
                params_combined['early_stop_epochs'] = round(mean_results.loc[best_index, 'early_stop_epochs'])
                params_combined['outer_fold'] = outer_fold
                if self.experiment_name == 'TransformerMultiSourcePaper':
                    params_combined['evaluate_pep_sources'] = True

                # pandas uses numpy types but pytorch uses python types for parameters
                params_casted = {key: value.item() if isinstance(value, np.integer) else value for key, value in params_combined.items()}

                yield params_casted

    def get_permutation_hyperparams(self):
        for outer_fold in range(cfg.outer_k):
            random_seeds = list(range(200))

            for random_seed in random_seeds:
                params = {'random_seed': random_seed, 'dataset_folder': 'nested_cv_thesis'}
                config = cfg.get_combined_config(self, outer_fold, test=True, config_id=0, params=params)
                config['early_stop_epochs'] = 200
                config['random_seed'] = random_seed

                # pandas uses numpy types but pytorch uses python types for parameters
                config_casted = {key: value.item() if isinstance(value, np.integer) else value for key, value in config.items()}

                yield config_casted

    def get_hyperparams_without_cv(self):
        for outer_fold in range(cfg.outer_k):
            random_seeds = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

            for random_seed in random_seeds:
                params = {'random_seed': random_seed, 'dataset_folder': 'nested_cv_thesis'}
                config = cfg.get_combined_config(self, outer_fold, test=True, config_id=0, params=params)
                config['early_stop_epochs'] = 120
                config['random_seed'] = random_seed
                config['adversary_learning_rate'] = 0.005
                config['learning_rate'] = 0.002
                config['embedding_dim'] = 32

                # pandas uses numpy types but pytorch uses python types for parameters
                config_casted = {key: value.item() if isinstance(value, np.integer) else value for key, value in config.items()}

                yield config_casted

    def inner_cv_to_tensorboard_mean(self, outer_fold):
        config = cfg.get_combined_config(self)
        fold_results_df = self.read_valid_results(config, outer_fold)
        value_counts = fold_results_df.nunique()
        variable_params = value_counts.loc[value_counts > 1].index
        # fold_results_df = self.add_selection_criterion(fold_results_df)
        param_keys = cfg.get_param_keys(self.config_name)
        parameter_selection = ''
        for param in set(variable_params) & set(param_keys):
            param_values = ', '.join(map(str, fold_results_df[param].unique()))
            parameter_selection += f'{param}: {param_values}\n\n'
        param_keys.remove('random_seed')
        param_keys.remove('batch_size')
        # take the mean over all inner folds with the same configuration
        grouped_results = fold_results_df.fillna(0.3).groupby(param_keys)
        config_strings = []
        for _, df in grouped_results:
            config_strings.append('_'.join(set([str(id) for id in df['config_id'].to_list()])))
        mean_results = grouped_results.mean()
        mean_results['config_string'] = config_strings
        writer = SummaryWriter(config['tensorboard_dir'])
        for param_values in mean_results.index:
            params = dict(zip(param_keys, list(param_values)))
            results = mean_results.loc[param_values].to_dict()
            config_id = results["config_string"]
            results_selection = {cfg.tensorboard_hparams_metrics[key]: value for key, value in results.items() if key in list(cfg.tensorboard_hparams_metrics.keys())}
            params_selection = {key: value for key, value in params.items() if key in variable_params}
            # params_selection = {**params_selection, 'random_seed': 'mean', 'inner_fold': 'mean'}
            writer.add_hparams(params_selection, results_selection,
                               run_name=f'{self.experiment_name}/config_{config_id}/cv_{outer_fold}_mean')
            self.log_tensorboard_text(parameter_selection, params, outer_fold, config_id, config['tensorboard_dir'])
        writer.close()

    def log_tensorboard_text(self, params, print_dict, outer_fold, config_id, tensorboard_dir, step=0):
        """
        Add the model configuration and results to the TensorBoard "Text" tab.
        """
        text_writer = SummaryWriter(f'{tensorboard_dir}/{self.experiment_name}/config_{config_id}/cv_{outer_fold}_mean')
        model_description = f'# {self.experiment_name}\n\n{self.config_name}\n\n{params}'
        table_head = f'|Parameter|Value|\n| :--- |:---|\n'
        table_main = "\n".join(f'|{k}|{v:.4f}|' if isinstance(v, float) else f'|{k}|{v}|' for k, v in print_dict.items())
        text_writer.add_text('Parameters', model_description + table_head + table_main, global_step=step)
        text_writer.close()

    def inner_cv_to_tensorboard(self, outer_fold):
        config = cfg.get_combined_config(self)
        fold_results_df = self.read_valid_results(config, outer_fold)
        value_counts = fold_results_df.nunique()
        variable_params = value_counts.loc[value_counts > 1].index
        # fold_results_df = self.add_selection_criterion(fold_results_df)
        param_keys = cfg.get_param_keys(self.config_name) + ['inner_fold']
        param_keys.remove('random_seed')
        # not really take the mean because all groups just have one element here
        mean_results = fold_results_df.fillna("").groupby(param_keys).mean()
        writer = SummaryWriter(config['tensorboard_dir'])
        for param_values in mean_results.index:
            params = dict(zip(param_keys, list(param_values)))
            results = mean_results.loc[param_values].to_dict()
            config_id = round(results["config_id"])
            inner_fold = params['inner_fold']
            results_selection = {cfg.tensorboard_hparams_metrics[key]: value for key, value in results.items() if key in list(cfg.tensorboard_hparams_metrics.keys())}
            params_selection = {key: value for key, value in params.items() if key in variable_params}
            # writer.add_hparams(params_selection, results_selection,
            #                    run_name=f'{self.experiment_name}/config_{config_id}/cv_{outer_fold}_mean')
            writer.add_hparams(params_selection, results_selection,
                               run_name=f'{self.experiment_name}/config_{config_id}/cv_{outer_fold}_{inner_fold}')

        writer.add_text('Experiment name', self.experiment_name, global_step=0)
        writer.close()
