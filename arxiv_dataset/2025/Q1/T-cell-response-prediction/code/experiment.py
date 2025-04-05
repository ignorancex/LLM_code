import os
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import random
from code.datasets import TcellData
from code.evaluation import get_evaluation_dict, get_source_evaluation_dict, get_allele_evaluation_dict
import code.params as cfg
from code import models
import torch.nn as nn
import time

class Experiment(object):
    """
    This class contains all training and evaluation methods.
    """
    def __init__(self, params, args, saved_model=None):
        self.params = params
        self.args = args
        self.writer = SummaryWriter(params['tensorboard_experiment_dir'])

        if not args.server:
            from tqdm import tqdm
            self.tqdm = tqdm
        else:
            self.tqdm = lambda i: i

        model_dict = {
            'Logistic Regression': models.TcellLogit,
            'MLP': models.TcellMLP,
            'PeptideTransformer': models.PeptideTransformer
        }
        TcellModel = model_dict[params['model']]
        self.TcellModel = TcellModel

        if params['random_seed'] is not None:
            torch.manual_seed(params['random_seed'])
            random.seed(params['random_seed'])
            np.random.seed(params['random_seed'])

        self.data = TcellData(params)

        self.model = TcellModel(params, self.data)

        self.steps_per_epoch = self.params['lr_step_size'] * self.data.train_dataset_size // self.params['batch_size']

        # print('actual batch size', self.params['batch_size'])
        # print('steps per epoch', self.steps_per_epoch)

        self.optimizer = torch.optim.AdamW(self.get_params(self.model, self.params['l2_reg']), lr=self.params['learning_rate'])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda step: self.get_lr_factor(step))

        if self.params['debiasing'] or self.params['evaluate_adv']:
            input_dim = self.model.adversary_input_dim

            Adversary = models.Adversary

            self.pep_source_adversary = Adversary(params, input_dim, len(self.data.pep_source_dict), self.data.pep_source_freq_pos, self.data.pep_source_freq_neg)
            self.allele_adversary = Adversary(params, input_dim, len(self.data.allele_dict), self.data.allele_freq_pos, self.data.allele_freq_neg)

            l2_reg = self.params['adversary_l2_reg']

            adversary_params = self.get_params(self.allele_adversary, l2_reg) + self.get_params(self.pep_source_adversary, l2_reg)
            self.adversary_optimizer = torch.optim.AdamW(adversary_params, lr=self.params['adversary_learning_rate'])
            self.adversary_scheduler = torch.optim.lr_scheduler.LambdaLR(self.adversary_optimizer, lr_lambda=lambda step: self.get_lr_factor(step))

            embedding_params = self.get_params(self.model.source_mhc_params, l2_reg)
            self.embedding_optimizer = torch.optim.AdamW(embedding_params, lr=self.params['learning_rate'])
            self.embedding_scheduler = torch.optim.lr_scheduler.LambdaLR(self.embedding_optimizer, lr_lambda=lambda step: self.get_lr_factor(step))

            if self.params['evaluate_adv']:
                self.eval_pep_source_adversary = Adversary(params, input_dim, len(self.data.pep_source_dict), self.data.pep_source_freq_pos, self.data.pep_source_freq_neg)
                self.eval_allele_adversary = Adversary(params, input_dim, len(self.data.allele_dict), self.data.allele_freq_pos, self.data.allele_freq_neg)
                eval_adversary_params = self.get_params(self.eval_allele_adversary, l2_reg) + self.get_params(self.eval_pep_source_adversary, l2_reg)
                self.eval_adversary_optimizer = torch.optim.AdamW(eval_adversary_params, lr=0.001)

        torch.manual_seed(params['random_seed'])
        random.seed(params['random_seed'])
        np.random.seed(params['random_seed'])

        if saved_model is not None:
            self.load_model(saved_model)

        if args.verbose:
            param_count = 0
            for name, parameter in self.model.named_parameters():
                if not parameter.requires_grad:
                    continue
                print(name, parameter.numel())
                param_count += parameter.numel()
            print(f'Number of params: {param_count}')
            if params['debiasing']:
                print('\nAdversary parameters:')
                for name, parameter in self.pep_source_adversary.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    print(name, parameter.numel())
                for name, parameter in self.allele_adversary.named_parameters():
                    if not parameter.requires_grad:
                        continue
                    print(name, parameter.numel())

    def get_params(self, model, l2_reg):
        decay_parameters = self.get_parameter_names(model, [nn.LayerNorm, nn.BatchNorm1d])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        embedding_parameter = 'embedding_projection.linear.weight'
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if n in decay_parameters and n != embedding_parameter],
                "weight_decay": l2_reg,
            },
            {
                "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
                "weight_decay": 0.0,
            }]
        if embedding_parameter in decay_parameters:
            optimizer_grouped_parameters += [
                {
                    "params": [p for n, p in model.named_parameters() if n == embedding_parameter],
                    "weight_decay": self.params['protein_l2_reg'],
                },
            ]
        return optimizer_grouped_parameters

    def get_parameter_names(self, model, forbidden_layer_types):
        """
        Returns the names of the model parameters that are not inside a forbidden layer.
        """
        result = []
        for name, child in model.named_children():
            result += [
                f"{name}.{n}"
                for n in self.get_parameter_names(child, forbidden_layer_types)
                if not isinstance(child, tuple(forbidden_layer_types))
            ]
        # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
        result += list(model._parameters.keys())
        return result

    def get_adv_loss(self, batch, pep_source_pred, allele_pred, invert_labels=False):
        pep_source_one_hot = F.one_hot(batch['pep_source'], num_classes=pep_source_pred.size(1)).float()
        if invert_labels:
            pep_source_one_hot = 1 - pep_source_one_hot
        pep_source_adversary_loss = F.binary_cross_entropy_with_logits(pep_source_pred, pep_source_one_hot)

        allele_multi_hot = batch['allele_vectors_ones']
        if invert_labels:
            allele_multi_hot = 1 - allele_multi_hot
        allele_adversary_loss = F.binary_cross_entropy_with_logits(allele_pred, allele_multi_hot)

        return pep_source_adversary_loss, allele_adversary_loss

    def run_model(self, batch):
        labels = batch['label']
        predictions, _, tcell_activations = self.model(batch)

        source_pred, _ = self.pep_source_adversary(tcell_activations, labels)
        allele_pred, _ = self.allele_adversary(tcell_activations, labels)

        assert not tcell_activations.isnan().any()
        assert not predictions.isnan().any()
        assert not source_pred.isnan().any()
        assert not allele_pred.isnan().any()

        return predictions, source_pred, allele_pred

    def train_one_epoch_debiased(self, epoch):
        """
        Adversarial debiasing for MHC alleles and peptide sources
        """
        global step
        local_step = 0

        pep_source_adv_weight = self.params['pep_source_adversary_weight']
        allele_adv_weight = self.params['allele_adversary_weight']
        if self.params['increasing_adv_weights']:
            scaling = 2*torch.sigmoid(0.5*torch.tensor(epoch))-1
            pep_source_adv_weight *= scaling
            allele_adv_weight *= scaling

        for i, batch in enumerate(self.data.train_dataloader):
            step += 1
            self.optimizer.zero_grad()
            self.adversary_optimizer.zero_grad()

            self.model.train()
            self.pep_source_adversary.eval()
            self.allele_adversary.eval()

            predictions, source_pred, allele_pred = self.run_model(batch)

            # Update the T-cell response predictor weights with adversarial domain adaptation
            labels = batch['label']
            t_cell_response_loss = F.binary_cross_entropy_with_logits(predictions[:, 0], labels)

            invert_labels = True if self.params['adv_grad_reversal_loss'] == 'invert_labels' else False
            source_adversary_loss, allele_adversary_loss = self.get_adv_loss(batch, source_pred, allele_pred, invert_labels=invert_labels)
            pep_source_adv_loss = source_adversary_loss if invert_labels else - source_adversary_loss
            allele_adv_loss = allele_adversary_loss if invert_labels else - allele_adversary_loss

            loss = t_cell_response_loss + pep_source_adv_weight * pep_source_adv_loss + allele_adv_weight * allele_adv_loss
            loss.backward()
            self.optimizer.step()

            if self.args.log_tensorboard and not self.args.server:
                self.writer.add_scalar('Gradient norm per step', self.model.linear.weight.grad.data.norm(2).item(), step)

            # Update the adversary parameters, set gradients accumulated during T-cell response model update to zero
            self.optimizer.zero_grad()
            self.adversary_optimizer.zero_grad()

            self.model.eval()
            self.pep_source_adversary.train()
            self.allele_adversary.train()
            predictions, source_pred, allele_pred = self.run_model(batch)

            source_adversary_loss, allele_adversary_loss = self.get_adv_loss(batch, source_pred, allele_pred)

            adversary_loss = source_adversary_loss + allele_adversary_loss

            # with torch.autograd.set_detect_anomaly(True):
            adversary_loss.backward()
            self.adversary_optimizer.step()

            if self.args.log_tensorboard and not self.args.server:
                self.writer.add_scalar('Loss per step/Combined', loss.item(), step)
                self.writer.add_scalar('Loss per step/T-cell response', t_cell_response_loss.item(), step)
                self.writer.add_scalar('Loss per step/Allele adversary - uniform', allele_adv_loss.item(), step)
                self.writer.add_scalar('Loss per step/Peptide source adversary - uniform', pep_source_adv_loss.item(), step)
                self.writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], step)

                self.writer.add_scalar('Loss per step/Source adversary', source_adversary_loss.item(), step)
                self.writer.add_scalar('Loss per step/Allele adversary', allele_adversary_loss.item(), step)
                self.writer.add_scalar('Loss per step/Combined adversaries', adversary_loss.item(), step)

            self.scheduler.step()
            self.adversary_scheduler.step()

            local_step += 1

    def train_one_epoch(self, epoch):
        self.model.train()
        global step

        for i, batch in enumerate(self.data.train_dataloader):
            step += 1

            self.optimizer.zero_grad()
            labels = batch['label']

            predictions, _, _ = self.model(batch)
            t_cell_response_loss = F.binary_cross_entropy_with_logits(predictions[:, 0], labels)
            loss = t_cell_response_loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if self.args.log_tensorboard and not self.args.server:
                self.writer.add_scalar('Loss per step/T-cell response', t_cell_response_loss.item(), step)
                self.writer.add_scalar('Loss per step/Combined', loss.item(), step)
                self.writer.add_scalar('Example weight 2', self.model.linear.weight[0,0].item(), step)
                # self.writer.add_scalar('Gradient norm per step', self.model.linear.weight.grad.data.norm(2).item(), step)
                self.writer.add_scalar('Learning rate', self.scheduler.get_last_lr()[0], step)

    def merge_eval_results(self, results_list, n=10):
        merged_results = {
            metric: np.mean([result[metric] for result in results_list[-n:]]) for metric in results_list[0].keys()
        }
        return merged_results

    def train_model(self, verbose=False):
        prediction_dict = {'valid': {}, 'train': {}}
        valid_results_list = []

        start_time = time.time()
        early_stopping_metric = cfg.get_early_stopping_metric(self.params)
        # print(early_stopping_metric)
        early_stop_valid_results = {early_stopping_metric: float('-inf')}
        early_stop_train_results = {}
        early_stop_valid_results_sources = {}
        global step
        step = 0
        global eval_optimizer_step
        eval_optimizer_step = 0

        if self.args.log_tensorboard and (self.args.single_cv or self.args.no_cv):
            self.writer.add_text('Model description', self.params['description'], global_step=0)
            log_params = {key: self.params[key] for key in cfg.get_param_keys(self.args.config)}
            self.log_tensorboard_text(log_params, 'Hyperparameters', 'Parameter', 'Value')

        pretraining_epochs = self.params['pretraining_epochs'] if self.params['pretrain_model'] else 0

        for t in self.tqdm(range(self.params['epochs'] + pretraining_epochs)):
            if t == self.params['pretraining_epochs'] and self.params['pretrain_model']:
                self.data.train_dataloader = self.data.train_dataloader_human

            if self.params['debiasing']:
                self.train_one_epoch_debiased(t)
            else:
                self.train_one_epoch(t)


            if t % self.params['eval_interval'] == 0 or t == self.params['epochs'] - 1:
                if verbose:
                    print(f"\nEpoch {t + 1}\n-------------------------------")
                    print(f'learning rate: {self.scheduler.get_last_lr()}')

                tsne_vis = 'tsne_visualization' in self.params and self.params['tsne_visualization']
                pred_vis = 'pred_visualization' in self.params and self.params['pred_visualization']

                valid_results, valid_prediction_df = self.evaluate(self.data.eval_batch, self.data.eval_batch_unique,
                                                                   self.data.eval_data, 'Validation', verbose=verbose,
                                                                   log=self.args.log_tensorboard, epoch=t, save_latent_rep=tsne_vis)

                if pred_vis:
                    source_1 = 'Vaccinia virus'
                    source_2 = 'Human betaherpesvirus 6B'
                    # source_1 = 'Phleum pratense'
                    # source_2 = 'Dengue virus'
                    prediction_dict['valid'][t] = self.evaluate_two_sources(valid_prediction_df, source_1, source_2, t)
                    train_results, train_prediction_df = self.evaluate(self.data.train_eval_batch, self.data.train_eval_batch_unique,
                                                                       self.data.train_eval_data, 'Training', verbose=verbose,
                                                                       log=self.args.log_tensorboard, epoch=t, save_latent_rep=False)
                    prediction_dict['train'][t] = self.evaluate_two_sources(train_prediction_df, source_1, source_2, t)

                # For peptide sources with not enough data, skip the experiment
                if early_stopping_metric not in valid_results:
                    return None, None

                if 'early_stopping_moving_average' in self.params and self.params['early_stopping_moving_average']:
                    valid_results_list.append(valid_results)
                    valid_results = self.merge_eval_results(valid_results_list)

                pretraining_finished = (t > self.params['pretraining_epochs']) or 'pretrain_no_early_stop' not in self.params or self.params['pretrain_no_early_stop'] == False

                if (valid_results[early_stopping_metric] > early_stop_valid_results[early_stopping_metric]) and pretraining_finished:
                    if 'early_stopping_moving_average' not in self.params or not self.params['early_stopping_moving_average'] \
                            or len(valid_results_list) >= 10:
                        if self.params['evaluate_pep_sources'] or self.params['evaluate_alleles']:
                            early_stop_valid_results_sources, valid_predictions_all_sources = self.evaluate_sources(verbose=verbose)
                        early_stop_valid_results = valid_results
                        early_stop_valid_results[f'Selected metric - {early_stopping_metric}'] = early_stop_valid_results[early_stopping_metric]
                        if not pred_vis:
                            train_results, _ = self.evaluate(self.data.train_eval_batch, self.data.train_eval_batch_unique,
                                                             self.data.train_eval_data, 'Training', verbose=verbose,
                                                             log=self.args.log_tensorboard, epoch=t, save_latent_rep=False)

                        early_stop_train_results = train_results
                        self.params['early_stop_epochs'] = t + 1

            if (t % 10 == 0 or t == 0) and self.args.server:
                print(f'Epoch {t}')

        # visualize distributions of prediction scores for paper
        # import pickle
        # Path(self.params['visualization_dir']).mkdir(parents=True, exist_ok=True)
        # with open(self.params['visualization_prediction_file'], 'wb') as f:
        #     pickle.dump(prediction_dict, f)

        if 'early_stopping' in self.params and not self.params['early_stopping']:
            if self.params['evaluate_pep_sources'] or self.params['evaluate_alleles']:
                early_stop_valid_results_sources, valid_predictions_all_sources = self.evaluate_sources(verbose=verbose)
            valid_results, valid_prediction_df = self.evaluate(self.data.eval_batch, self.data.eval_batch_unique,
                                                               self.data.eval_data, 'Validation', verbose=verbose,
                                                               log=self.args.log_tensorboard, epoch=self.params['epochs'])
            early_stop_valid_results = valid_results
            early_stop_valid_results[f'Selected metric - {early_stopping_metric}'] = early_stop_valid_results[early_stopping_metric]
            train_results, _ = self.evaluate(self.data.train_eval_batch, self.data.train_eval_batch_unique,
                                             self.data.train_eval_data, 'Training', verbose=verbose,
                                             log=self.args.log_tensorboard, epoch=self.params['epochs'], save_latent_rep=False)

            early_stop_train_results = train_results

        if self.args.log_tensorboard:
            self.log_tensorboard_text(early_stop_valid_results, 'Valid results', 'Metric', 'Value', self.params['early_stop_epochs'])
            self.log_tensorboard_text(early_stop_train_results, 'Train results', 'Metric', 'Value', self.params['early_stop_epochs'])
            log_params = {key: self.params[key] for key in cfg.get_param_keys(self.args.config) + ['early_stop_epochs']}
            log_params['parameter_count'] = sum(p.numel() for p in self.model.parameters())
            self.log_tensorboard_text(log_params, 'Hyperparameters', 'Parameter', 'Value')

        early_stop_results = {**early_stop_train_results, **early_stop_valid_results, **early_stop_valid_results_sources,
                              'training_time': time.time() - start_time}

        if self.args.log_file and (self.args.single_cv or self.args.no_cv):
            Path(self.params['valid_predictions_dir']).mkdir(parents=True, exist_ok=True)
            results_df = pd.DataFrame([list(self.params.values()) + list(early_stop_results.values())],
                                      columns=list(self.params.keys()) + list(early_stop_results.keys()))
            results_df.T.to_csv(self.params['valid_results_text_file'])

        return early_stop_results, self.params

    def evaluate_two_sources(self, prediction_df, source_1, source_2, t):
        # visualize distributions of prediction scores
        pred_df_unique = prediction_df.drop_duplicates(['Epitope Description'])

        source_1_pos = pred_df_unique.loc[(pred_df_unique['Epitope Parent Species'] == source_1) &
                                          (pred_df_unique['Assay Qualitative Measure'] == 'Positive'), 'Prediction'].values

        source_1_neg = pred_df_unique.loc[(pred_df_unique['Epitope Parent Species'] == source_1) &
                                          (pred_df_unique['Assay Qualitative Measure'] == 'Negative'), 'Prediction'].values

        print(f'{source_1} pos: {np.mean(source_1_pos):.2f} +- {np.std(source_1_pos):.2f}, neg: {np.mean(source_1_neg):.2f} +- {np.std(source_1_neg):.2f}')

        source_2_pos = pred_df_unique.loc[(pred_df_unique['Epitope Parent Species'] == source_2) &
                                          (pred_df_unique['Assay Qualitative Measure'] == 'Positive'), 'Prediction'].values

        source_2_neg = pred_df_unique.loc[(pred_df_unique['Epitope Parent Species'] == source_2) &
                                          (pred_df_unique['Assay Qualitative Measure'] == 'Negative'), 'Prediction'].values

        print(f'{source_2} pos: {np.mean(source_2_pos):.2f} +- {np.std(source_2_pos):.2f}, neg: {np.mean(source_2_neg):.2f} +- {np.std(source_2_neg):.2f}')

        self.writer.add_histogram(f'Predictions/{source_1} positives (minority) - Valid', source_1_pos, t)
        self.writer.add_histogram(f'Predictions/{source_1} negatives - Valid', source_1_neg, t)
        self.writer.add_histogram(f'Predictions/{source_2} positives - Valid', source_2_pos, t)
        self.writer.add_histogram(f'Predictions/{source_2} negatives (minority) - Valid', source_2_neg, t)
        self.writer.add_scalar(f'Mean prediction/{source_1} positives (minority) - Valid', np.mean(source_1_pos), t)
        self.writer.add_scalar(f'Mean prediction/{source_1} negatives - Valid', np.mean(source_1_neg), t)
        self.writer.add_scalar(f'Mean prediction/{source_2} positives - Valid', np.mean(source_2_pos), t)
        self.writer.add_scalar(f'Mean prediction/{source_2} negatives (minority) - Valid', np.mean(source_2_neg), t)

        source_prediction_dict = {
            f'{source_1} pos': source_1_pos,
            f'{source_1} neg': source_1_neg,
            f'{source_2} pos': source_2_pos,
            f'{source_2} neg': source_2_neg
        }

        return source_prediction_dict

    def log_tensorboard_text(self, print_dict, description, key_header, value_header, step=0):
        """
        Add the model configuration and results to the TensorBoard "Text" tab.
        """
        model_description = f'{self.params["description"]}\n\n'
        table_head = f'|{key_header}|{value_header}|\n| :--- |:---|\n'
        table_main = "\n".join(f'|{k}|{v:.4f}|' if isinstance(v, float) else f'|{k}|{v}|' for k, v in print_dict.items())
        self.writer.add_text(description, model_description + table_head + table_main, global_step=step)

    def train_and_test_final_model(self, verbose=False):
        early_stopping_metric = cfg.get_early_stopping_metric(self.params)
        # print(early_stopping_metric)
        early_stop_train_results = {}
        global step
        step = 0
        global eval_optimizer_step
        eval_optimizer_step = 0

        if self.args.log_tensorboard and (self.args.single_cv or self.args.no_cv):
            self.writer.add_text('Model description', self.params['description'], global_step=0)
            log_params = {key: self.params[key] for key in cfg.get_param_keys(self.args.config)}
            self.log_tensorboard_text(log_params, 'Hyperparameters', 'Parameter', 'Value')

        for t in self.tqdm(range(self.params['early_stop_epochs'])):
            if t == self.params['pretraining_epochs'] and self.params['pretrain_model']:
                self.data.train_dataloader = self.data.train_dataloader_human

            if self.params['debiasing']:
                self.train_one_epoch_debiased(t)
            else:
                self.train_one_epoch(t)

            if t % self.params['eval_interval'] == 0 or t == self.params['epochs'] - 1:
                if verbose:
                    print(f"\nEpoch {t + 1}\n-------------------------------")
                    print(f'learning rate: {self.scheduler.get_last_lr()}')

                train_results, train_prediction_df = self.evaluate(self.data.train_eval_batch, self.data.train_eval_batch_unique,
                                                                   self.data.train_eval_data, 'Training', verbose=verbose,
                                                                   log=self.args.log_tensorboard, epoch=t, save_latent_rep=False)

            if (t % 10 == 0 or t == 0) and self.args.server:
                print(f'Epoch {t}')

        test_results, test_prediction_df = self.evaluate(self.data.eval_batch, self.data.eval_batch_unique,
                                                           self.data.eval_data, 'Test', verbose=verbose,
                                                           log=self.args.log_tensorboard, epoch=t)

        if self.params['evaluate_pep_sources'] or self.params['evaluate_alleles']:
            test_results_sources, _ = self.evaluate_sources(verbose=verbose)
        else:
            test_results_sources = {}

        if self.args.log_tensorboard:
            self.log_tensorboard_text(early_stop_train_results, 'Train results', 'Metric', 'Value', self.params['early_stop_epochs'])
            log_params = {key: self.params[key] for key in cfg.get_param_keys(self.args.config) + ['early_stop_epochs']}
            log_params['parameter_count'] = sum(p.numel() for p in self.model.parameters())
            self.log_tensorboard_text(log_params, 'Hyperparameters', 'Parameter', 'Value')

        early_stop_results = {**train_results, **test_results, **test_results_sources}

        if self.args.log_file:
            Path(self.params['test_results_dir']).mkdir(parents=True, exist_ok=True)
            results_df = pd.DataFrame([list(self.params.values()) + list(early_stop_results.values())],
                                      columns=list(self.params.keys()) + list(early_stop_results.keys()))
            results_df.T.to_csv(self.params['test_results_text_file'])
            # test_prediction_df.to_csv(self.params['test_prediction_file'])

        if self.args.save_model:
            self.save_model()

        return {**test_results, **test_results_sources}

    def evaluate_sources(self, verbose=True):
        self.model.eval()
        if self.params['debiasing']:
            self.pep_source_adversary.eval()
            self.allele_adversary.eval()
        with torch.no_grad():
            predictions, _, _ = self.model(self.data.eval_batch_full)
            eval_data_tmp = self.data.eval_data_full
            eval_data_tmp['Prediction'] = torch.sigmoid(predictions).numpy()
            result_dict_tmp = {}
            if self.params['evaluate_pep_sources']:
                result_dict_tmp = {**result_dict_tmp, **get_source_evaluation_dict(eval_data_tmp, 'Validation')}
            if self.params['evaluate_alleles']:
                result_dict_tmp = {**result_dict_tmp, **get_allele_evaluation_dict(eval_data_tmp, 'Validation')}

            result_dict = {}
            for metric in ['ROC AUC', 'AP']:
                for key, results in result_dict_tmp.items():
                    result_dict[f'{metric} - {key}'] = results[metric]
                    result_dict[f'Count - {metric} - {key}'] = results['count']
                    result_dict[f'Skipped count - {metric} - {key}'] = results['skipped_count']

            if verbose:
                for key, value in result_dict.items():
                    print(f'{key}: {value:.3f}')

            return result_dict, eval_data_tmp

    def evaluate(self, batch, batch_unique, eval_data, description, verbose=True, log=False, epoch=None, save_latent_rep=False):
        labels_unique = batch_unique['label']
        with torch.no_grad():
            self.model.eval()
            if not self.params['debiasing']:
                predictions_unique, _, _ = self.model(batch_unique)
                t_cell_response_loss = F.binary_cross_entropy_with_logits(predictions_unique[:, 0], labels_unique).item()
            else:
                self.pep_source_adversary.eval()
                self.allele_adversary.eval()
                predictions_unique, source_pred_unique, allele_pred_unique = self.run_model(batch_unique)

                t_cell_response_loss = F.binary_cross_entropy_with_logits(predictions_unique[:, 0], labels_unique).item()

                # Turn on the evaluation mode
                pep_source_adversary_loss, allele_adversary_loss = self.get_adv_loss(batch_unique, source_pred_unique, allele_pred_unique)


            predictions, transformer_output, _ = self.model(batch)
            eval_data['Prediction'] = torch.sigmoid(predictions).numpy()

            if save_latent_rep:
                Path(self.params['visualization_dir']).mkdir(parents=True, exist_ok=True)
                np.savetxt(f'{self.params["visualization_data_file"]}_{epoch}.tsv', transformer_output.detach(), delimiter='\t')
                if epoch < 11:
                    eval_data[['MHC Class', 'Epitope Description', 'MHC Allele Prediction', 'Epitope Parent Species', 'Peptide Source', 'Assay Qualitative Measure', 'Prediction']].to_csv(self.params['visualization_meta_file'], sep='\t')

            result_dict_tmp = get_evaluation_dict(eval_data, description, eval_human=self.params['eval_human'])

            result_dict = {
                f'Loss/T-cell response - {description}': t_cell_response_loss,
            }

        for metric in ['ROC AUC', 'AP']:
            for key, results in result_dict_tmp.items():
                result_dict[f'{metric} - {key}'] = results[metric]
                result_dict[f'Count - {metric} - {key}'] = results['count']
                result_dict[f'Skipped count - {metric} - {key}'] = results['skipped_count']

        if self.params['debiasing']:
            debiasing_result_dict = {
                f'Adversary loss/Combined - {description}': pep_source_adversary_loss.item() + allele_adversary_loss.item(),
                f'Adversary loss/Peptide source adversary - {description}': pep_source_adversary_loss.item(),
                f'Adversary loss/Allele adversary - {description}': allele_adversary_loss.item(),
            }
            result_dict = {**result_dict, **debiasing_result_dict}

        if log:
            for key, value in result_dict.items():
                if 'Count - ' not in key and 'Skipped count - ' not in key:
                    self.writer.add_scalar(key, value, epoch)
            self.writer.flush()

        if verbose:
            for key, value in result_dict.items():
                if key in ['ROC AUC - MHC I+II - MHC+source corrected/Validation',
                           'ROC AUC - MHC I - MHC+source corrected - no defaults/Validation',
                           'ROC AUC - MHC I+II - MHC+source corrected - no defaults/Validation',
                           'ROC AUC - MHC II - MHC+source corrected - no defaults/Validation',
                           'ROC AUC - MHC I+II - MHC corrected - Human - no defaults/Validation',
                           'ROC AUC - MHC I - MHC corrected - Human - no defaults/Validation',
                           'ROC AUC - MHC II - MHC corrected - Human - no defaults/Validation',
                           'ROC AUC - MHC I+II - no MHC+source defaults/Validation',
                           'ROC AUC - MHC I+II/Validation',
                           'ROC AUC - MHC I+II - MHC+source corrected - no defaults/Training']:
                    print(f'{key}: {value:.3f}')
                elif self.params['debiasing'] and key in ['Loss/MHC allele adversary/Validation',
                                                          'Loss/Peptide source adversary/Validation']:
                    print(f'{key}: {value:.3f}')

        return result_dict, eval_data

    def get_lr_factor(self, step):
        if step <= self.steps_per_epoch:
            # one linear warmup epoch
            factor = (step+1) / (self.steps_per_epoch)
        else:
            # exponential learning rate decay
            epoch = step // self.steps_per_epoch
            factor = self.params['lr_gamma'] ** epoch
        return factor

    def save_model(self):
        Path(self.params['saved_models_dir']).mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            # 'allele_adversary_state_dict': self.allele_adversary.state_dict(),
            # 'pep_source_adversary_state_dict': self.pep_source_adversary.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            # 'adversary_optimizer_state_dict': self.adversary_optimizer.state_dict(),
            'params': self.params
        }, self.params['model_file'])
        print(f'Model saved: {self.params["model_file"]}')

    def load_model(self, saved_model):
        self.model.load_state_dict(saved_model['model_state_dict'])
        # self.allele_adversary.load_state_dict(saved_model['allele_adversary_state_dict'])
        # self.pep_source_adversary.load_state_dict(saved_model['pep_source_adversary_state_dict'])
        self.optimizer.load_state_dict(saved_model['optimizer_state_dict'])
        self.scheduler.load_state_dict(saved_model['scheduler_state_dict'])
        # self.adversary_optimizer.load_state_dict(saved_model['adversary_optimizer_state_dict'])

    def run_saved_model(self):
        global step
        step = 0
        global eval_optimizer_step
        eval_optimizer_step = 0

        test_results, test_prediction_df = self.evaluate(self.data.eval_batch, self.data.eval_batch_unique,
                                                         self.data.eval_data, 'Test', verbose=False,
                                                         log=False, epoch=self.params['early_stop_epochs'])

        if self.params['evaluate_pep_sources'] or self.params['evaluate_alleles']:
            self.evaluate_sources(verbose=True)

        input_file_dir = os.path.dirname(self.params['eval_file'])
        input_file_name = os.path.basename(self.params['eval_file'])
        prediction_file_path = cfg.file_path(input_file_dir, f'predictions_{self.params["test_experiment_name"]}_{input_file_name}')
        results_file_path = cfg.file_path(input_file_dir, f'results_{self.params["test_experiment_name"]}_{input_file_name}')

        results_df = pd.DataFrame([list(self.params.values()) + list(test_results.values())],
                                  columns=list(self.params.keys()) + list(test_results.keys()))
        results_df.T.to_csv(results_file_path)
        test_prediction_df.to_csv(prediction_file_path)
