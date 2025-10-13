import os
import gc
import torch
import torch.nn as nn
import logging
import numpy as np
from time import time
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Dict, List

from utils import utils
from models.BaseModel import BaseModel
from functools import reduce

class BaseRunner(object):
    @staticmethod
    def evaluate_method(output_dict: dict, topk: list, metrics: list, id_multihot: dict, users_gender:dict) -> Dict[str, float]:
        predictions: np.ndarray = output_dict["predictions"]
        evaluations = dict()
        gt_rank = (predictions >= predictions[:,0].reshape(-1,1)).sum(axis=-1)
        for k in topk:
            hit = (gt_rank <= k)
            for metric in metrics:
                key = '{}@{}'.format(metric, k)
                if metric == 'HR':
                    evaluations[key] = hit.mean()
                elif metric == 'NDCG':
                    evaluations[key] = (hit / np.log2(gt_rank + 1)).mean()
                elif metric == 'ALPHA_NDCG':
                    scores = BaseRunner.best_alpha_nDCG(output_dict, id_multihot, 0.5, k)
                    evaluations[key] = BaseRunner.alpha_ndcg(output_dict, id_multihot, 0.5,k, normaliztion = scores)
                elif metric == 'NDCG_GAP':
                    evaluations[key] = BaseRunner.calculate_fairness_ndcg(output_dict, users_gender, k)
                else:
                    raise ValueError('Undefined evaluation metric: {}.'.format(metric))
        return evaluations
    
    @staticmethod
    def alpha_ndcg(output_dict: dict, id_multihot: Dict[int, list], alpha, k, normaliztion) -> float:
        batch_predictions = output_dict["predictions"]
        batch_ids = output_dict["dataset_ids"]
        a_ndcg_list = []
        category = []
        for i in range(batch_ids.shape[0]):
            sorted_indices = np.argsort(batch_predictions[i])[::-1]
            sorted_ids = batch_ids[i][sorted_indices]
            grade_list = np.zeros((len(sorted_ids), len(id_multihot[str(sorted_ids[0])])))
            for j, id_ in enumerate(sorted_ids):
                grade_list[j] = id_multihot[str(id_)]
            grade_list = np.transpose(np.array(grade_list))
            
            category.append(BaseRunner.test_category(grade_list)[k])

            alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
            grade_list = (grade_list>0).astype(int)
            cum = reduce(lambda t,i:  [t[0]+(grade_list[:,i]),
                                    t[1]+np.sum(np.dot(np.power(alpha,t[0]), grade_list[:,i]))/np.log2(i+2)],
                        range(k), [np.zeros(n_subtopics), 0])[1]
            a_ndcg_list.append(cum)

        result = [a / b for a, b in zip(a_ndcg_list, normaliztion)]
        return np.mean(result)
    
    @staticmethod
    def test_category(grade_list):
        grad_array = np.array(grade_list)
        topics = grad_array.sum(axis=0).tolist()
        for i in range(1,len(topics)):
            topics[i] = topics[i]+topics[i-1]
        return topics

    @staticmethod    
    def best_alpha_nDCG(output_dict: dict, id_multihot: Dict[int, list], alpha, k):
        batch_predictions = output_dict["predictions"]
        batch_ids = output_dict["dataset_ids"]
        score_batch = []
        for i in range(batch_ids.shape[0]):
            sorted_indices = np.argsort(batch_predictions[i])[::-1]
            sorted_ids = batch_ids[i][sorted_indices]
            grade_list = np.zeros((len(sorted_ids), len(id_multihot[str(sorted_ids[0])])))
            for j, id_ in enumerate(sorted_ids):
                grade_list[j] = id_multihot[str(id_)]
            alpha, n_subtopics, n_docs = 1 - alpha, grade_list.shape[0], grade_list.shape[1]
            grade_list = (grade_list > 0).astype(int)
            mask = np.zeros(n_docs)
            discount = np.zeros(n_subtopics)
            score, rank, score_list = 0, [], []
            for i in range(k):
                scores = np.matmul(grade_list.T, np.power(alpha, discount)) + mask
                r = np.argmax(scores)
                discount += grade_list[:,r]
                score += scores[r]/np.log2(i+2)
                score_list.append(score)
                rank.append(r)
                mask[r] = np.finfo(np.float32).min
            score_batch.append(score)
        return score_batch

    def __init__(self, args):
        self.accuracy_weight = args.accuracy_weight 
        self.diversity_weight = args.diversity_weight
        self.finetune_acc_weight = args.finetune_acc_weight
        self.finetune_div_weight = args.finetune_div_weight
        self.pretrain_epochs = args.pretrain_epochs
        self.finetune_epochs = args.finetune_epochs
        self.is_finetune = False

        self.model_path_root = args.model_path_root
        self.save_epoch = args.save_epoch
        self.train_models = args.train
        self.check_epoch = args.check_epoch
        self.test_epoch = args.test_epoch
        self.early_stop = args.early_stop
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.eval_batch_size = args.eval_batch_size
        self.l2 = args.l2
        self.optimizer_name = args.optimizer
        self.num_workers = args.num_workers
        self.pin_memory = args.pin_memory
        self.topk = [int(x) for x in args.topk.split(',')]
        self.metrics = [m.strip().upper() for m in args.metric.split(',')]
        self.main_metric = '{}@{}'.format(self.metrics[0], self.topk[1]) if not len(args.main_metric) else args.main_metric
        self.main_topk = int(self.main_metric.split("@")[1])
        self.time = None

        self.log_path = os.path.dirname(args.log_file)
        self.save_appendix = args.log_file.split("/")[-1].split(".")[0]
        self.stage = 1

    def _check_time(self, start=False):
        if self.time is None or start:
            self.time = [time()] * 2
            return self.time[0]
        tmp_time = self.time[1]
        self.time[1] = time()
        return self.time[1] - tmp_time

    def _build_optimizer(self, model):
        if self.is_finetune:
            params = [p for n, p in model.named_parameters() if 'adapter' in n and p.requires_grad]
            lr = self.learning_rate
        else:
            params = [p for n, p in model.named_parameters() if 'adapter' not in n and p.requires_grad]
            lr = self.learning_rate

        optimizer = eval('torch.optim.{}'.format(self.optimizer_name))(params, lr=lr, weight_decay=self.l2)
        return optimizer
    
    def freeze_layers(self, model, n_last_layers):
        for param in model.parameters():
            param.requires_grad = False
        
        trainable_params = list(model.parameters())[-n_last_layers:]
        for param in trainable_params:
            param.requires_grad = True

    def train(self, data_dict: Dict[str, BaseModel.Dataset]):
        model = data_dict['train'].model
        if not os.path.exists(self.model_path_root):
            os.makedirs(self.model_path_root)

        main_metric_results, dev_results = list(), list()
        self._check_time(start=True)

        try:
            # pretrain
            logging.info("Starting pretrain phase")
            self.is_finetune = False
            model.use_adapter = False  # adapter isn't uesd in the phase
            for epoch in range(self.pretrain_epochs):
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                if np.isnan(loss):
                    logging.info(f"Loss is Nan. Stop training at {epoch+1}.")
                    break
                training_time = self._check_time()

                if len(model.check_list) > 0 and self.check_epoch > 0 and epoch % self.check_epoch == 0:
                    utils.check(model.check_list)

                if(epoch + 1 >= 0):
                    dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
                    dev_results.append(dev_result)
                    main_metric_results.append(dev_result[self.main_metric])
                    logging_str = f'Pretrain Epoch {epoch + 1:<5} loss={loss:<.4f} [{training_time:<3.1f} s]    dev=({utils.format_metric(dev_result)})'

                    testing_time = self._check_time()
                    logging_str += f' [{testing_time:<.1f} s]'

                    if max(main_metric_results) == main_metric_results[-1] or (hasattr(model, 'stage') and model.stage == 1):
                        logging_str += ' *'
                    logging.info(logging_str)

            # save the pretrained model
            model_path = f'{self.model_path_root}pretrain_final.pt'
            model.save_model(model_path)
            logging.info("Pretraining completed. Model saved.")

            # fintune
            self.is_finetune = True
            model.use_adapter = True  # use adapter
            model.optimizer = None # Rebuild the optimizer
            self.accuracy_weight = self.finetune_acc_weight
            self.diversity_weight = self.finetune_div_weight

            logging.info("Starting finetune phase")
            for epoch in range(self.finetune_epochs):
                self._check_time()
                gc.collect()
                torch.cuda.empty_cache()
                loss = self.fit(data_dict['train'], epoch=epoch + 1)
                if np.isnan(loss):
                    logging.info(f"Loss is Nan. Stop training at Stage 2 Epoch {epoch+1}.")
                    break
                training_time = self._check_time()
                dev_result = self.evaluate(data_dict['dev'], [self.main_topk], self.metrics)
                dev_results.append(dev_result)
                main_metric_results.append(dev_result[self.main_metric])
                logging_str = f'Finetune Epoch {epoch + 1:<5} loss={loss:<.4f} [{training_time:<3.1f} s]    dev=({utils.format_metric(dev_result)})'
                testing_time = self._check_time()
                logging_str += f' [{testing_time:<.1f} s]'
                
                if epoch > self.save_epoch:
                    model_path = f'{self.model_path_root}finetune_epoch{epoch - self.save_epoch}.pt'
                    model.save_model(model_path)
                logging.info(logging_str)

        except KeyboardInterrupt:
            logging.info("Early stop manually")
            exit_here = input("Exit completely without evaluation? (y/n) (default n):")
            if exit_here.lower().startswith('y'):
                logging.info(os.linesep + '-' * 45 + ' END: ' + utils.get_time() + ' ' + '-' * 45)
                exit(1)

    def fit(self, dataset: BaseModel.Dataset, epoch=-1) -> float:
        model = dataset.model
        if self.is_finetune:
            for name, param in model.named_parameters():
                if 'adapter' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        else:
            for name, param in model.named_parameters():
                if 'adapter' not in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

        if model.optimizer is None:
            model.optimizer = self._build_optimizer(model)
        model.train()

        loss_lst = list()
        dl = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        
        for batch in tqdm(dl, leave=False, desc=f'Epoch {epoch:<3}', ncols=100, mininterval=1):
            batch = utils.batch_to_gpu(batch, model.device)
            item_ids = batch['item_id']
            indices = torch.argsort(torch.rand(*item_ids.shape), dim=-1)        
            batch['item_id'] = item_ids[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices]

            model.optimizer.zero_grad()
            out_dict = model(batch)
            
            prediction = out_dict['prediction']
            if len(prediction.shape)==2:
                restored_prediction = torch.zeros(*prediction.shape).to(prediction.device)
                restored_prediction[torch.arange(item_ids.shape[0]).unsqueeze(-1), indices] = prediction   
                out_dict['prediction'] = restored_prediction
                out_dict['item_id'] = item_ids

            accuracy_loss = model.loss(out_dict)
            diversity_loss = model.diversity_loss(out_dict)
            fairness_loss = model.fairness_loss(out_dict,batch['user_id'])
            loss = float(self.accuracy_weight) * accuracy_loss + float(self.diversity_weight) * diversity_loss + float(self.fairness_weight) * fairness_loss

            loss.backward()
            model.optimizer.step()
            loss_lst.append(loss.detach().cpu().data.numpy())

        return np.mean(loss_lst).item()

    def eval_termination(self, criterion: List[float]) -> bool:
        if len(criterion) > self.early_stop and utils.non_increasing(criterion[-self.early_stop:]):
            return True
        elif len(criterion) - criterion.index(max(criterion)) > self.early_stop:
            return True
        return False

    def evaluate(self, dataset: BaseModel.Dataset, topks: list, metrics: list) -> Dict[str, float]:
        """
        Evaluate the results for an input dataset.
        :return: result dict (key: metric@k)
        """
        id_multihot = dataset.model.item_multihot_mapping
        output_dict = self.predict(dataset)
        return self.evaluate_method(output_dict, topks, metrics, id_multihot)

    def predict(self, dataset: BaseModel.Dataset, save_prediction: bool = False) -> Dict[str, np.ndarray]:
        """
        The returned prediction is a 2D-array, each row corresponds to all the candidates,
        and the ground-truth item poses the first.
        Example: ground-truth items: [1, 2], 2 negative items for each instance: [[3,4], [5,6]]
                 predictions like: [[1,3,4], [2,5,6]]
        """
        dataset.model.eval()
        predictions = list()
        dl = DataLoader(dataset, batch_size=self.eval_batch_size, shuffle=False, num_workers=self.num_workers,
                        collate_fn=dataset.collate_batch, pin_memory=self.pin_memory)
        dataset_ids = []
        for batch in tqdm(dl, leave=False, ncols=100, mininterval=1, desc='Predict'):
            dataset_ids.extend(batch['item_id'].cpu().numpy().tolist())
            if hasattr(dataset.model,'inference'):
                prediction = dataset.model.inference(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
            else:
                prediction = dataset.model(utils.batch_to_gpu(batch, dataset.model.device))['prediction']
            predictions.extend(prediction.cpu().data.numpy())
        predictions = np.array(predictions)
        dataset_ids = np.array(dataset_ids)

        if dataset.model.test_all:
            rows, cols = list(), list()
            for i, u in enumerate(dataset.data['user_id']):
                clicked_items = list(dataset.corpus.train_clicked_set[u] | dataset.corpus.residual_clicked_set[u])
                idx = list(np.ones_like(clicked_items) * i)
                rows.extend(idx)
                cols.extend(clicked_items)
            predictions[rows, cols] = -np.inf
        return {"predictions": predictions, "dataset_ids": dataset_ids}

    def print_res(self, dataset: BaseModel.Dataset) -> str:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        res_str = '(' + utils.format_metric(result_dict) + ')'
        return res_str
    
    def easy_print_res(self, dataset: BaseModel.Dataset) -> Dict[str, float]:
        """
        Construct the final result string before/after training
        :return: test result string
        """
        result_dict = self.evaluate(dataset, self.topk, self.metrics)
        easy_result_dict = {
            'NDCG@10': result_dict['NDCG@10'],
            'ALPHA_NDCG@10': result_dict['ALPHA_NDCG@10']
        }
        return easy_result_dict
