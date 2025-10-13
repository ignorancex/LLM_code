import torch
import numpy as np
from tqdm import tqdm
from .utils import cuda, loss_nll_flatten
from model import RDesign_Model
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
import pickle
from copy import deepcopy

alphabet = 'AUCG'
pre_base_pairs = {0: 1, 1: 0, 2: 3, 3: 2}
pre_great_pairs = ((0, 1), (1, 0), (2, 3), (3, 2))

class RDesign:
    def __init__(self, args, device, steps_per_epoch):
        self.args = args
        self.device = device
        self.config = args.__dict__

        self.model = self._build_model()

    def _build_model(self):
        return RDesign_Model(self.args).to(self.device)

    def _cal_recovery(self, dataset, featurizer):
        recovery = []
        S_preds, S_trues = [], []
        S_preds_short, S_trues_short = [], []
        S_preds_long, S_trues_long = [], []
        S_preds_medium, S_trues_medium = [], []
        recovery_short, recovery_long, recovery_medium = [], [], []
        f1_short_all, f1_medium_all, f1_long_all = [], [], []
        paris_correct_rate_short, pairs_correct_rate_medium, paris_correct_rate_long = [], [], []
        none_pair = 0
        none_pair_short = 0
        none_pair_medium = 0
        none_pair_long = 0
        result_list = []
        one_result_dict = {}

        for sample in tqdm(dataset):
            sample = featurizer([sample])
            X, S, mask, lengths, clus, ss_pos, ss_pair, names = sample
            #print(S)
            one_result_dict['name'] = names
            one_result_dict['gt_seq'] = S
            one_result_dict['gt_coord'] = X
            #print(ss_pair,"################\n" ,ss_pos)
            X, S, mask, ss_pos = cuda((X, S, mask, ss_pos), device=self.device)
            #logits, gt_S_temp = self.model.sample(X=X, S_gtt=S, mask=mask)
            #gt_S = gt_S_temp[0]
            logits, gt_S = self.model.sample(X=X, S_gtt=S, mask=mask)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # secondary sharpen
            ss_pos = ss_pos[mask == 1].long()
            log_probs = log_probs.clone()
            log_probs[ss_pos] = log_probs[ss_pos] / self.args.ss_temp
            S_pred = torch.argmax(log_probs, dim=1)
            one_result_dict['pred_seq'] = S_pred.cpu().numpy()
            pos_log_probs = log_probs.softmax(-1)
            result_list.append(deepcopy(one_result_dict))
            good_pairs = 0
            all_pairs = 0
            for pair in ss_pair[0]:
                all_pairs += 1
                s_pos_a, s_pos_b = pair
                if s_pos_a == None or s_pos_b == None or s_pos_b >= S_pred.shape[0]:
                    continue
                if (S_pred[s_pos_a].item(), S_pred[s_pos_b].item()) in pre_great_pairs:
                    good_pairs += 1
                    continue
                '''
                if pos_log_probs[s_pos_a][S_pred[s_pos_a]] > pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                    S_pred[s_pos_b] = pre_base_pairs[S_pred[s_pos_a].item()]
                elif pos_log_probs[s_pos_a][S_pred[s_pos_a]] < pos_log_probs[s_pos_b][S_pred[s_pos_b]]:
                    S_pred[s_pos_a] = pre_base_pairs[S_pred[s_pos_b].item()]
                '''
            #cmp_0 = S_pred.eq(gt_S_temp[0])
            #cmp_1 = S_pred.eq(gt_S_temp[1])
            #cmp = torch.logical_or(cmp_0, cmp_1)
            cmp = S_pred.eq(gt_S)
            recovery_ = cmp.float().mean().cpu().numpy()
            S_preds += S_pred.cpu().numpy().tolist()
            S_trues += gt_S.cpu().numpy().tolist()
            if np.isnan(recovery_): recovery_ = 0.0
            recovery.append(recovery_)
            if lengths[0] <= 50:
                S_preds_short += S_pred.cpu().numpy().tolist()
                S_trues_short += gt_S.cpu().numpy().tolist()
                recovery_short.append(recovery_)
                _,_,f1,_ = precision_recall_fscore_support(gt_S.cpu().numpy().tolist(), S_pred.cpu().numpy().tolist(), average=None)
                f1_short_all.append(f1.mean())
                if all_pairs !=0:
                    paris_correct_rate_short.append(good_pairs/all_pairs)
                else:
                    none_pair += 1
                    none_pair_short += 1
            elif lengths[0]>50 and lengths[0]<=100:
                S_preds_medium += S_pred.cpu().numpy().tolist()
                S_trues_medium += gt_S.cpu().numpy().tolist()
                recovery_medium.append(recovery_)
                _,_,f1,_ = precision_recall_fscore_support(gt_S.cpu().numpy().tolist(), S_pred.cpu().numpy().tolist(), average=None)
                f1_medium_all.append(f1.mean())
                if all_pairs !=0:
                    pairs_correct_rate_medium.append(good_pairs/all_pairs)
                else:
                    none_pair += 1
                    none_pair_medium += 1
            else:
                S_preds_long += S_pred.cpu().numpy().tolist()
                S_trues_long += gt_S.cpu().numpy().tolist()
                recovery_long.append(recovery_)
                _,_,f1,_ = precision_recall_fscore_support(gt_S.cpu().numpy().tolist(), S_pred.cpu().numpy().tolist(), average=None)
                f1_long_all.append(f1.mean())
                if all_pairs !=0:
                    paris_correct_rate_long.append(good_pairs/all_pairs)
                else:
                    none_pair += 1
                    none_pair_long += 1
        fw = open("result_dict.pt", 'wb')
        pickle.dump(result_list, fw)
        recovery_meidan = np.median(recovery)
        recovery_mean = np.mean(recovery)
        precision, recall, f1, _ = precision_recall_fscore_support(S_trues, S_preds, average=None)
        _, _, f1_short, _ = precision_recall_fscore_support(S_trues_short, S_preds_short, average=None)
        _, _, f1_medium, _ = precision_recall_fscore_support(S_trues_medium, S_preds_medium, average=None)
        _, _, f1_long, _ = precision_recall_fscore_support(S_trues_long, S_preds_long, average=None)

        macro_f1 = f1.mean()
        print('macro f1', macro_f1)
        print('f1 short', f1_short.mean())
        print('f1 medium', f1_medium.mean())
        print('f1 long', f1_long.mean())
        print('f1 short mean', np.mean(f1_short_all))
        print('f1 medium mean', np.mean(f1_medium_all))
        print('f1 long mean', np.mean(f1_long_all))
        print('pairs_correct_rate_short', np.mean(paris_correct_rate_short))
        print('pairs_correct_rate_medium', np.mean(pairs_correct_rate_medium))
        print('pairs_correct_rate_long', np.mean(paris_correct_rate_long))
        print("No pair RNA:", none_pair, len(f1_short_all)+len(f1_medium_all)+len(f1_long_all))
        print("No pair short:", none_pair_short, len(f1_short_all))
        print("No pair medium:", none_pair_medium, len(f1_medium_all))
        print("No pair long:", none_pair_long, len(f1_long_all))
        
        return recovery_meidan, recovery_mean, np.mean(recovery_short), np.mean(recovery_medium), np.mean(recovery_long)

    def pretrain_one_epoch(self, train_loader, optimizer, criterion):
        train_sum, train_weight = 0., 0.
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
            X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
            optimizer.zero_grad()
            logits, S, _ = self.model.pretrain(X, S, mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss, loss_av = loss_nll_flatten(S, log_probs, mask)
            loss_av.backward()
            optimizer.step()
            #scheduler.step()
            train_sum += torch.sum(loss).cpu().data.numpy()
            train_weight += len(loss)
            train_pbar.set_description('train loss: {:.4f}'.format(loss.mean().item()))
        train_loss = train_sum / train_weight
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity


    def train_one_epoch(self, train_loader, optimizer, criterion):
        train_sum, train_weight = 0., 0.
        train_pbar = tqdm(train_loader)
        for batch in train_pbar:
            X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
            X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
            optimizer.zero_grad()
            logits, S, _ = self.model(X, S, mask)
            log_probs = F.log_softmax(logits, dim=-1)
            loss, loss_av = loss_nll_flatten(S, log_probs, mask)
            loss_av.backward()
            optimizer.step()
            #scheduler.step()
            train_sum += torch.sum(loss).cpu().data.numpy()
            train_weight += len(loss)
            train_pbar.set_description('train loss: {:.4f}'.format(loss.mean().item()))
        train_loss = train_sum / train_weight
        train_perplexity = np.exp(train_loss)
        return train_loss, train_perplexity
    
    def valid_one_epoch(self, valid_loader):
        self.model.eval()
        with torch.no_grad():
            valid_sum, valid_weights = 0., 0.
            valid_pbar = tqdm(valid_loader)
            for batch in valid_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                logits, S, _ = self.model(X, S, mask)
                
                log_probs = F.log_softmax(logits, dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs, mask)
                
                valid_sum += torch.sum(loss).cpu().data.numpy()
                valid_weights += len(loss)
                valid_pbar.set_description('valid loss: {:.4f}'.format(loss.mean().item()))
        
        valid_loss = valid_sum / valid_weights
        valid_perplexity = np.exp(valid_loss)        
        return valid_loss, valid_perplexity

    def test_one_epoch(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_pbar = tqdm(test_loader)
            for batch in test_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                #logits, S, _ = self.model(X, S[0], mask)
                logits, S, _ = self.model(X, S, mask)
                log_probs = F.log_softmax(logits, dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs, mask)
                test_sum += torch.sum(loss).cpu().data.numpy()
                test_weights += len(loss)
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))
            test_recovery, recovery_mean, recovery_short, recovery_medium, recovery_long = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery, recovery_mean, recovery_short, recovery_medium, recovery_long

    def test_one_epoch_copy(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            test_sum, test_weights = 0., 0.
            test_pbar = tqdm(test_loader)
            for batch in test_pbar:
                X, S, mask, lengths, clus, ss_pos, ss_pair, names = batch
                X, S, mask, lengths, clus, ss_pos = cuda((X, S, mask, lengths, clus, ss_pos), device=self.device)
                logits, S, _ = self.model(X, S, mask)

                log_probs = F.log_softmax(logits, dim=-1)
                loss, _ = loss_nll_flatten(S, log_probs, mask)
                
                test_sum += torch.sum(loss).cpu().data.numpy()
                test_weights += len(loss)
                test_pbar.set_description('test loss: {:.4f}'.format(loss.mean().item()))

            test_recovery, recovery_mean, recovery_short, recovery_medium, recovery_long = self._cal_recovery(test_loader.dataset, test_loader.featurizer)
            
        test_loss = test_sum / test_weights
        test_perplexity = np.exp(test_loss)
        return test_perplexity, test_recovery, recovery_mean, recovery_short, recovery_medium, recovery_long