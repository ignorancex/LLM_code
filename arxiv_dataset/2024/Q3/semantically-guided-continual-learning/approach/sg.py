import time
import torch
import torch.nn.functional as F

from copy import deepcopy
from clip import tokenize
from argparse import ArgumentParser
from .incremental_learning import Inc_Learning_Appr
from datasets.exemplars_dataset import ExemplarsDataset


class Appr(Inc_Learning_Appr):
    """Class implementing our proposed method"""

    def __init__(self, model, device, nepochs=100, lr=0.05, lr_min=1e-4, lr_factor=3, lr_patience=5, clipgrad=10000,
                 momentum=0, wd=0, multi_softmax=False, wu_nepochs=0, wu_lr_factor=1, fix_bn=False,
                 eval_on_train=False, nc_first=None, num_tasks=10, template="", logger=None,
                 exemplars_dataset=None, coef_kd=1, coef_sl=1, softness=10, T=4):
        super(Appr, self).__init__(model, device, nepochs, lr, lr_min, lr_factor, lr_patience, clipgrad, momentum, wd,
                                   multi_softmax, wu_nepochs, wu_lr_factor, fix_bn, eval_on_train, nc_first, num_tasks,
                                   template, logger, exemplars_dataset)
        self.T = T
        self.model_old = None
        self.coef_kd = coef_kd
        self.coef_sl = coef_sl
        self.softness = softness
        self.soft_labels = None

    @staticmethod
    def exemplars_dataset_class():
        return ExemplarsDataset

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument('--coef_kd', default=0.5, type=float, required=False,
                            help='coefficient for semantically-guided knowledge distillation loss')
        parser.add_argument('--coef_sl', default=0.5, type=float, required=False,
                            help='coefficient for semantically-guided representation learning loss')
        parser.add_argument('--softness', default=13, type=float, required=False,
                            help='control the softening of generated labels')
        parser.add_argument('--T', default=4, type=float, required=False,
                            help='temperature scaling for distillation loss')
        return parser.parse_known_args(args)

    def _get_optimizer(self, t):
        """Returns the optimizer"""
        for name, parameter in self.model.named_parameters():
            if 'vision_transformer.resblocks.11' not in name:
                parameter.requires_grad = False
        params = list(self.model.vision_transformer.resblocks[11].parameters())
        # double check
        self.check_trainable_params()
        return torch.optim.SGD(params, lr=self.lr, weight_decay=self.wd, momentum=self.momentum)

    def _init_optimizer(self, t=0):
        self.optimizer = self._get_optimizer(t)
        self.scheduler = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=1.0)

    def post_train_process(self, t, trn_loader):
        """Runs after training all the epochs of the task (after the train session)"""
        # Restore best and save model for future tasks
        self.model_old = deepcopy(self.model)
        self.model_old.eval()
        for param in self.model_old.parameters():
            param.requires_grad = False

    def train_loop(self, t, trn_loader, val_loader, num_old, num_cur):
        """Contains the epochs loop"""

        # add exemplars to train_loader
        if len(self.exemplars_dataset) > 0 and t > 0:
            trn_loader = torch.utils.data.DataLoader(trn_loader.dataset + self.exemplars_dataset,
                                                     batch_size=trn_loader.batch_size,
                                                     shuffle=True,
                                                     num_workers=trn_loader.num_workers,
                                                     pin_memory=trn_loader.pin_memory)

        # TRAINING -- contains the epochs loop
        self._init_optimizer(t)
        self.model.encode_current_text(self.template, self.device, num_old, num_cur)
        self.soft_labels = self.model.generate_soft_labels(num_cur, self.device, self.softness)
        cur_text_features = torch.cat(self.model.text_features_per_task, dim=0)
        prev_text_features = None
        if t > 0:
            prev_text_features = torch.cat(self.model.text_features_per_task[:t], dim=0)
        for e in range(self.nepochs):
            # Train
            train_loss, train_time = self.train_epoch(t, trn_loader, num_cur, cur_text_features, prev_text_features)
            self.scheduler.step()
            print('| Epoch {:3d}, time={:5.1f}s | lr: {:.5f}, loss={:.5f}'.format(e + 1, train_time,
                                                                                  self.optimizer.param_groups[0]['lr'],
                                                                                  train_loss / len(trn_loader)), end='')
            print()
        # EXEMPLAR MANAGEMENT -- select training subset
        self.exemplars_dataset.collect_exemplars(self.model, trn_loader, val_loader.dataset.transform)

    def train_epoch(self, t, trn_loader, num_cur, cur_text_features=None, prev_text_features=None):
        clock0 = time.time()
        self.model.train()
        losses = 0.
        for images, captions, targets in trn_loader:
            # Forward current model
            prompted_text = tokenize(captions)
            image_features = self.model.encode_image(images.to(self.device))
            text_features = self.model.encode_text(prompted_text.to(self.device))

            # normalized features
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)

            # cosine similarity as logits
            logit_scale = self.model.logit_scale.exp()
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()
            labels = torch.arange(len(logits_per_image)).to(self.device)
            image_loss = F.cross_entropy(logits_per_image, labels)
            text_loss = F.cross_entropy(logits_per_text, labels)
            contrastive_loss = (image_loss + text_loss) / 2

            pred = logit_scale * image_features @ cur_text_features.t()
            pred = F.log_softmax(pred, dim=1)
            target_distribution = self.make_batch_soft_labels(self.soft_labels, targets, num_cur, len(pred))
            soft_loss = F.kl_div(pred, target_distribution.to(self.device))
            if t > 0:
                with torch.no_grad():
                    features_old = self.model_old.encode_image(images.to(self.device))
                    features_old = features_old / features_old.norm(dim=1, keepdim=True)
                    inter_task_sim = self.model.calculate_inter_task_similarity(targets, self.device,
                                                                                prev_text_features)

                prev_pred = logit_scale * features_old @ prev_text_features.t()
                inter_task_sim = inter_task_sim.type(torch.HalfTensor).to(self.device)
                max_values, max_indices = torch.max(inter_task_sim, dim=1)
                max_values, max_indices = max_values.to(self.device), max_indices.to(self.device)
                max_values = max_values.unsqueeze(1)
                prev_pred += torch.where(torch.arange(prev_pred.shape[1]).unsqueeze(0).cuda() == max_indices.unsqueeze(1), max_values, -max_values)
                new_pred = logit_scale * image_features @ prev_text_features.t()
                kd_loss = self.dist_loss(new_pred, prev_pred.to(self.device))
                loss = contrastive_loss + self.coef_sl * soft_loss + self.coef_kd * kd_loss
            else:
                loss = contrastive_loss + self.coef_sl * soft_loss
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clipgrad)
            self.optimizer.step()
            losses += loss.item()
        clock1 = time.time()
        return losses, clock1 - clock0

    def make_batch_soft_labels(self, all_soft_labels, target, num_classes, batch_size):
        soft_labels = torch.zeros((batch_size, num_classes), dtype=torch.float32, device=self.device)
        for i in range(batch_size):
            this_label = all_soft_labels[:, target[i]]
            soft_labels[i, :] = this_label
        return soft_labels

    def dist_loss(self, y_s, y_t):
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction='sum') * (self.T ** 2) / y_s.shape[0]
        return loss

    def forward(self, images, captions, task_id=0, train=False, trained_classes=0):
        image_features = self.model.encode_image(images.to(self.device))
        text_features = torch.cat(self.model.text_features_per_task, dim=0)
        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # cosine similarity as logits
        logit_scale = self.model.model.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        return logits_per_image

    def criterion(self, t, outputs, targets):
        return F.cross_entropy(outputs, targets.to(self.device))

    def eval(self, t, val_loader, trained_classes, last, cur):
        """Contains the evaluation code"""
        with torch.no_grad():
            total_loss, total_acc_taw, total_acc_tag, total_num = 0, 0, 0, 0
            self.model.eval()
            for images, captions, targets in val_loader:
                # Forward current model
                image_features = self.model.encode_image(images.to(self.device))
                image_features = image_features / image_features.norm(dim=1, keepdim=True)
                taw_pred = self.model.task_aware_pred(image_features, t)
                tag_pred = self.model.task_agnos_pred(image_features)
                loss = self.criterion(t, tag_pred, targets.to(self.device))
                hits_taw, hits_tag = self.calculate_metrics(taw_pred, tag_pred, targets, last)
                # Log
                total_loss += loss.item() * len(targets)
                total_acc_taw += hits_taw.sum().item()
                total_acc_tag += hits_tag.sum().item()
                total_num += len(targets)
        return total_loss / total_num, total_acc_taw / total_num, total_acc_tag / total_num

    def calculate_metrics(self, taw_pred, tag_pred, targets, last):
        """Contains the main Task-Aware and Task-Agnostic metrics"""
        tag_probs = tag_pred.softmax(dim=-1).cpu()
        tag_top_labels = tag_probs.argmax(1)
        hits_tag = (tag_top_labels.to(self.device) == targets.to(self.device)).float()

        taw_probs = taw_pred.softmax(dim=-1).cpu()
        taw_top_labels = taw_probs.argmax(1)
        taw_top_labels += last
        hits_taw = (taw_top_labels.to(self.device) == targets.to(self.device)).float()
        return hits_taw, hits_tag
