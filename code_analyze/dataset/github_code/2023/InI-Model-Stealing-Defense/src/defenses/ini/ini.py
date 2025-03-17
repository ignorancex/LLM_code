import itertools
import os
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from torch.distributions import Categorical

from datasets import REGISTRY_INFO, REGISTRY_TRAIN_DATASET, REGISTRY_TEST_DATASET
from models import get_family, REGISTRY_MODEL

from .knockoff import Knockoff

class INI:
    def __init__(self, attack_args, model, dataset, dataset_root, dataset_ood, dataset_ood_root, exp_path, lr_step=50, lr_gamma=0.1, test_knockoff=True, use_ie=False,kl_reverse=False, batch_size=128, epochs=150, num_workers=0, student_path=None, student_iter=5, victim_iter=1, student_lr=0.1, victim_lr=0.1, weight_decay=1e-3, momentum=0.5, scheduler="multi", use_pcgrad=False, sched_gamma=0.3, sched_milestones=[20, 60, 100], clip_grad=True, max_grad_norm=10, loss="l1", grad_loss="kl", beta=0.01, gamma=0.1, delta=0.1, epsilon=0.1, save_interval=10, log_interval=100, device=torch.device("cuda"), **kwargs):
        self.attack_args = attack_args

        self.model = model
        self.dataset = dataset
        self.dataset_root = dataset_root

        self.dataset_ood = dataset_ood
        self.dataset_ood_root = dataset_ood_root

        self.exp_path = exp_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.epochs = epochs
        self.student_iter = student_iter
        self.victim_iter = victim_iter
        self.student_lr = student_lr
        self.victim_lr = victim_lr
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.scheduler = scheduler
        self.use_pcgrad = use_pcgrad
        self.sched_gamma = sched_gamma
        self.sched_milestones = sched_milestones
        self.clip_grad = clip_grad
        self.max_grad_norm = max_grad_norm
        self.loss = loss
        self.grad_loss = grad_loss
        self.beta = beta
        self.save_interval = save_interval
        self.log_interval = log_interval
        self.device = device

        self.use_ie = use_ie
        self.test_knockoff = test_knockoff
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.kl_reverse = kl_reverse
        self.lr_step = lr_step
        self.lr_gamma = lr_gamma

        self.student_path = student_path
        
        self.log_dir = os.path.join(self.exp_path, "logs")
        self.save_dir = self.exp_path

    def loss_func(self, victim, clone):
        if self.loss == "l1":
            victim_ = F.log_softmax(victim, dim=-1)
            victim__ = victim_ - victim_.mean(dim=-1, keepdim=True)
            loss = torch.abs(clone - victim__).mean()
        elif self.loss == "kl":
            x = torch.log_softmax(clone, dim=-1)
            y = victim.softmax(dim=-1)
            y_logits = torch.log_softmax(victim, dim=-1)
            loss_ = y * (y_logits - x)
            loss = loss_.sum(dim=-1).mean()
        elif self.loss == "ce":
            x = torch.log_softmax(clone, dim=-1)
            y = victim.softmax(dim=-1)
            loss_ = y * x
            loss = loss_.sum(dim=-1).mean()
        else:
            raise NotImplementedError
        return loss
    
    def grad_loss_func(self, victim, clone):
        if self.grad_loss == "l1":
            victim_ = F.log_softmax(victim, dim=-1)
            victim__ = victim_ - victim_.mean(dim=-1, keepdim=True)
            loss = torch.abs(clone - victim__).mean()
        elif self.grad_loss == "kl":
            x = torch.log_softmax(clone, dim=-1)
            y = victim.softmax(dim=-1)
            y_logits = torch.log_softmax(victim, dim=-1)
            loss_ = y * (y_logits - x)
            loss = loss_.sum(dim=-1).mean()
        elif self.grad_loss == "ce":
            x = torch.log_softmax(clone, dim=-1)
            y = victim.softmax(dim=-1)
            loss_ = y * x
            loss = loss_.sum(dim=-1).mean()
        else:
            raise NotImplementedError
        return loss

    def train(self):
        family = get_family(self.dataset)
        print("Building dataset: {}, {} family, num_classes {}, image_size {}, channels {}.".format(self.dataset, family, *REGISTRY_INFO[self.dataset]))
        train_set = REGISTRY_TRAIN_DATASET[self.dataset](root=self.dataset_root)
        test_set = REGISTRY_TEST_DATASET[self.dataset](root=self.dataset_root)
        ood_set = REGISTRY_TRAIN_DATASET[self.dataset_ood](root=self.dataset_ood_root)

        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        test_loader = DataLoader(test_set, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        ood_loader_v = DataLoader(ood_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        # ood_loader_s = DataLoader(ood_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        ood_iter_v = itertools.cycle(ood_loader_v)
        # ood_iter_s = itertools.cycle(ood_loader_s)

        print("Building models, victim: {}, student: {}".format(self.model, self.model))

        victim_model = REGISTRY_MODEL[family][self.model](REGISTRY_INFO[self.dataset][0]).to(self.device)
        student_model = REGISTRY_MODEL[family][self.model](REGISTRY_INFO[self.dataset][0]).to(self.device)

        victim_optimizer = torch.optim.SGD(victim_model.parameters(), lr=self.victim_lr, momentum=self.momentum,
                                        weight_decay=self.weight_decay)
        student_optimizer = torch.optim.SGD(student_model.parameters(), lr=self.student_lr, momentum=self.momentum,
                                            weight_decay=self.weight_decay)

        if self.scheduler == "multi":
            victim_scheduler = torch.optim.lr_scheduler.MultiStepLR(victim_optimizer, self.sched_milestones,
                                                                    self.sched_gamma)
            student_scheduler = torch.optim.lr_scheduler.MultiStepLR(student_optimizer, self.sched_milestones,
                                                                    self.sched_gamma)
        elif self.scheduler == "cosine":
            victim_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(victim_optimizer, self.epochs)
            student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, self.epochs)
        elif self.scheduler == "steplr":
            victim_scheduler = torch.optim.lr_scheduler.StepLR(victim_optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
            student_scheduler = torch.optim.lr_scheduler.StepLR(student_optimizer, step_size=self.lr_step, gamma=self.lr_gamma)
        else:
            raise NotImplementedError
        
        if self.use_pcgrad:
            from .pcgrad import PCGrad
            victim_optimizer = PCGrad(victim_optimizer)

        print("Tensorboard logs save to", self.log_dir)
        logger = SummaryWriter(log_dir=self.log_dir)

        if self.student_path:
            student_model.load_state_dict(torch.load(self.student_path))

        victim_acc = self.eval(victim_model, test_loader)
        print("Accuracy of victim model: {:.2f}%".format(victim_acc * 100))

        best_acc = 0
        best_epoch = -1
        epoch_iters = len(train_loader)
        times = 0
        for ep in range(self.epochs):
            begin = time.time()
            id_iter = iter(train_loader)
            self.t = 0
            while self.t < epoch_iters:
                self.train_V(epoch_iters, id_iter, victim_model, student_model, ood_iter_v, victim_optimizer, ep, logger)
                # self.train_S(ood_iter_s, victim_model, student_model, student_optimizer)

            if self.student_iter != 0:
                student_scheduler.step()
            victim_scheduler.step()

            # evaluation process
            student_model.eval()
            if ep == 0 or self.student_iter != 0:
                student_acc = self.eval(student_model, test_loader)

            victim_model.eval()
            victim_acc = self.eval(victim_model, test_loader)

            # log and save process
            if victim_acc > best_acc:
                best_acc = victim_acc
                best_epoch = ep
                torch.save(victim_model.state_dict(), os.path.join(self.save_dir, "model_best.pth"))

            if (ep + 1) in [120, 150]:
                torch.save(victim_model.state_dict(), os.path.join(self.save_dir, f"model_{ep + 1}.pth"))
            if (ep + 1) % self.save_interval == 0 or (ep + 1) == self.epochs:
                torch.save(victim_model.state_dict(), os.path.join(self.save_dir, "model.pth"))

            print("Epoch {}/{}, Accuracy of victim model: {:.2f}%, student model: {:.2f}% ({:.2f}x of victim)".format(
                ep + 1, self.epochs, victim_acc * 100, student_acc * 100, student_acc / victim_acc))
            logger.add_scalar("test/victim_accuracy", victim_acc, (ep + 1) * epoch_iters)
            logger.add_scalar("test/student_accuracy", student_acc, (ep + 1) * epoch_iters)
            logger.add_scalar("test/relative_accuracy", student_acc / victim_acc, (ep + 1) * epoch_iters)

            if ep + 1 in [120, 150, self.epochs] and self.test_knockoff:
                steal_acc, relative_acc = self.knockoff(victim_model, test_loader, victim_acc)
                logger.add_hparams({"beta": self.beta, "gamma": self.gamma, "delta": self.delta, "epsilon": self.epsilon, "pcgrad": self.use_pcgrad, "student_iter": self.student_iter, "epoch": ep + 1},
                                  {"victim_acc": victim_acc, "steal_acc": steal_acc, "relative_acc": relative_acc})

            time_ = time.time() - begin
            times += time_

            h = times // 3600
            m = (times - h * 3600) // 60
            s = times - h * 3600 - m * 60
            print("Time passed: %dh%dm%ds" % (h, m, s), end=', ')
            left_s = time_ * (self.epochs - ep - 1)
            h = left_s // 3600
            m = (left_s - h * 3600) // 60
            s = left_s - h * 3600 - m * 60
            print("Remaining Time: %dh%dm%ds\n" % (h, m, s))

        print("End of training, best accuracy: {:.2f}%, in epoch {}.".format(best_acc * 100, best_epoch + 1))


    def eval(self, model, loader):
        correct = 0
        total = 0
        model.eval()
        for data, label in loader:
            data, label = data.to(self.device), label.to(self.device)
            pred = torch.argmax(model(data), dim=-1)
            correct += torch.sum(pred == label).item()
            total += label.shape[0]
        acc = correct / total
        return acc


    def train_V(self, epoch_iters, id_iter, victim_model, student_model, ood_iter_v, victim_optimizer, ep, logger):
        for i in range(self.victim_iter):
            if self.t == epoch_iters:
                break
            self.t += 1

            # forward process
            victim_model.train()
            student_model.eval()

            id_data, label = next(id_iter)
            id_data, label = id_data.to(self.device), label.to(self.device)
            ood_data, _ = next(ood_iter_v)
            ood_data = ood_data.to(self.device)
            
            logits_victim_id = victim_model(id_data)
            logits_student_id = student_model(id_data)
            logits_victim_ood = victim_model(ood_data)
            logits_student_ood = student_model(ood_data)

            # loss calculation process
            # 1. Benign classification loss: CE(V(x), y)
            loss_benign = F.cross_entropy(logits_victim_id, label)
            # 2. In-distribution distance loss
            loss_id = -self.loss_func(logits_victim_id, logits_student_id.detach())
            # 3. Out-of-distribution distance loss
            if self.use_ie:
                dist_ood = Categorical(logits=logits_victim_ood)
                loss_ood = - dist_ood.entropy().mean()
            else:
                loss_ood = self.loss_func(logits_victim_ood, logits_student_ood.detach())
            # 4. Grad direction loss
            loss1 = self.grad_loss_func(logits_student_id, logits_victim_id)
            loss2 = F.cross_entropy(logits_student_id, label)
            grad1 = torch.cat([e.flatten() for e in torch.autograd.grad(loss1, student_model.parameters(), create_graph=True)])
            grad2 = torch.cat([e.flatten() for e in torch.autograd.grad(loss2, student_model.parameters(), retain_graph=True)]).detach()
            loss_grad_id = F.cosine_similarity(grad1, grad2, dim=0)
            # 5. Out-of-distribution grad loss
            loss3 = self.grad_loss_func(logits_student_ood, logits_victim_ood)
            loss_grad_ood = torch.abs(torch.cat([e.flatten() for e in torch.autograd.grad(loss3, student_model.parameters(), create_graph=True)])).mean()
            
            loss = loss_benign + self.beta * loss_id + self.gamma * loss_ood + self.delta * loss_grad_id + self.epsilon * loss_grad_ood

            # backward process
            if self.use_pcgrad:
                losses = [loss_benign]
                if self.beta != 0:
                    losses.append(self.beta * loss_id)
                if self.gamma != 0:
                    losses.append(self.gamma * loss_ood)
                if self.delta != 0:
                    losses.append(self.delta * loss_grad_id)
                if self.epsilon != 0:
                    losses.append(self.epsilon * loss_grad_ood)
                victim_optimizer.pc_backward(losses)
            else:
                victim_optimizer.zero_grad()
                loss.backward()
                if self.clip_grad:
                    grad_norm = clip_grad_norm_(victim_model.parameters(), max_norm=self.max_grad_norm)
            victim_optimizer.step()

        if self.t % self.log_interval == 0 or self.t == epoch_iters - 1:
            print("Epoch {}/{}, Iteration {}/{}, total loss: {:.4f}, benign loss: {:.4f}, id loss: {:.4f}, ood loss: {:.4f}, grad_id_loss: {:.4f}, grad_ood_loss: {:.4f}".format(
                ep + 1, self.epochs, self.t + 1, epoch_iters, loss.item(), loss_benign.item(), loss_id.item(), loss_ood.item(), loss_grad_id.item(), loss_grad_ood.item()
            ))
            logger.add_scalar("train/victim_loss", loss.item(), ep * epoch_iters + self.t)
            logger.add_scalar("train/victim_loss_benign", loss_benign.item(), ep * epoch_iters + self.t)
            logger.add_scalar("train/victim_loss_id", loss_id.item(), ep * epoch_iters + self.t)
            logger.add_scalar("train/victim_loss_ood", loss_ood.item(), ep * epoch_iters + self.t)
            logger.add_scalar("train/victim_loss_grad_id", loss_grad_id.item(), ep * epoch_iters + self.t)
            logger.add_scalar("train/victim_loss_grad_ood", loss_grad_ood.item(), ep * epoch_iters + self.t)
            if (not self.use_pcgrad) and self.clip_grad:
                logger.add_scalar("train/grad_norm", grad_norm.item(), ep * epoch_iters + self.t)


    def train_S(self, ood_iter_s, victim_model, student_model, student_optimizer):
        student_model.train()
        victim_model.eval()
        for i in range(self.student_iter):
            x, _ = next(ood_iter_s)
            x = x.to(self.device)

            with torch.no_grad():
                logits_victim = victim_model(x).detach()

            logits_student = student_model(x)

            loss = self.loss_func(logits_victim, logits_student)

            student_optimizer.zero_grad()
            loss.backward()
            student_optimizer.step()
    

    def knockoff(self, model, test_loader, victim_acc):
        attack = Knockoff(teacher=model, **self.attack_args)
        sur_set = REGISTRY_TRAIN_DATASET[self.attack_args.sur_dataset](root=self.attack_args.sur_dataset_root)
        sur_loader = torch.utils.data.DataLoader(sur_set, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        steal_acc, relative_acc = attack.run(test_loader, sur_loader, victim_acc)
        print("knockoff steal acc: {:.2f}%({:.2f}x)".format(steal_acc * 100, relative_acc))
        return steal_acc, relative_acc
        

