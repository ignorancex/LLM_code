# import necessary packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import logging
import numpy as np


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)



class BayesWrap(nn.Module):
    def __init__(self, opt, NET):
        super().__init__()

        num_particles = opt.num_particles
        self.h_kernel = 0
        self.particles = []

        for i in range(num_particles):
            self.particles.append(copy.deepcopy(NET))
            # set init weights for different particle
            self.particles[i].apply(init_weights)


        for i, particle in enumerate(self.particles):
            self.add_module(str(i), particle)

        logging.info("num particles: %d" % len(self.particles))

    def sample_particle(self):
        return self.particles[np.random.randint(0, len(self.particles))]

    def get_particle(self, index):
        return self.particles[index]

    def get_losses(self, image, labels, criterion, **kwargs):
        losses = []
        for particle in self.particles:
            l = particle(image)
            loss = criterion(l, labels)
            losses.append(loss)
        losses = torch.stack(losses).mean(0)
        return losses


    def forward(self, x, **kwargs):
        logits, entropies = [], []
        return_entropy = "return_entropy" in kwargs and kwargs["return_entropy"]
        for particle in self.particles:
            l = particle(x)
            logits.append(l)
            if return_entropy:
                l = torch.softmax(l, 1)
                entropies.append((-l * torch.log(l + 1e-8)).sum(1))
        logits = torch.stack(logits).mean(0)

        if return_entropy:
            entropies = torch.stack(entropies).mean(0)
            return logits, entropies
        return logits

    def update_grads(self):
        if np.random.rand() < 0.6:
            return
        all_pgs = self.particles
        if self.h_kernel <= 0:
            self.h_kernel = 0.1  # 1
        dists = []
        alpha = 0.01  # if t < 100 else 0.0
        new_parameters = [None] * len(all_pgs)

        for i in range(len(all_pgs)):
            new_parameters[i] = {}
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is None:
                    new_parameters[i][l] = None
                else:
                    new_parameters[i][l] = p.grad.data.new(
                        p.grad.data.size()).zero_()
            for j in range(len(all_pgs)):
                # if i == j:
                #     continue
                for l, params in enumerate(
                        zip(all_pgs[i].parameters(), all_pgs[j].parameters())):
                    p, p2 = params
                    if p.grad is None or p2.grad is None:
                        continue
                    if p is p2:
                        dists.append(0)
                        new_parameters[i][l] = new_parameters[i][l] + \
                            p.grad.data
                    else:
                        d = (p.data - p2.data).norm(2)
                        # if p is not p2:
                        dists.append(d.cpu().item())
                        kij = torch.exp(-(d**2) / self.h_kernel**2 / 2)
                        new_parameters[i][l] += kij * (p2.grad.data + (alpha / self.h_kernel**2) * d)
                        new_parameters[i][l] /= float(len(all_pgs))

        self.h_kernel = np.median(dists)
        self.h_kernel = np.sqrt(0.5 * self.h_kernel / np.log(len(all_pgs)) + 1)
        for i in range(len(all_pgs)):
            for l, p in enumerate(all_pgs[i].parameters()):
                if p.grad is not None:
                    p.grad.data = new_parameters[i][l]
