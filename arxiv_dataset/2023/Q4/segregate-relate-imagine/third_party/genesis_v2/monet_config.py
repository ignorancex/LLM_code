# =========================== A2I Copyright Header ===========================
#
# Copyright (c) 2003-2021 University of Oxford. All rights reserved.
# Authors: Applied AI Lab, Oxford Robotics Institute, University of Oxford
#          https://ori.ox.ac.uk/labs/a2i/
#
# This file is the property of the University of Oxford.
# Redistribution and use in source and binary forms, with or without
# modification, is not permitted without an explicit licensing agreement
# (research or commercial). No warranty, explicit or implicit, provided.
#
# =========================== A2I Copyright Header ===========================

import torch
import torch.nn.functional as F

from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence


class MONet:

    @staticmethod
    def get_mask_recon_stack(m_r_logits_k, prior_mode, log):
        if prior_mode == 'softmax':
            if log:
                return F.log_softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
            return F.softmax(torch.stack(m_r_logits_k, dim=4), dim=4)
        elif prior_mode == 'scope':
            log_m_r_k = []
            log_s = torch.zeros_like(m_r_logits_k[0])
            for step, logits in enumerate(m_r_logits_k):
                if step == len(m_r_logits_k) - 1:
                    log_m_r_k.append(log_s)
                else:
                    log_a = F.logsigmoid(logits)
                    log_neg_a = F.logsigmoid(-logits)
                    log_m_r_k.append(log_s + log_a)
                    log_s = log_s +  log_neg_a
            log_m_r_stack = torch.stack(log_m_r_k, dim=4)
            return log_m_r_stack if log else log_m_r_stack.exp()
        else:
            raise ValueError("No valid prior mode.")

    @staticmethod
    def kl_m_loss(log_m_k, log_m_r_k, debug=False):
        if debug:
            assert len(log_m_k) == len(log_m_r_k)
        batch_size = log_m_k[0].size(0)
        m_stack = torch.stack(log_m_k, dim=4).exp()
        m_r_stack = torch.stack(log_m_r_k, dim=4).exp()
        # Lower bound to 1e-5 to avoid infinities
        m_stack = torch.max(m_stack, torch.tensor(1e-5))
        m_r_stack = torch.max(m_r_stack, torch.tensor(1e-5))
        q_m = Categorical(m_stack.view(-1, len(log_m_k)))
        p_m = Categorical(m_r_stack.view(-1, len(log_m_k)))
        kl_m_ppc = kl_divergence(q_m, p_m).view(batch_size, -1)
        return kl_m_ppc.sum(dim=1)
