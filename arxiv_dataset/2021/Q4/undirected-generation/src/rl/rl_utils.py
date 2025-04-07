import torch

def compute_advantage(bleu_score, critic, mask, discount_factor):
    full_reward = torch.zeros(mask.size(0), mask.size(1)).cuda()
    cur_reward = bleu_score

    for i in reversed(range(mask.size(1))):
        full_reward[:, i] = cur_reward * mask[:, i]
        cur_reward = cur_reward * torch.pow(discount_factor, mask[:, i])

    critic = critic.detach() # Block the gradient from baseline_fn
    return full_reward - critic * mask, full_reward


def add_vtarg_and_adv(bleu_scores, critic, mask, gamma=0.99, lam=0.95):
    """
    Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
    """
    #new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    #vpred = np.append(seg["vpred"], seg["nextvpred"])
    vpred = critic.detach()
    T = mask.size(1)
    #seg["adv"] = \
    adv = gaelam = torch.zeros(mask.size(0), T).cuda()
    #rew = seg["rew"]
    #lastgaelam = 0
    lastgaelam = bleu_scores

    for t in reversed(range(T-1)):
        #nonterminal = 1-new[t+1]
        nonterminal = 1
        cur_reward = bleu_scores * mask[:, t]
        rew = 0
        #print(vpred[t+1].size())
        #print(mask[:, t].size())
        delta = (rew + gamma * vpred[:, t+1] * nonterminal - vpred[:, t]) * mask[:, t]
        lastgaelam = delta + torch.pow(gamma, mask[:, t]) * torch.pow(lam, mask[:, t]) * nonterminal * lastgaelam
        gaelam[:, t] = lastgaelam * mask[:, t]
        #seg["tdlamret"] = seg["adv"] + seg["vpred"]
    tdlamret = adv + vpred
    return adv, tdlamret

# def compute_td_gae(seg, gamma, lam):
#     """
#     Compute target value using TD(lambda) estimator, and advantage with GAE(lambda)
#     """
#     #new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
#     vpred = np.append(seg["vpred"], seg["nextvpred"])
#     T = len(seg["rew"])
#     seg["adv"] = gaelam = np.empty(T, 'float32')
#     rew = seg["rew"]
#     lastgaelam = 0
#     for t in reversed(range(T)):
#         nonterminal = True
#         delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
#         gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
#     seg["tdlamret"] = seg["adv"] + seg["vpred"]