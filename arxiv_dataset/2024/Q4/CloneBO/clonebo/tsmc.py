import torch
import gc
import numpy as np
from scipy.special import logsumexp
from .mcmc import energy_function
from .seq_tools import remove_spaces
from copy import deepcopy
from tqdm import tqdm
import time

class state:
    def __init__(self, clone, labeled_seqs, model, tokenizer, lik_func,
                 keep_on_gpu=False):
        past_key_values = None
        clone_w_labeled_seqs = (['']
                                 + [' ' + labeled_seq + ' '+ tokenizer.seq_sep_token
                                    for labeled_seq in labeled_seqs])
        if len(clone) > 0:
            clone_w_labeled_seqs = [clone + ' ' + tokenizer.seq_sep_token + seqs
                                    for seqs in clone_w_labeled_seqs]
            clone_w_labeled_seqs = [c.strip() for c in clone_w_labeled_seqs]
        # tokenize and pad clones
        tokens = [[tokenizer.bos_token] + x.split(" ") for x in clone_w_labeled_seqs]
        input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokens if len(x) > 0]
        max_len = max([len(x) for x in input_ids])
        input_ids = [[tokenizer.pad_token_id] * (max_len - len(x)) + x for x in input_ids]
        input_ids = torch.tensor(input_ids)
        if torch.cuda.is_available():
            input_ids = input_ids.cuda()

        # init funcs
        self.model = model
        self.labeled_seqs = labeled_seqs
        self.tokenizer = tokenizer
        self.lik_func = lik_func
        
        # init update logs
        self.input_ids = input_ids
        self.past_key_values = None
        self.keep_on_gpu = keep_on_gpu
        self.next_token = input_ids
        self.sum_zs = torch.zeros(len(labeled_seqs)).to(model.device).to(model.dtype)

        # init smc params
        self.on_gpu = keep_on_gpu
        self.log_w = 0
        self.log_bias = 0
        self.p_y_tm1 = lik_func(torch.tensor(0.))[0]

    def put_on_gpu(self):
        if self.past_key_values is not None and not self.on_gpu:
            self.past_key_values = [[k.to('cuda') for k in kv] for kv in self.past_key_values]
        self.on_gpu = True

    def take_off_gpu(self):
        if self.past_key_values is not None and self.on_gpu:
            self.past_key_values = [[k.to('cpu') for k in kv] for kv in self.past_key_values]
        self.on_gpu = False
    
    def update(self, n_steps=1, max_n_seqs=1e7):
        t0 = time.time()
        if not self.keep_on_gpu:
            self.put_on_gpu()
        t1 = time.time()
        for n in range(n_steps):
            if self.num_seqs() <= max_n_seqs:
                self._single_update()
        t2 = time.time()
        if not self.keep_on_gpu:
            self.take_off_gpu()
        t3 = time.time()
        return [t1-t0 + t3-t2, t2-t1]
        
    def _single_update(self):
        # pkv must be on gpu
        with torch.no_grad():
            model_output = self.model(input_ids=self.next_token,
                                      attention_mask=(self.input_ids != self.tokenizer.pad_token_id).float(),
                                      past_key_values=self.past_key_values)
            self.past_key_values = model_output.past_key_values
            # get the next token logits
            logits = model_output.logits[:, -1, :]
            # logits[self.tokenizer.pad_token_id] = -1e7
            logits -= torch.logsumexp(logits, -1, keepdim=True)
            # calc lik diff
            self.sum_zs = (logits[1:] - logits[0]).T + self.sum_zs[None, :]
            with torch.no_grad():
                liks_y = self.lik_func(self.sum_zs)
    
            # sample
            sample_logits = liks_y + logits[0]
            sample_logits -= torch.logsumexp(sample_logits, -1, keepdim=True)
            self.next_token = torch.multinomial(torch.softmax(sample_logits, dim=-1), num_samples=1) 
            self.next_token = self.next_token[None, :].expand(len(self.input_ids), -1)
            # update
            self.input_ids = torch.cat([self.input_ids, self.next_token], dim=-1) # add the new token to the current generation
            self.sum_zs = self.sum_zs[self.next_token[0, 0]]
            # update weight
            bias = sample_logits[self.next_token[0, 0]] - logits[0][self.next_token[0, 0]]
            self.log_bias += bias
            self.log_w += (liks_y[self.next_token[0, 0]] - self.p_y_tm1) - bias
            self.p_y_tm1 = liks_y[self.next_token[0, 0]]

    def get_string(self):
        return self.tokenizer.batch_decode(self.input_ids, skip_special_tokens=True)[0]

    def num_seqs(self):
        return len(self.get_string().split(self.tokenizer.seq_sep_token))

    def update_weight(self, new_weight):
        self.weight = new_weight

    @staticmethod
    def copy_state(old_state):
        # weight set to 0!
        clone = old_state.get_string()
        new_state = state(clone, old_state.labeled_seqs, old_state.model, old_state.tokenizer, old_state.lik_func)
        new_state.input_ids = old_state.input_ids
        new_state.past_key_values = old_state.past_key_values
        new_state.next_token = old_state.next_token
        new_state.sum_zs = old_state.sum_zs
        new_state.log_bias = old_state.log_bias
        new_state.p_y_tm1 = old_state.p_y_tm1
        new_state.log_w = old_state.log_w * 0
        new_state.keep_on_gpu = old_state.keep_on_gpu
        return new_state


class smc:
    def __init__(self, args, model, tokenizer, clones, labeled_seqs, labels,
                 sample_one_seq=False, keep_on_gpu=False):
        start_E, _, lik_funcs = energy_function(
            args, model, tokenizer, clones, labeled_seqs, labels,
        )
        # print("Starting E:", start_E)

        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.labeled_seqs = labeled_seqs
        self.labels = labels
        self.n_particles = len(clones)
        self.lik_funcs = lik_funcs
        self.states = [state(clone, labeled_seqs, model, tokenizer, lik_func, keep_on_gpu)
                       for clone, lik_func in zip(clones, lik_funcs)]
        self.start_n_seqs = self.num_seqs(clones[0])
        self.sample_one_seq = sample_one_seq
        self.ess = []
        self.keep_on_gpu = keep_on_gpu

    def num_seqs(self, clone):
        if len(clone) == 0:
            return 0
        else:
            return len(clone.split(self.tokenizer.seq_sep_token))

    def update(self, n_steps=1):
        max_n_seqs = self.start_n_seqs + 1
        updated = False
        times = np.zeros(2)
        for state in self.states:
            if (not self.sample_one_seq
                or self.num_seqs(state.get_string()) <= max_n_seqs):
                times += state.update(n_steps, max_n_seqs=max_n_seqs)
                updated = True
        return updated, times

    def get_weights(self):
        return np.array([state.log_w.to('cpu') for state in self.states])

    def get_biases(self):
        return np.array([state.log_bias.to('cpu') for state in self.states])

    def get_ess(self):
        weights = self.get_weights()
        log_ess = 2 * logsumexp(weights) - logsumexp(2 * weights)
        return np.exp(log_ess)

    def get_clones(self):
        clones = [state.get_string() for state in self.states]
        clones = [self.tokenizer.seq_sep_token.join(
            [s for s in clone.split(self.tokenizer.seq_sep_token)
             if len(s.strip())>0]) # strip ending sep tokens
            for clone in clones]
        return clones

    def print_states(self):
        clones = self.get_clones()
        weights = self.get_weights()
        biases = self.get_biases()
        probs = np.exp(weights - logsumexp(weights))
        for clone, prob, bias in zip(clones, probs, biases):
            print()
            print(prob)
            print(bias)
            for seq in remove_spaces(clone.split(self.tokenizer.seq_sep_token)):
                print(seq)

    def resample(self):
        weights = np.array([state.log_w.to('cpu') for state in self.states])
        probs = np.exp(weights - logsumexp(weights))
        samples = np.random.multinomial(self.n_particles, probs)
        inds = np.repeat(np.arange(self.n_particles), samples)
        new_states = []
        for i in inds:
            old_state = self.states[i] 
            new_states.append(state.copy_state(old_state))
        self.states = new_states

    def energy_resample(self):
        [state.take_off_gpu() for state in self.states]
        p_ys = - energy_function(
            self.args, self.model, self.tokenizer,
            self.get_clones(), self.labeled_seqs, self.labels,
        )[0]
        if self.keep_on_gpu:
            [state.put_on_gpu() for state in self.states]
        pyts = np.array([state.p_y_tm1.to('cpu') for state in self.states])
        weights = np.array([state.log_w.to('cpu') for state in self.states])
        weights = weights + p_ys - pyts
        probs = np.exp(weights - logsumexp(weights))
        samples = np.random.multinomial(self.n_particles, probs)
        inds = np.repeat(np.arange(self.n_particles), samples)
        new_states = []
        for i in inds:
            old_state = self.states[i] 
            new_states.append(state.copy_state(old_state))
        self.states = new_states

    def run_smc(self, n_steps, steps_per_update=5):
        for n in (pbar := tqdm(range(n_steps))):
            updated, times = self.update(steps_per_update)
            self.ess.append(self.get_ess())
            pbar.set_description(f"Running tSMC. ESS: {np.round(self.ess[-1], 2)}")
            if not updated:
                break
            if self.ess[-1] < np.sqrt(self.n_particles):
                print("Resampling!")
                self.resample()

    @staticmethod
    def refresh_smc(args, old_smc, energy_resample=False):
        if energy_resample:
            old_smc.energy_resample()
        else:
            old_smc.resample()
        clones = old_smc.get_clones()
        new_smc = smc(args, old_smc.model, old_smc.tokenizer, clones,
                       old_smc.labeled_seqs, old_smc.labels,
                       old_smc.sample_one_seq, old_smc.keep_on_gpu)
        new_smc.ess = old_smc.ess
        return new_smc




