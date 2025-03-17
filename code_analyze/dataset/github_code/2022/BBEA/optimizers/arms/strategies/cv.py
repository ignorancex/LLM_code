from xoa.commons.logger import *
from .div import DiversificationStrategy

import numpy as np


class CVSelectionStrategy(DiversificationStrategy):
    def __init__(self, arms, 
                values, counts, 
                choosers, 
                title="",
                check_iters=10):
        name = "CVS_" + title
        super(CVSelectionStrategy, self).__init__(name, len(arms), values, counts)
        self.arms = arms
        self.check_iters = check_iters
        self.choosers = choosers

    def evaluate(self, metric='top1'):       
        all_nominees = []
        for i in range(len(self.arms)):
            arm = self.arms[i]
            optimizer = arm['model']
            aquisition_func = arm['acq_func']
            chooser = self.choosers[optimizer]
            score = chooser.estimate(aquisition_func, metric=metric)

            all_nominees.append({
                "arm_index": i,
                "name": '{}-{}'.format(optimizer, aquisition_func),
                "score" : score
            })
    
        return all_nominees

    def next(self, step):
        best_i = 0
        if step < self.check_iters * 2:
            idx = step % self.num_arms
        
        elif step >= self.check_iters:
            if step % self.check_iters == 0:
                models = self.evaluate()
                # choose best scored model
                best_score = 0.0
                best_model = models[0]['name']
                best_i = 0
                for m in models:
                    if m["score"] > best_score:
                        best_score = m["score"]
                        best_i = m['arm_index']
                        best_model = m['name']
                info('Best model {} is only used to choose next from {} iterations.'.format(best_model, step))
            idx = best_i

        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        for i in range(len(self.values)):
            self.values[i] = float(self.counts[i]) / float(sum(self.counts))


class CVDropStrategy(DiversificationStrategy):
    def __init__(self, arms, 
                values, counts, 
                choosers, 
                title="",
                check_iters=10):
        name = "CVD_" + title

        super(CVDropStrategy, self).__init__(name, len(arms), values, counts)
        self.arms = arms
        self.check_iters = check_iters
        self.choosers = choosers
        self.good_models = []

    def evaluate(self, metric='top1'):       
        all_nominees = []
        for i in range(len(self.arms)):
            arm = self.arms[i]
            optimizer = arm['model']
            aquisition_func = arm['acq_func']
            chooser = self.choosers[optimizer]
            score = chooser.estimate(aquisition_func, metric=metric)

            all_nominees.append({
                "arm_index": i,
                "name": '{}-{}'.format(optimizer, aquisition_func),
                "score" : score
            })
    
        return all_nominees

    def next(self, step):
        if step < self.check_iters * 2:
            idx = step % self.num_arms
        
        elif step >= self.check_iters:
            if step % self.check_iters == 0:
                s_t = time.time()
                self.good_models = []
                models = self.evaluate()
                # drop models which have 
                mean_score = float(sum([ m['score'] for m in models ]) / len(models))
                for m in models:
                    if m["score"] >= mean_score:
                        self.good_models.append(m)
                #info('[CVD] {} models will be used to choose next from {} iterations.'.format(len(self.good_models), step))
                dur = time.time() - s_t
                debug('Model drop by CV takes {:.2f} sec. {} surviors over threshold {:.4f}: {}'.format(dur, len(models), mean_score, [m['name'] for m in models]))
            # randomize a good model selection
            rand = np.random.randint(0, len(self.good_models))
            idx = self.good_models[rand]['arm_index']

        return idx

    def update(self, arm_index, curr_acc, opt):
        self.counts[arm_index] += 1
        for i in range(len(self.values)):
            self.values[i] = float(self.counts[i]) / float(sum(self.counts))