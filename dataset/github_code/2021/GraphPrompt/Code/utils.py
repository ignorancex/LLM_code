import time
import numpy as np

disabled_tags = set(['make batch', 'candidates_retrieve_separate', 'get_batch_inputs_for_stage_1', 
                    'bert_candidate_generator', 'optimizer', 'get emb', 
                    'eval cross encoder', 'cross encoder cls', 'cross encoder score network'])

class TimeIt:
    start = 0
    end = -1
    def __init__(self, tag='NA'):
        self.tag = tag

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, type, value, tb):
        self.end = time.time()
        if self.tag not in disabled_tags:
            print('tag={} time={:.2e}ms'.format(self.tag, (self.end-self.start)*1000))


def random_walk_restart(A, R = 0.2, max_iters = 100):
    """
    A : adj matrix
    R : reset prob
    """
    N = A.shape[0] # n node
    init_x = np.eye(N)
    x = init_x

    T = A / (A.sum(1, keepdims=True) + 1e-8) # transition probabilities. each row is normed
    
    for i in range(max_iters):
        old_x = x
        x = (1 - R) * (x.dot(T)) + (R * init_x)
        if np.linalg.norm(old_x-x) <= 1e-6:
            print('break')
            break
        print(x)
    return x
