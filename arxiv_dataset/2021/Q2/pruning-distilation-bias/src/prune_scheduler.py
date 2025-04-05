class AgpPruningRate(object):
    def __init__(self, initial_sparsity, final_sparsity, 
                 starting_epoch, ending_epoch, freq):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.starting_epoch = starting_epoch
        self.ending_epoch = ending_epoch
        self.freq = freq
        self.last_sparsity = 0
    
    def step(self, current_epoch):
        span = ((self.ending_epoch - self.starting_epoch - 1) // self.freq) * self.freq
        target_sparsity = (self.final_sparsity + 
                           (self.initial_sparsity - self.final_sparsity) *
                           (1.0 - ((current_epoch - self.starting_epoch)/span))**3)
        
        if current_epoch < self.ending_epoch:
            return target_sparsity
        else:
            return 0