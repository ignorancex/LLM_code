from xoa.commons.logger import * 


class TrainerPrototype(object):

    def __init__(self, *args, **kwargs):
        self.history = []

    def initialize(self):
        self.history = []

    def add_train_history(self, curve, train_time=None, cur_epoch=None, measure='test_accuracy'):
        h = {
            "curve": curve,
            "measure" : measure 
        }
        if train_time:
            h["train_time"] = train_time
        if cur_epoch: 
            h["train_epoch"] = cur_epoch
        else:
            h["train_epoch"] = len(curve)
                    
        self.history.append(h)

    def train(self, cand_index, train_epoch=None, estimates=None, space=None):
        raise NotImplementedError("This should return loss and duration.")

    def get_interim_error(self, model_index, cur_dur=0):
        raise NotImplementedError("This should return interim loss.")

    def get_acc_curve(self, i, start_index=0, end_index=None):
        if i >= len(self.history):
            raise ValueError("Trial index {} > history size".format(i))
        
        if 'accuracy' in self.history[i]['measure']:
            acc_curve = self.history[i]["curve"]
            curve_length = len(acc_curve)
            if end_index == None:
                end_index = curve_length - 1
            if start_index >= curve_length:                
                return []

            if end_index >= curve_length:
                end_index = curve_length - 1

            return acc_curve[start_index:end_index+1]
        else:
            return []

    def get_verifier(self):
        return None