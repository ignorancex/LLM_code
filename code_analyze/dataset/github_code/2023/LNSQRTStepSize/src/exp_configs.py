import os
import itertools

from haven import haven_utils as hu

ours_opt_list = [  
           "SGD_1sqrt_Decay"           
            ]


EXP_GROUPS = {
            
	"mushrooms":{"dataset":["mushrooms"],
            "model":["linear"],
            "loss_func": ["logistic_loss"],
            "opt":[{"name":"SGD_1sqrt_Decay"}],
            "acc_func":["logistic_accuracy"],
            "batch_size":[800],
            "max_epoch":[100],
            "runs":[0]},
	
	
	"a1a":{"dataset":["a1a"],
            "model":["linear"],
            "loss_func": ["logistic_loss"],
            "opt":[{"name":"SGD_1sqrt_Decay"}],
            "acc_func":["logistic_accuracy"],
            "batch_size":[800],
            "max_epoch":[100],
            "runs":[0]},
            
	"a2a":{"dataset":["a2a"],
            "model":["linear"],
            "loss_func": ["logistic_loss"],
            "opt":[{"name":"SGD_1sqrt_Decay"}],
            "acc_func":["logistic_accuracy"],
            "batch_size":[800],
            "max_epoch":[100],
            "runs":[0]},
         
         "w1a":{"dataset":["w1a"],
            "model":["linear"],
            "loss_func": ["logistic_loss"],
            "opt":[{"name":"SGD_1sqrt_Decay"}],
            "acc_func":["logistic_accuracy"],
            "batch_size":[800],
            "max_epoch":[100],
            "runs":[0]},
           
	
	"rcv1":{"dataset":["rcv1"],
            "model":["linear"],
            "loss_func": ["logistic_loss"],
            "opt":[{"name":"SGD_1sqrt_Decay"}],
            "acc_func":["logistic_accuracy"],
            "batch_size":[100],
            "max_epoch":[100],
            "runs":[0]},
         
	"phishing":{"dataset":["phishing"],
            "model":["linear"],
            "loss_func": ["logistic_loss"],
            "opt":[{"name":"SGD_1sqrt_Decay"}],
            "acc_func":["logistic_accuracy"],
            "batch_size":[800],
            "max_epoch":[100],
            "runs":[0]}
            
            }

EXP_GROUPS = {k:hu.cartesian_exp_group(v) for k,v in EXP_GROUPS.items()}
