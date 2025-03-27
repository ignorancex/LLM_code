from nn_utils import *
import numpy as np
from z3 import *

def sample_search(probability, label, seq_len):
    max_len = seq_len.max()
    train_X = []
    train_Y = []
    for latent, res, length in zip(probability, label, seq_len):
        prob = latent.clone().cpu().detach()
        max_prob, pred = torch.max(prob, dim=-1)
        # 0 correction
        expr_pred, res_pred = eval_pred(pred.data.cpu().numpy(), length)
        if (res_pred-res).abs() < 1e-10:
            train_X.append(latent[0:length, :])
            train_Y.append(pred[0:length])
            continue
    if len(train_X) > 0 and len(train_Y) > 0:
        train_X = torch.cat(train_X, dim=0).cuda()
        train_Y = torch.cat(train_Y, dim=0).long().cuda()
        return train_X, train_Y
    else:
        return None, None

def operation(a, b, op):
    """
    :param a: val
    :param b: val
    :param op: 10->+; 11->-; 12->*; 13->/
    :return: val
    """
    if op == 10:
        return a+b
    elif op == 11:
        return a-b
    elif op == 12:
        return a*b
    elif op == 13:
        return a/b

def check(pred, sol):
    # check
    pred = [p if p <= 9 else p-9 for p in pred]
    s = Solver()
    c = RealVal(1.0)
    X = [Int('x%s' % i) for i in range(1)]
    for i in range(1):
        s.add(X[i]>=0, X[i]<=9)
    explist = [pred[0], X[0], pred[1], pred[2]]
    
    nums = list()
    r0 = operation(pred[0], pred[1], 12)
    r1 = operation(X[0], pred[2], 12)
    nums.append(operation(r0, r1, 10))

    s.add(nums[0] - sol >= -1e-5)
    s.add(nums[0] - sol <= 1e-5)
    if s.check() == sat:
        return True, [s.model()[X[i]].as_long() for i in range(1)]
    else:
        return False, None

def init_check(pred, sol):
    # check
    pred = [p if p <= 9 else p-9 for p in pred]
    s = Solver()
    c = RealVal(1.0)
    X = [Int('x%s' % i) for i in range(arity)]
    for i in range(arity):
        s.add(X[i]>=0, X[i]<=9)
    explist = [X[0], X[1], pred[0], pred[1]]

    nums = list()
    r0 = operation(X[0], pred[0], 12)
    r1 = operation(X[1], pred[1], 12)
    nums.append(operation(r0, r1, 10))

    s.add(nums[0] - sol >= -1e-5)
    s.add(nums[0] - sol <= 1e-5)
    if s.check() == sat:
        return True, [s.model()[X[i]].as_long() for i in range(arity)]
    else:
        return False, None


# def check(pred, sol):
#     # check
#     pred = [p if p <= 9 else p-9 for p in pred]
#     s = Solver()
#     c = RealVal(1.0)
#     X = [Int('x%s' % i) for i in range(1)]
#     for i in range(1):
#         s.add(X[i]>=1, X[i]<=9)
#     explist = [pred[0], X[0], pred[1], pred[2]]
    
#     nums = list()
#     r0 = operation(pred[0], pred[1], 12)
#     r1 = operation(X[0], pred[2], 12)
#     nums.append(operation(r0, r1, 10))

#     s.add(nums[0] - sol >= -1e-5)
#     s.add(nums[0] - sol <= 1e-5)
#     if s.check() == sat:
#         return True, [s.model()[X[i]].as_long() for i in range(1)]
#     else:
#         return False, None

# def init_check(pred, sol):
#     # check
#     pred = [p if p <= 9 else p-9 for p in pred]
#     s = Solver()
#     c = RealVal(1.0)
#     X = [Int('x%s' % i) for i in range(arity)]
#     for i in range(arity):
#         s.add(X[i]>=0, X[i]<=9)
#     explist = [X[0], X[1], pred[0], pred[1]]

#     nums = list()
#     r0 = operation(X[0], pred[0], 12)
#     r1 = operation(X[1], pred[1], 12)
#     nums.append(operation(r0, r1, 10))

#     s.add(nums[0] - sol >= -1e-5)
#     s.add(nums[0] - sol <= 1e-5)
#     if s.check() == sat:
#         return True, [s.model()[X[i]].as_long() for i in range(arity)]
#     else:
#         return False, None



if __name__ == "__main__":

    # pred = [11, 12, 11]
    # sol = 5.2222
    # print(init_check(pred, sol))

    pred = [5, 11, 12, 10, 5]
    sol = 0.0
    print(check(pred, sol))










