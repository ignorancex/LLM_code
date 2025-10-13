import pickle

with open("../result_dict.pt",'rb') as f:
    obj = pickle.load(f)
    for i in range(200):
        print(obj[i]['pred_seq'])
        print(obj[i]['name'])
        #print(obj[i]['gt_seq'])
        #print(obj[i]['gt_coord'])