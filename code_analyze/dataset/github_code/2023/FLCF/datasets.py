import torch.utils.data as data
import numpy as np
import torch
import pdb


class NCFData(data.Dataset):
    def __init__(self, interaction_file, num_neg):
        # create a csr matrix for negative sampling
        users = []
        items = []
        negs = {}
        item_max = -1
        self.num_neg = num_neg

        with open(interaction_file, "r") as f:
            for line in f.readlines():
                arr = line.strip().split(" ")
                arr = [int(a) for a in arr]
                users.append(arr[0])
                items.append(np.array(arr[1:]))
                item_max = max(item_max, np.max(arr[1:]))

        if num_neg > 0:
            for i in range(len(users)):
                all_items = np.ones(item_max + 1)
                all_items[items[i]] = 0
                
                negs[users[i]] = np.argwhere(all_items > 0.5).reshape(-1)

          
        # create a dataset helper
        users_l = []
        self.user_starts = {}
        self.user_ends = {}
        idx_temp = 0
        for i in range(len(users)):
            self.user_starts[users[i]] = idx_temp
            users_l.append(np.array([users[i]] * len(items[i])))
            idx_temp += len(np.array([users[i]] * len(items[i])))
            self.user_ends[users[i]] = idx_temp

        self.user_data = np.concatenate(users_l)
        self.item_data = np.concatenate(items)
        self.item_max = item_max
        self.user_max = np.max(self.user_data)
        
        
        self.neg_sampled = False
        if num_neg > 0:
            negs_l = []
            for i in range(len(self.user_data)):
                u = self.user_data[i]
                negs_l.append(np.random.choice(negs[u], (num_neg,)))
            self.neg_data = np.stack(negs_l)
            self.neg_sampled = True
            #print("neg ex", self.neg_data[0])

        self.pos_items = {}
        for i in range(len(users)):
            self.pos_items[users[i]] = items[i]

        self.unique_users = np.unique(users)
        self.negs = negs

    def refresh_negs(self):
        if self.num_neg > 0:
            #print("dataset refreshing negative samples")
            negs_l = []
            for i in range(len(self.user_data)):
                u = self.user_data[i]
                negs_l.append(np.random.choice(self.negs[u], (self.num_neg,)))
            self.neg_data = np.stack(negs_l)
            self.neg_sampled = True
            #print("neg ex", self.neg_data[0])
                  

    def __len__(self):
        return len(self.user_data)

    def __getitem__(self, idx):
        if idx == 0:
            self.refresh_negs()
        if self.neg_sampled:
            return self.user_data[idx], self.item_data[idx], self.neg_data[idx]
        else:
            return self.user_data[idx], self.item_data[idx], None


class NCFUserData(data.Dataset):
    def __init__(self, fulldata, user_id):
        self.fulldata = fulldata
        self.user_id = user_id

    def __len__(self):
        return self.fulldata.user_ends[self.user_id] - self.fulldata.user_starts[self.user_id]

    def refresh_user_negs(self):
        if self.fulldata.num_neg > 0:
            #print("dataset refreshing negative samples")
            start_idx = self.fulldata.user_starts[self.user_id]
            end_idx = self.fulldata.user_ends[self.user_id]
            for i in range(start_idx, end_idx):
                u = self.fulldata.user_data[i]
                assert(u == self.user_id)
                self.fulldata.neg_data[i,:] = np.random.choice(self.fulldata.negs[u], (self.fulldata.num_neg,))
            self.fulldata.neg_sampled = True
            #print("neg for user", self.user_id, self.fulldata.neg_data[start_idx][0])

    def __getitem__(self, idx):
        start_idx = self.fulldata.user_starts[self.user_id]
        end_idx = self.fulldata.user_ends[self.user_id]
        
        global_idx = idx + start_idx
        if idx == 0:
            self.refresh_user_negs()
        if self.fulldata.neg_sampled:
            return self.fulldata.user_data[global_idx], self.fulldata.item_data[global_idx], self.fulldata.neg_data[global_idx]
        else:
            return self.fulldata.user_data[global_idx], self.fulldata.item_data[global_idx], None


if __name__ == '__main__':
    dataset = NCFData("./data/debug_m1/train.txt", 2)

    print("------------ full ---------------")
    print(dataset.user_data)
    print(dataset.item_data)
    print(dataset.neg_data)
    print(dataset.unique_users)
    print("------------ xxxx ---------------")
  
    for r in range(2):
        print ("ROUND", r)
        for u in dataset.unique_users:
            userdata = NCFUserData(dataset, u)
            dataloader = data.DataLoader(userdata, batch_size=1)
            print("data for ", userdata.user_id, "len:", userdata.__len__(), len(userdata))
            for i, x in enumerate(dataloader):
                print(x)
    
            print("------------ full ---------------")
            print(dataset.user_data)
            print(dataset.item_data)
            print(dataset.neg_data)
            print(dataset.unique_users)
            print("------------ xxxx ---------------")
