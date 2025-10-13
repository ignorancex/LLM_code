import random
import numpy as np
from torch.utils.data import BatchSampler, Sampler
from utils.datasets import OnlineSimulationDataSet

class UserSampler(Sampler):
    def __init__(self, dataset, batch_size):
        super().__init__(dataset)
        self.dataset = dataset
        group_sizes = dataset.n_groups_by_user_id
        self.group_sizes = group_sizes
        self.groups = list(group_sizes.keys())
        self.batch_size = batch_size
        self.current_groups = [self.groups[i] for i in range(min(batch_size, len(self.groups)))]
        self.current_idx = [0 for _ in range(min(self.batch_size, len(self.groups)))]
        self.group_counter = len(self.current_groups) - 1

    def __iter__(self):
        while len(self.current_groups):
            items = []
            remove_idx = []
            for i in range(len(self.current_groups)):
                checked_group = self.current_groups[i]
                items.append((checked_group, self.group_sizes[checked_group][self.current_idx[i]]))
                self.current_idx[i] += 1
                if len(self.group_sizes[checked_group]) == self.current_idx[i]:
                    self.group_counter += 1
                    if self.group_counter < len(self.groups):
                        self.current_groups[i] = self.groups[self.group_counter]
                        self.current_idx[i] = 0
                    else:
                        remove_idx.append(i)
            if len(remove_idx):
                for idx in sorted(remove_idx, reverse=True):
                    self.current_groups.pop(idx)
                    self.current_idx.pop(idx)
            for i, item in enumerate(items):
                is_last = i + 1 == len(items)
                yield self.dataset.group_to_idx[item], is_last
        else:
            self.__init__(self.dataset, self.batch_size)

    def __len__(self):
        return len(self.dataset)  # todo: this is not the exactly len. I need to fixed it.


class SimulationSampler(Sampler):
    def __init__(self, dataset: OnlineSimulationDataSet, batch_size):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size

    def get_batch(self):
        cur_batch_size = min(self.batch_size, len(self.dataset.active_users))
        return random.sample(self.dataset.active_users, cur_batch_size)

    def __iter__(self):
        batch = self.get_batch()
        while len(batch):
            yield batch
            batch = self.get_batch()
        else:
            self.__init__(self.dataset, self.batch_size)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)


class NewUserBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, shuffle=False, sampling_type="distribution"):
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.active_users = {user: 0 for user, groups in dataset.n_groups_by_user_id.items() if len(groups)}
        self.total_groups_per_users = {user: len(groups) for user, groups in dataset.n_groups_by_user_id.items() if len(groups)}
        self.groups_remain = sum([l for user, l in self.total_groups_per_users.items()])
        self.shuffle = shuffle
        self.sampling_type = sampling_type
        assert sampling_type in ["simple", "distribution"]

    def get_batch(self):
        cur_batch_size = min(self.batch_size, len(self.active_users))
        if self.shuffle:
            if self.sampling_type == "distribution":
                distribution = [(self.total_groups_per_users[user] - group_idx) / self.groups_remain for user, group_idx
                                in self.active_users.items()]
                self.groups_remain -= cur_batch_size
                if self.groups_remain > 0:
                    users_in_batch = np.random.choice(list(self.active_users.keys()), cur_batch_size, replace=False, p=distribution)
                else:
                    users_in_batch = []
            else:
                users_in_batch = random.sample(self.active_users.keys(), cur_batch_size)
        else:
            users_in_batch = list(self.active_users.keys())[:cur_batch_size]
        batch_idx = []
        for user in users_in_batch:
            group_of_user = self.dataset.n_groups_by_user_id[user][self.active_users[user]]
            batch_idx += [(user, group_of_user)]
            self.active_users[user] += 1
            if self.active_users[user] == len(self.dataset.n_groups_by_user_id[user]):
                del self.active_users[user]
                del self.total_groups_per_users[user]
        return batch_idx

    def __iter__(self):
        batch = self.get_batch()
        while len(batch):
            yield batch
            batch = self.get_batch()
        else:
            self.__init__(self.dataset, self.batch_size, self.shuffle)

    def __len__(self):
        return int(len(self.dataset) / self.batch_size)


class UserBatchSampler(BatchSampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        self.batch_size = batch_size
        sampler = UserSampler(dataset, self.batch_size)
        super().__init__(sampler, self.batch_size, drop_last)

    def __iter__(self):
        batch = [0] * self.batch_size
        idx_in_batch = 0
        for idx, is_last in self.sampler:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if is_last:
                yield batch[:idx_in_batch]
                idx_in_batch = 0
                batch = [0] * self.batch_size
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]

