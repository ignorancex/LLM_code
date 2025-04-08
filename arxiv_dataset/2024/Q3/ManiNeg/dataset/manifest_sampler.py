from dataset import create_dataset, create_loader
import os
import numpy as np
from torch.utils.data import DataLoader, Sampler
import random
from scipy.stats import truncnorm


def attrhash(attr):
    base = 1
    ret = 0
    for i in attr:
        ret += base * i
        base <<= 1
    return ret


def count_bits(n):
    count = 0
    while n:
        n = n & (n - 1)
        count += 1
    return count


def hamming_distance(n1, n2):
    x = n1 ^ n2
    return count_bits(x)


class LinearMuSchedular:
    def __init__(self, max_mu, min_mu, min_epoch):
        # mu = max(a * epoch + b, min_mu)
        max_mu, min_mu, min_epoch = float(max_mu), float(min_mu), float(min_epoch)
        self.b = max_mu
        self.a = (min_mu - max_mu) / min_epoch
        self.min_mu = min_mu

    def get_mu(self, epoch):
        return max(self.min_mu, self.a * epoch + self.b)


class TruncatedGaussianSampler:
    def __init__(self, mu, sigma, left, right, sample_size):
        self.mu = mu
        self.sigma = sigma
        self.left = left - 0.5
        self.right = right + 0.499
        self.sample_size = sample_size

    def set_param(self, mu):
        self.mu = mu
        # self.sigma = sigma

    def get_param(self):
        return self.mu, self.sigma

    def sample(self):
        a, b = (self.left - self.mu) / self.sigma, (self.right - self.mu) / self.sigma
        data = np.floor(truncnorm.rvs(a, b, loc=self.mu, scale=self.sigma, size=self.sample_size) + 0.5)
        return data


class NegManiBatchSampler(Sampler):
    def __init__(self, train_dataset, gaussian_sampler, deduplicate=False, max_length=None):
        # num of negative samples are set in the TruncatedGaussianSampler
        # limited_length: use a subset of dataset
        super().__init__()
        self.length = len(train_dataset)
        self.indice_list = [i for i in range(self.length)]
        self.gaussian_sampler = gaussian_sampler
        self.max_length = self.length if max_length is None else max_length
        self.dedup = deduplicate
        attr2datasetid = {}
        attr_set = set()
        pointer = 0
        for i in range(len(train_dataset)):
            assert 'attr' in train_dataset[
                i], "To use manifestation negative sampler, you have to output attr modality."
            attr_id = attrhash(train_dataset[i]['attr'].numpy().astype('uint8'))
            attr_set.add(attr_id)
            if attr_id not in attr2datasetid:
                attr2datasetid[attr_id] = []

                pointer += 1
            attr2datasetid[attr_id].append(i)

        self.attr2datasetid = attr2datasetid

        self.datasetid2attr = [0] * self.length
        for i in self.attr2datasetid:
            for j in self.attr2datasetid[i]:
                self.datasetid2attr[j] = i

        # distance_matrix = np.zeros((pointer, pointer))  # pointer: dataset length
        # for ni, i in enumerate(attr2index):
        #     for nj, j in enumerate(attr2index):
        #         distance_matrix[ni, nj] = hamming_distance(i, j)

        attr2samplepool = {}
        for i in attr_set:
            dist_dict = {}
            for j in attr_set:
                dist = int(hamming_distance(i, j))
                if dist not in dist_dict:
                    dist_dict[dist] = []
                dist_dict[dist].append(j)
                attr2samplepool[i] = dist_dict
        self.attr2samplepool = attr2samplepool

    def __iter__(self):
        random.shuffle(self.indice_list)
        self.cursor = 0
        return self

    def __next__(self):
        if self.cursor >= self.max_length:
            raise StopIteration
        pos = self.indice_list[self.cursor]
        pos_attr = self.datasetid2attr[pos]
        negative_pool = self.attr2samplepool[pos_attr]
        neg_hamming_distance = self.gaussian_sampler.sample().astype('uint8')
        negative_pool_keys = np.array(list(negative_pool.keys()))
        negative_pool_keys = negative_pool_keys[negative_pool_keys != 0]
        # filter: if sample are not in sample pool, it needed to be re-calculated to in the sample pool
        neg_hamming_distance = neg_hamming_distance[None, :]  # [1,63]
        negative_pool_keys = negative_pool_keys[:, None]  # [18,1]

        d = np.abs(neg_hamming_distance - negative_pool_keys)
        arg = np.argmin(d, 0)
        # arg = negative_pool_keys[,0]

        neg_hamming_distance_filtered = negative_pool_keys[arg, 0]

        ret = [pos]
        # sample
        ## 重要信息：sample分为两种可能的形式，
        ## 一是同距离的manifestation等可能，由于每个manifestation对应的instance数不一样
        ## 所以对instance来说可能性是不等同的
        ## 二是instance等可能，这时manifestation不是等可能，同mani的样本越多越容易被sample到
        ## 这里的代码暂时使用第一种，也就是先采样manifestation，再在manifestation中采样具体的样本，这种也许更加公平
        for d in neg_hamming_distance_filtered:
            mani_list_per_d = negative_pool[d]
            neg_mani = random.sample(mani_list_per_d, 1)[0]
            neg_dataset_id = random.sample(self.attr2datasetid[neg_mani], 1)[0]
            ret.append(neg_dataset_id)
        # deduplicate
        ## 重要信息： 关于去重：去重是为了防止同一个batch内出现同样的sample，如果需要计算所有的距离，那么则需要做这样的去重
        ## 方法一：直接去掉重复样本
        ## 方法二，是对重复的样本进行重采样（不放回采样）。这会导致分布发生一些变化
        ## 目前先采用方法一
        if self.dedup:
            ret = list(set(ret))
        # Debug print
        # print(negative_pool_keys[:, 0])
        # print(neg_hamming_distance_filtered)
        # print(neg_hamming_distance[0])
        self.cursor += 1
        return ret

    def batch_mutual_distance(self, reset=False, print_batch_info=True):
        """
        用于debug的代码，分析一个batch内所有的相互距离

        Args:
            reset: reset Iter
            print_batch_info: print batch size and unique manifestation nums for debug. Due to the deduplication, the
                batch size is variable.
        """
        if reset:
            self.__iter__()
            return

        ret = next(self)
        if print_batch_info:
            attr_set = set()
            for i in ret:
                attr_set.add(self.datasetid2attr[i])
            print(f"batch size: {len(ret)}, uni mani num: {len(attr_set)}")

        debug_distance_set = []
        for i in range(len(ret)):
            for j in range(i + 1, len(ret)):
                mani_i = self.datasetid2attr[ret[i]]
                mani_j = self.datasetid2attr[ret[j]]

                debug_distance_set.append(hamming_distance(mani_i, mani_j))
        return debug_distance_set

    def __len__(self):
        return self.max_length


if __name__ == "__main__":
    ## 这部分代码一方面作为sampler的单元测试，另一方面也可以画batch内所有距离的分布
    os.environ["https_proxy"] = "http://172.17.146.34:8891"
    os.environ["http_proxy"] = "http://172.17.146.34:8891"

    config = {'batch_size': 24, 'image_res': 256, 'crop_min_scale': 0.5, "binary_label": True,
              "pre_downsample": True, 'modal': 'a', 't_backbone': 'bert-base-chinese', 'attr_noise': None,
              "partial_data": None, 'padding_square': True}
    datasets = create_dataset('hmbm', config)
    train, val, test = datasets[0], datasets[1], datasets[2]
    train_loader, val_loader, test_loader = create_loader(
        [train, val, test],
        samplers=[None, None, None],
        batch_size=[config['batch_size'], config['batch_size'], config['batch_size']],
        # batch_size=[1,1,1],
        num_workers=[1, 1, 1],
        is_trains=[True, False, False],
        collate_fns=[None, None, None],
        drop_last=[False, False, False],
        pin_memory=False)

    # attr2datasetid

    # print(attr2datasetid)
    # controller = Controller()
    mu = 17
    sigma = 3
    gaussian_sampler = TruncatedGaussianSampler(mu, sigma, 1, 18, 63)

    sampler = NegManiBatchSampler(train, gaussian_sampler, deduplicate=True)
    from matplotlib import pyplot as plt
    import matplotlib

    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.figure(figsize=(10, 2.2))
    plt.gcf().subplots_adjust(bottom=0.15)
    for n, i in enumerate([0, 4, 11]):
        plt.subplot(1, 3, n + 1)
        gaussian_sampler.set_param(i)
        sampler.batch_mutual_distance(reset=True)
        dists = []
        try:
            while 1:
                dist = sampler.batch_mutual_distance()
                dists += dist
        except StopIteration:
            print("Stop")

        # plt.title(f"mu={i}, sigma={sigma}")
        x, y = np.unique(dists, return_counts=True)
        if 0 in x:
            y[x == 0] = 0  # multi batch may same manifestation, delete them here
        y = y.astype('float32') / np.sum(y)
        # hist, bin_edges = np.histogram(data, bins=[i for i in range(round(left),round(right)+1)], density=True)
        # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        # 绘制直方图的折线图
        plt.plot(x, y, linestyle='-', color='g', marker='o', label='Histogram')
        # print(x,y)
        plt.ylabel('Freq')
        plt.xlabel(f'Hamming Distance, $\mu$={i}')
        # plt.xticks(range(17))
        if n == 2:
            plt.legend()
    plt.tight_layout()
    plt.savefig("../fig/ovo.pdf")
    # plt.show()
    # loader = DataLoader(
    #     train,
    #     num_workers=1,
    #     pin_memory=False,
    #     batch_sampler=sampler,
    #     persistent_workers=False,
    # )
    # mu_schedular = LinearMuSchedular(18, 4, 150)
    # x = [i for i in range(300)]
    # y = [mu_schedular.get_mu(i) for i in range(300)]
    # from matplotlib import pyplot as plt
    #
    # plt.plot(x, y)
    # plt.show()
    # for e in range(18):
    #     gaussian_sampler.set_param(1+e)
    #     for i in loader:
    #         print(i.keys())

    # it = iter(loader)
    #
    # c = [17179878433, 17179872292, 17179872321, 17179874337, 17179872290, 1057, 134220833, 17179872296, 536874017, 2081,
    #      17179872385, 67111969, 25836915745, 17179872273, 2147486753, 16780321]
    # controller.set_state(c)
    # print(next(it))
    # c = [17179872289, 545, 289, 1057, 134220833, 4129, 536874017, 67111969, 134220321, 2147486753, 16780321]
    # controller.set_state(c)
    # print(next(it))
