from root.correlations.correlations import Correlations
from root.data.data import Data
from root.model.eval import Eval
from root.model.hopfield import HopfieldNetwork
from root.plotter.plotter import LinePlotInput, Plotter

"""
two things for the datasets
1. if we fix the polydegree, we must almost create the datasets on the fly because subsampling the datasets 
will change the correlation as we iteratively add more memories to get k_max in the current implementation
- however, the correlation is only calculated with the subset of the sampling, so that's about 80/20 where I got most feedback from

2. if we fix the number of memories to be restored, we need to vary the degree of the interaction function

3. I am curious to see if I can reproduce the results from last time, with the same dataset, but what happens with mean HD
4.: Mean HD is also a shit predictor. So is min.

[TODO]: write class so that it can account for "fixing" either parameter and then dynamically adjust the datasets

"""

# it seems n = 1 or n = 2 and just running the dataset at one scale yields the most promising results tbh
# or nah?
# also we can just create thos arbitrary same-sized datasets via the func


def create_data():
    data = Data.load("lower_dims_correlated")
    y1 = Eval(data, HopfieldNetwork(neurons=64, polydegree=2)).get_k_maxes()
    Data.save("k_max_lower_dims_n2", y1)
    y2 = Eval(data, HopfieldNetwork(neurons=64, polydegree=6)).get_k_maxes()
    Data.save("k_max_lower_dims_n6", y2)
    y3 = Eval(
        data, HopfieldNetwork(neurons=64, polydegree=None)
    ).get_k_maxes()  # self-attention
    Data.save("k_max_lower_dims_exp", y3)
    y3 = Eval(
        data,
        HopfieldNetwork(
            neurons=64,
        ),
    ).get_k_maxes()


def create_uncorrelated_data():
    data = Data.load("lower_dims_uncorrelated")
    print(data.shape)
    # y1 = Eval(data, HopfieldNetwork(neurons=64, polydegree=6)).get_k_maxes()
    # Data.save("k_max_lower_dims_n6_uncorrelated", y1)


def create_simulated_real_world_data():
    data = Data.load("real_world_sim_data")
    print(data.shape)
    y1 = Eval(data, HopfieldNetwork(neurons=64, polydegree=2)).get_k_maxes()
    Data.save("k_max_real_world_sim_data_n2", y1)
    y2 = Eval(data, HopfieldNetwork(neurons=64, polydegree=4)).get_k_maxes()
    Data.save("k_max_real_world_sim_data_n4", y2)
    y3 = Eval(data, HopfieldNetwork(neurons=64, polydegree=7)).get_k_maxes()
    Data.save("k_max_real_world_sim_data_n7", y3)
    y4 = Eval(
        data, HopfieldNetwork(neurons=64, polydegree=None)
    ).get_k_maxes()  # self-attention
    Data.save("k_max_real_world_sim_data_n_exp", y4)


def create_5000_data_correlated_but_fake():
    data = Data.load("5000_corr_datasets")
    # y1 = Eval(data, HopfieldNetwork(neurons=64, polydegree=2)).get_k_maxes_parallel()
    # Data.save("k_max_5000_corr_datasets_n2", y1)
    y2 = Eval(data, HopfieldNetwork(neurons=64, polydegree=7)).get_k_maxes()
    Data.save("k_max_5000_corr_datasets_n7", y2)
    y3 = Eval(data, HopfieldNetwork(neurons=64, polydegree=25)).get_k_maxes_parallel()
    Data.save("k_max_5000_corr_datasets_n25", y3)
    y4 = Eval(data, HopfieldNetwork(neurons=64, polydegree=None)).get_k_maxes_parallel()
    Data.save("k_max_5000_corr_datasets_n_exp", y4)


def create_155_and_above():
    data = Data.load("mnist_subsets_all_165_190_3")
    y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=4)).get_k_maxes_parallel()
    Data.save("k_max_mnist_subsets_all_165_190_3_n4", y1)
    y2 = Eval(data, HopfieldNetwork(neurons=784, polydegree=5)).get_k_maxes_parallel()
    Data.save("k_max_mnist_subsets_all_165_190_3_n5", y2)
    y3 = Eval(data, HopfieldNetwork(neurons=784, polydegree=6)).get_k_maxes_parallel()
    Data.save("k_max_mnist_subsets_all_165_190_3_n6", y3)
    y4 = Eval(data, HopfieldNetwork(neurons=784, polydegree=7)).get_k_maxes_parallel()
    Data.save("k_max_mnist_subsets_all_165_190_3_n7", y4)


def create_k_max_decor_n7():
    data = Data.load("decreasingly_correlated_sets")
    y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=7)).get_k_maxes_parallel()
    Data.save("k_max_decor_n7", y1)


def create_more_data():
    # data = Data.load("decreasingly_correlated_sets")
    # y0 = Eval(data, HopfieldNetwork(neurons=784, polydegree=8)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n8", y0)
    # y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=9)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n9", y1)
    # y2 = Eval(data, HopfieldNetwork(neurons=784, polydegree=11)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n11", y2)
    # y3 = Eval(data, HopfieldNetwork(neurons=784, polydegree=13)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n13", y3)
    # y0 = Eval(data, HopfieldNetwork(neurons=784, polydegree=14)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n14", y0)
    # y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=16)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n16", y1)
    # y2 = Eval(data, HopfieldNetwork(neurons=784, polydegree=18)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n18", y2)
    # y3 = Eval(data, HopfieldNetwork(neurons=784, polydegree=20)).get_k_maxes_parallel()
    # Data.save("k_max_decor_n20", y3)
    # y4 = Eval(
    #     data, HopfieldNetwork(neurons=784, polydegree=None)
    # # ).get_k_maxes_parallel()
    # # Data.save("k_max_decor_exp", y4)

    # # # # do the same for mnist
    # data = Data.load("all_subsets_MNIST_only_and_165_190_3")
    # y0 = Eval(data, HopfieldNetwork(neurons=784, polydegree=2)).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_n2", y0)
    # y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=3)).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_n3", y1)

    # # y0 = Eval(data, HopfieldNetwork(neurons=784, polydegree=8)).get_k_maxes_parallel()
    # # Data.save("k_max_subsets_combined_n8", y0)
    # # y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=9)).get_k_maxes_parallel()
    # # Data.save("k_max_mnist_n9", y1)
    # # y2 = Eval(data, HopfieldNetwork(neurons=784, polydegree=11)).get_k_maxes_parallel()
    # # Data.save("k_max_mnist_n11", y2)
    # # y3 = Eval(data, HopfieldNetwork(neurons=784, polydegree=13)).get_k_maxes_parallel()
    # # Data.save("k_max_mnist_n13", y3)
    # y0 = Eval(data, HopfieldNetwork(neurons=784, polydegree=14)).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_n14", y0)
    # y1 = Eval(data, HopfieldNetwork(neurons=784, polydegree=16)).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_n16", y1)
    # y2 = Eval(data, HopfieldNetwork(neurons=784, polydegree=18)).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_n18", y2)
    # y3 = Eval(data, HopfieldNetwork(neurons=784, polydegree=20)).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_n20", y3)
    # y4 = Eval(
    #     data, HopfieldNetwork(neurons=784, polydegree=None)
    # ).get_k_maxes_parallel()
    # Data.save("k_max_subsets_combined_exp", y4)

    # # Artificial data
    # data = Data.load("decreasingly_correlated_sets")
    # for n in range(22, 41, 2):
    #     y = Eval(data, HopfieldNetwork(neurons=784, polydegree=n)).get_k_maxes_parallel(
    #         stop_count=2, threshold=50
    #     )
    #     Data.save(f"k_max_decor_n{n}", y)

    # MNIST data
    data = Data.load("all_subsets_MNIST_only_and_165_190_3")
    print(data.shape)
    # for n in range(22, 41, 2):
    #     y = Eval(data, HopfieldNetwork(neurons=784, polydegree=n)).get_k_maxes_parallel(
    #         stop_count=2, threshold=50
    #     )
    #     Data.save(f"k_max_subsets_combined_n{n}", y)


def main():
    # create_uncorrelated_data()
    # create_simulated_real_world_data()
    # create_5000_data_correlated_but_fake()
    # create_155_and_above()
    # create_k_max_decor_n7()
    create_more_data()


if __name__ == "__main__":
    main()
