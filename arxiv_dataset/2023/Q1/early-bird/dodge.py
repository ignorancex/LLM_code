from abstract_dodge import *


"""
@author : Shrikanth N C
Implementation of the DODGE algorithm
DOI: 10.1109/TSE.2019.2945020
"""

import rq_run

class DODGE(abstract_dodge):

    def build_default_options(self):

        times = 1

        preprocess = [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * times, kernel_centerer,
                      [quantile_transform] * times, normalizer, [binarize] * times]

        MLs = [[NB] * times, [KNN] * times, [RF] * times, [DT] * times, [LR] * times]

        # pre-processing error  Kernel matrix must be a square matrix. Input is a 50x14 matrix. KernelCenterer() KernelCenterer

        preprocess_list = unpack(preprocess)
        MLs_list = unpack(MLs)
        combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

        default_options = []

        for c in combine:
            node = Node(c[1](True)[0], c[0]()[0])
            node.default = True
            default_options.append(node)


        # to ensure there is a worst node for every type
        self.default_options = default_options

    def build_tree_of_options(self):
        """
        Overridden to customize
        :return:
        """

        mn = 1
        tree_of_options = []

        np.random.seed(mn)
        seed(mn)


        preprocess = [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * 20, kernel_centerer,
                      [quantile_transform] * 200
            , normalizer, [binarize] * 100]  # ,[polynomial]*5
        MLs = [NB, [KNN] * 20, [RF] * 50, [DT] * 30, [LR] * 50]  # [SVM]*100,

        # preprocess = [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * 1, kernel_centerer,
        #               [quantile_transform] * 1
        #     , normalizer, [binarize] * 1]  # ,[polynomial]*5
        # MLs = [NB, [KNN] * 1, [RF] * 1, [DT] * 1, [LR] * 1]  # [SVM]*100,


        # pre-processing error  Kernel matrix must be a square matrix. Input is a 50x14 matrix. KernelCenterer() KernelCenterer

        preprocess_list = unpack(preprocess)
        MLs_list = unpack(MLs)
        combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]

        # print(combine)

        for c in combine:
            node = Node(c[1]()[0], c[0]()[0])
            tree_of_options.append(node)

        # sample 100 tree of options at random
        self.tree_of_options = tree_of_options

        print("Tree of options = ", len(self.tree_of_options))

        self.build_default_options()






    def local_eval(self, node, train_changes, test_changes, goal):


        temp_train_changes = train_changes.copy(deep=True)
        temp_test_changes = test_changes.copy(deep=True)

        try:
            temp_train_changes = transform(temp_train_changes, node.preprocessor)
            temp_test_changes = transform(temp_test_changes, node.preprocessor)
        except Exception as e1:
            node.weight = None
            # print("\t [ERROR] pre-processing error ", e1, node.preprocessor)
            return None

        try:

            trainY = temp_train_changes.Buggy
            trainX = temp_train_changes.drop(labels=['Buggy'], axis=1)

            node.classifier.fit(trainX, trainY)
            F = rq_run.computeMeasures(temp_test_changes, node.classifier, [], [0 for x in range(0, len(temp_test_changes))])
            current_score = float(F[goal][0])


        except Exception as e2:
            node.weight = None
            print('[DODGE_ERROR] local eval error ', e2,  'classifier = ', node.classifier, 'pre-processor', node.preprocessor)
            return None

        return current_score

    def compute_model_performance(self, node, train_changes, test_changes, goal):
        """
        Overridden to customize
        :param node:
        :param train_changes:
        :param tune_changes:
        :param goal:
        :return:
        """

        # score = 0
        # times = 0
        #
        # while len(tune_changes_all) > 0:
        #
        #     score +=
        #     times += 1
        #     tune_changes_all = tune_changes_all.head(len(tune_changes_all) - 50)



        return self.local_eval(node, train_changes, test_changes, goal)



if __name__ == '__main__':

    project_changes = release_manager.getProject('scikit-learn').getAllChanges()

    train, test = project_changes.head(int(len(project_changes)/2)), project_changes.tail(int(len(project_changes)/2))

    _dodge = DODGE(train, test, 'd2h')

    print("Best Settings : ", _dodge.run())

    # preprocess = [standard_scaler, minmax_scaler, maxabs_scaler, [robust_scaler] * 20, kernel_centerer,
    #               [quantile_transform] * 200, normalizer, [binarize] * 100]
    #
    # MLs = [NB, [KNN] * 20, [RF] * 50, [DT] * 30, [LR] * 50]
    #
    # preprocess_list = unpack(preprocess)
    # MLs_list = unpack(MLs)
    # combine = [[r[0], r[1]] for r in product(preprocess_list, MLs_list)]
    #
    # print(len(combine))


