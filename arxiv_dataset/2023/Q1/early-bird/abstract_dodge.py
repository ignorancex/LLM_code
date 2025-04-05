"""
@author : Shrikanth N C
Abstract DODGE algorithm
DOI: 10.1109/TSE.2019.2945020
"""

from abc import ABC, abstractmethod

from utilities import _randchoice, _randuniform, _randint, unpack
from ML import *
from itertools import product
from transformation import *
from random import seed
import random


class Node(object):

    def __init__(self, classifier, preprocessor):

        self.weight = 0
        self.classifier = classifier
        self.preprocessor = preprocessor
        self.score = None
        self.mutated = False
        self.default = False

    def name(self, item):
        return str(item.__class__.__name__)

    def get_classifier(self):
        return self.classifier

    def get_preprocessor(self):
        return self.preprocessor

    def get_weight(self):
        return self.weight


    def set_score(self, score):
        self.score = score

    def increment_weight(self):
            self.weight += 1

    def decrement_weight(self):
            self.weight -= 1

    def get_score(self):
        return self.score

    def print(self):

        print(self.classifier, self.preprocessor, 'weight = ', self.weight, 'score = ', self.score, 'default = ', self.default, 'mutated = ', self.mutated)


    # def get_error_score(self, goal):
    #     if goal == 'd2h':
    #         return 1
    #
    #     float('goal undefined')



    def __str__(self):
        return self.name(self.classifier) + ' - ' + self.name(self.preprocessor) + ' weight: '+str(self.weight) + " score: "+str(self.score) +\
    str(self.classifier.get_params())+ " - "+str(self.preprocessor.get_params())


class abstract_dodge(ABC):

    def __init__(self, train, test, goal='d2h',  epsilon=0.2, N1=12, N2=30):

        self.goal = goal
        self.N1 = N1
        self.N2 = N2
        self.epsilon = epsilon

        self.tree_of_options = []
        self.current_options = []
        self.default_options = []
        self.rejected_options = []

        self.train = train
        self.test = test
        # self.sample = sample
        self.mutated = 0
        # self.version = version
        self.best_score = None

        super().__init__()

    @abstractmethod
    def build_tree_of_options(self):
        pass

    def get_best_nodes(self):

        best_nodes = []


        for n in self.current_options:


            if n.get_weight() == 1:
                best_nodes.append(n)


        # if self.version == 1:
        #     if len(self.tree_of_options) > 0:
        #
        #         best_weight = self.tree_of_options[0].get_weight()
        #
        #         for node in self.tree_of_options:
        #             if node.get_weight() == best_weight:
        #                 best_nodes.append(node)
        #             else:
        #                 break
        #
        # elif self.version == 2:
        #
        #     if len(self.tree_of_options) > 0:
        #
        #         best_score = self.tree_of_options[0].score
        #
        #         for node in self.tree_of_options:
        #             if node.score <= best_score + 0.05:
        #                 best_nodes.append(node)
        #             else:
        #                 break

        # print("best nodes weights = ", len(best_nodes), [x.weight for x in best_nodes])
        # print("best nodes score = ", len(best_nodes), [x.score for x in best_nodes])
        return best_nodes

    def is_same_type(self, best_node, worst_node):

        return worst_node.classifier.__class__ == best_node.classifier.__class__ \
                and worst_node.preprocessor.__class__ == best_node.preprocessor.__class__

    def get_worst_node(self, best_node):

        worst_node = None

        current_score = best_node.score

        # print("Best node")
        # best_node.print()
        for current_node in self.current_options:

            # print("compared with ")
            #
            # current_node.print()

            if self.is_same_type(best_node, current_node) and    current_node.score > current_score :

                worst_node = current_node
                current_score = current_node.score


        return worst_node

        #
        #
        #
        #
        # if self.version == 1:
        #
        #     worst_weight_so_far = best_node.weight
        #
        #     for current_node in self.tree_of_options:
        #
        #         if current_node.weight < worst_weight_so_far and self.is_same_type(best_node, current_node):
        #             worst_node = current_node
        #             worst_weight_so_far = current_node.weight
        #
        # elif self.version == 2:
        #
        #     worst_score_so_far = best_node.score
        #
        #     for current_node in self.tree_of_options:
        #
        #         if current_node.score > worst_score_so_far + 0.1 and self.is_same_type(best_node, current_node):
        #             worst_node = current_node
        #             worst_score_so_far = current_node.score
        #             break
        #
        #
        # return worst_node


    def mutate_classifier(self, best_classifier, worst_classifier):

        worst_params =  worst_classifier.get_params()
        best_params = best_classifier.get_params()
        mutated_params = {}

        for k, v in best_params.items():

            if isinstance(v, bool):
                mutated_params[k] = _randchoice([True, False])
            elif isinstance(v, float):

                if worst_params[k] is None:
                    worst_params[k] = best_params[k] + 0.0001

                mutated_params[k] = _randuniform(best_params[k], (best_params[k] + worst_params[k]) / 2)

            elif isinstance(v, int):

                if worst_params[k] is None:
                    worst_params[k] = best_params[k] + 1

                mutated_params[k] = _randint(best_params[k], int((best_params[k] + worst_params[k]) / 2 ) )

            else:

                mutated_params[k] = best_params[k]


        # for k,v in best_params.items():
        #     print('best param = ', k,v)
        # for k,v in mutated_params.items():
        #     print('mutated param = ', k,v)

        mutated_classifier = new_classifier(best_classifier)

        mutated_classifier.set_params(**mutated_params)

        return mutated_classifier

    def name(self, item):
        return str(item.__class__.__name__)

    def mutate_preprocessor(self, best_preprocessor, worst_preprocessor):

        # if str(best_preprocessor) == 'no_transform':
        #     return best_preprocessor

        best_params = best_preprocessor.get_params()
        worst_params = worst_preprocessor.get_params()
        mutated_params = {}

        for k, v in best_params.items():

            if isinstance(v, bool):
                mutated_params[k] = _randchoice([True, False])
            elif isinstance(v, float):
                if worst_params[k] is None:
                    worst_params[k] = best_params[k] + 0.0001

                mutated_params[k] = _randuniform(best_params[k], (best_params[k] + worst_params[k]) / 2)

            elif isinstance(v, int):

                if worst_params[k] is None:
                    worst_params[k] = best_params[k] + 1

                mutated_params[k] = _randint(best_params[k], int((best_params[k] + worst_params[k]) / 2 ) )

            else:

                mutated_params[k] = best_params[k]


        mutated_preprocessor = new_preprocessor(best_preprocessor)
        mutated_preprocessor.set_params(**mutated_params)

        return mutated_preprocessor


    def to_string(self, old_node):

        return str(old_node.classifier.get_params()) + '-' + str(old_node.preprocessor) + '-'

    def is_new_node(self, new_node):


        for old_node in self.current_options:

            if self.to_string(old_node) == self.to_string(new_node):
                return False

        return True

    def mutate(self, best_nodes):

        mutated_nodes = []

        for best_node in best_nodes:

            worst_node = self.get_worst_node(best_node)

            # print("Best node")
            # best_node.print()
            # print("Worst node")
            # worst_node.print()

            if worst_node is not None:



                mutated_node = Node(self.mutate_classifier(best_node.classifier, worst_node.classifier),
                                          self.mutate_preprocessor(best_node.preprocessor, worst_node.preprocessor))
                mutated_node.mutated = True

                if self.is_new_node(mutated_node):

                    mutated_nodes.append(mutated_node)
                    self.mutated += 1

                # print("Mutated between  ------------------")
                # best_node.print()
                # worst_node.print()
                # mutated_node.print()
                # print("------------------")

        return mutated_nodes

    @abstractmethod
    def compute_model_performance(self, node, train_changes, tune_changes, goal):
        pass

    def process_node(self, current_node, train_changes, tune_changes):



        current_score = self.compute_model_performance(current_node, train_changes, tune_changes, self.goal)
        current_node.set_score(current_score)

        if current_score is None:
            return

        # print("weights = ", len(self.current_options), [x.weight for x in self.current_options])
        # print("score = ", len(self.current_options), [x.score for x in self.current_options])


        if len(self.current_options) > 1:
            for past_node in self.current_options:

                # print("\t comparing ",current_score, 'with', past_node.score, past_node.score - current_score, '?', self.epsilon)
                if abs(past_node.score - current_score) < self.epsilon or (current_score > past_node.score):
                    current_node.decrement_weight()
                    self.current_options.append(current_node)
                    # print("\n returning because ", past_node.score, current_score, '<>', self.epsilon,  past_node.score - current_score, abs(past_node.score - current_score) < self.epsilon,
                    #       (current_score > past_node.score))
                    return


        # print("Incrementing weight for ")
        # current_node.print()
        current_node.increment_weight()
        self.current_options.append(current_node)

        # self.current_options.append(current_node)




        # node.set_score(current_score)
        # previous_score = node.score
        #
        # if previous_score is not None and current_score != node.get_error_score(self.goal):
        #
        #     delta = abs(previous_score - current_score)
        #
        #
        #     if self.version == 1:
        #         if delta > self.epsilon:
        #             node.increment_weight()
        #         else:
        #             node.decrement_weight()
        #
        #     elif self.version == 2:
        #         print()
        #
        # node.set_score(current_score)

    def get_best_settings(self):

        self.current_options.sort(key=lambda x: x.score, reverse=False)
        # print("warn return median")
        # return self.current_options[int(len(self.current_options)/2)]
        self.current_options[0].print()
        return self.current_options[0]


    def should_add(self, mutated_node):

        return mutated_node.score < self.get_best_nodes()[0].score


    def evaluate_nodes(self):

        n1 = self.N1



        start = 0
        chunk = int(len(self.tree_of_options)/self.N1) - 1
        end = start + chunk



        while n1 > 0:

            print("Processing ", n1, "default options = ", len(self.default_options), "current options = ", len(self.current_options))
            if n1 == self.N1:

                for node in self.default_options:
                    # print("processing default node")
                    # node.print()
                    self.process_node(node, self.train.copy(deep=True), self.test.copy(deep=True))
            else:
                # print("Sampling between ", start, end)
                for node in random.sample(self.tree_of_options[start : end], min(chunk, 100)):
                    self.process_node(node, self.train.copy(deep=True), self.test.copy(deep=True))

                start = end
                end = start + chunk

            self.current_options.sort(key=lambda x: x.score)
            # print(">> current_options \t scores = ", len(self.current_options), [ str(x.score)+"("+str(x.weight) for x in self.current_options][0])




            n1 -= 1


        print("total = ", len(self.current_options))

        print('Mutating...')
        n2 = self.N2


        print("\n\t\n\n N2\n\n")

        self.current_options.sort(key=lambda x: x.weight, reverse=False)

        print("weights = ", len(self.current_options), [x.weight for x in self.current_options])
        print("score = ", len(self.current_options), [x.score for x in self.current_options])




        while n2 > 0:

            self.current_options.sort(key=lambda x: x.weight, reverse=False)



            mutated_nodes = self.mutate(self.get_best_nodes())
            # print(n2, ' about to mutate ', len(mutated_nodes))

            # for m in mutated_nodes:
            #     m.print()


            for mutated_node in mutated_nodes:
                self.process_node(mutated_node, self.train.copy(deep=True), self.test.copy(deep=True))


            print(n2, "mutated = ",self.mutated)


            n2 -= 1


    def print_all_nodes(self):

        for n in self.current_options:
            print(str(n))

    def run(self):

        self.build_tree_of_options()
        self.evaluate_nodes()

        best_node = self.get_best_settings()

        info = None
        if best_node.mutated:
            info = 'MUTATED'
        elif best_node.default:
            info = 'DEFAULT'
        else:
            info = 'N1'

        return best_node.classifier,  best_node.preprocessor , best_node.score, info

        # return LogisticRegression(), StandardScaler()


    def is_same_node(self, mutated_node, previous_mutated_node):

        if previous_mutated_node is None and mutated_node is not None:
            return False
        elif previous_mutated_node is not None and mutated_node is None:
            return False
        else:
            if str(previous_mutated_node.classifier.get_params()) != str(mutated_node.classifier.get_params()):
                return False
            elif str(previous_mutated_node.preprocessor.get_params()) != str(mutated_node.preprocessor.get_params()):
                return False
            else:
                return True

    # def remove_similar_nodes(self):
    #
    #     self.tree_of_options.sort(key=lambda x: x.score, reverse=False)
    #
    #     new_nodes = []
    #     new_nodes.append(self.tree_of_options[0])
    #
    #     next_score = new_nodes[0].score + 0.02
    #
    #     for i in range(1, len(self.tree_of_options)):
    #
    #         top = self.tree_of_options[i]
    #
    #         if top.score >= next_score:
    #             new_nodes.append(top)
    #             next_score = top.score + 0.05
    #
    #     if len(new_nodes) == 1:
    #         new_nodes.append(self.tree_of_options[len(self.tree_of_options) - 1])
    #
    #     self.tree_of_options = new_nodes

    # def get_random_node(self, best_node):
    #
    #     sample = []
    #     for top in self.tree_of_options:
    #
    #         if top != best_node and self.is_same_type(top, best_node):
    #             sample.append(top)
    #
    #     if len(sample) > 1:
    #         return sample[_randint(0,len(sample)-1)]
    #     elif len(sample) == 1:
    #         return sample[0]
    #     else:
    #         return None


