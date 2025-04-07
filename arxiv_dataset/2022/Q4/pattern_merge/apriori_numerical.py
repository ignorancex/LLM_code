import sys
from itertools import chain, combinations
from collections import defaultdict
from optparse import OptionParser
import itertools
import numpy as np
import pandas as pd
from scipy.special import betainc


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, indexList):
    _itemSet = set()
    localSet = defaultdict(int)

    for item in itemSet:
        for i in range(len(transactionList)):
            transaction = transactionList[i]
            if item.issubset(transaction):
                if item not in list(freqSet.keys()):
                    freqSet[item] = []
                freqSet[item].append(indexList[i])
                localSet[item] += 1

    for item, count in localSet.items():
        support = float(count) / len(transactionList)

        if support >= minSupport:
            _itemSet.add(item)

    return _itemSet


def joinSet(itemSet, length):
    return set(
        [i.union(j) for i in itemSet for j in itemSet if len(i.union(j)) == length]
    )

# If we need to remove missing it should be here
def getItemSetTransactionList(data_iterator):
    transactionList = list()
    itemSet = set()
    indexList = list()
    for index, row in data_iterator.iterrows():
    #for record in data_iterator:
        # Remove missings
        index_to_drop = []
        for s_index, s_row in row.items():
            if "?" in s_row:
                index_to_drop.append(s_index)
        row = row.drop(index=index_to_drop)
        #if row.size == 0:
        #    continue
        # Adiciona cada transação a uma lista
        transaction = frozenset(row)
        transactionList.append(transaction)
        indexList.append(index)
        # Adiciona todos os items unicos ao conjunto de items
        for item in transaction:
            itemSet.add(frozenset([item]))  # Generate 1-itemSets
    return itemSet, transactionList, indexList


def runApriori(data_iter, class_vector, minSupport, minLift):
    data_iter = data_iter.applymap(lambda x: str(x))
    for column in data_iter.columns:
        i = 0
        for element in data_iter[column]:
            data_iter.at[i, column] = column + "_" + str(element)
            i += 1
    #print(data_iter)
    # Rename to class and create usefull variables
    #class_vector = class_vector.rename("class")
    #class_vector_mean = class_vector.map(float).mean()
    #class_vector_std = class_vector.map(float).std()

    #itemset  = a uma lista de items unicos
    #transactionList = ao dataset (conjunto de transações)
    itemSet, transactionList, indexList = getItemSetTransactionList(data_iter)
    freqSet = defaultdict(int)
    largeSet = dict()
    oneCSet = returnItemsWithMinSupport(itemSet, transactionList, minSupport, freqSet, indexList)

    currentLSet = oneCSet
    k = 2
    while currentLSet != set([]):
        largeSet[k - 1] = currentLSet
        currentLSet = joinSet(currentLSet, k)
        currentCSet = returnItemsWithMinSupport(
            currentLSet, transactionList, minSupport, freqSet, indexList
        )
        currentLSet = currentCSet
        k = k + 1

    toRetItems = []
    for key, value in largeSet.items():
        temp = []
        for item in value:
            columns_of_p = []
            for val in item:
                columns_of_p.append(val.split("_")[0])
            temp.append({"columns": frozenset(columns_of_p), "indexes": freqSet[item]})
        toRetItems.extend(temp)
        #toRetItems.extend([(tuple(item), getSupport(item)) for item in value])

    return toRetItems







