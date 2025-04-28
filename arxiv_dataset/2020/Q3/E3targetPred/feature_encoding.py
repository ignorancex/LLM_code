## Feature Extracting Util
import re
import numpy as np
import pandas as pd

## Extracting Features
## Define CKSAAP feature-extraction function
def minSequenceLength(fastas):
    minLen = 10000
    for i in fastas:
        if minLen > len(i[1]):
            minLen = len(i[1])
    return minLen

def CKSAAP(fastas, gap=5, **kw):
    if gap < 0:
        print('Error: the gap should be equal or greater than zero' + '\n\n')
        return 0

    if minSequenceLength(fastas) < gap+2:
        print('Error: all the sequence length should be larger than the (gap value) + 2 = ' + str(gap+2) + '\n\n')
        return 0

    AA = 'ACDEFGHIKLMNPQRSTVWY'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)
    header = ['#']
    for g in range(gap+1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)
    for i in fastas:
        name, sequence = i[0], i[1]
        code = [name]
        for g in range(gap+1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

def CKSAAP_Features(seq, gap_size):
    cksaapfea = []
    for i in seq:
        temp= CKSAAP([i], gap=gap_size)
        cksaapfea.append(temp)
    dt = []
    for i in range(len(cksaapfea)):
        temp = cksaapfea[i][1][1:]
        dt.append(temp)
    return np.array(dt)

def Extract_Features(dataset, gap_size):
    """Extract features from dataset """
    Sequence_Pair=[]
    E3_Seq=[]
    Substrate_Seq=[]

    for index, row in dataset.iterrows():
        array = [row['Label'], row['E3_Seq'], row['Substrate_Seq']]        
        label, E3seq = array[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1]).upper())  
        label, Subseq = array[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[2:]).upper())
        label, SeqPair = array[0], re.sub('[^ARNDCQEGHILKMFPSTWYV-]', '-', ''.join(array[1:]).upper())
        E3_Seq.append([label, E3seq])
        Substrate_Seq.append([label, Subseq])
        Sequence_Pair.append([label, SeqPair])

    E3_features = CKSAAP_Features(E3_Seq,gap_size=gap_size)
    Sub_features = CKSAAP_Features(Substrate_Seq,gap_size=gap_size)
    pair_features = CKSAAP_Features(Sequence_Pair,gap_size=gap_size)

    return E3_features, Sub_features, pair_features