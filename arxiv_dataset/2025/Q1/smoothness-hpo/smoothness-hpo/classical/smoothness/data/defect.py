from raise_utils.data import DataLoader
from raise_utils.hooks import Hook


# For the Wang et al. experiments
file_dic_wang = {"ivy1":     ["ivy-1.4.csv", "ivy-2.0.csv"],
                 "lucene1":  ["lucene-2.0.csv", "lucene-2.2.csv"],
                 "lucene2": ["lucene-2.2.csv", "lucene-2.4.csv"],
                 "poi1":     ["poi-1.5.csv", "poi-2.5.csv"],
                 "poi2": ["poi-2.5.csv", "poi-3.0.csv"],
                 "synapse1": ["synapse-1.0.csv", "synapse-1.1.csv"],
                 "synapse2": ["synapse-1.1.csv", "synapse-1.2.csv"],
                 "camel1": ["camel-1.2.csv", "camel-1.4.csv"],
                 "camel2": ["camel-1.4.csv", "camel-1.6.csv"],
                 "xerces1": ["xerces-1.2.csv", "xerces-1.3.csv"],
                 "jedit1": ["jedit-3.2.csv", "jedit-4.0.csv"],
                 "jedit2": ["jedit-4.0.csv", "jedit-4.1.csv"],
                 "log4j1": ["log4j-1.0.csv", "log4j-1.1.csv"],
                 "xalan1": ["xalan-2.4.csv", "xalan-2.5.csv"]
                 }


def load_defect_prediction_data(dataset: str):
    base_path = './data/'
    def _binarize(x, y): y[y > 1] = 1
    dataset = DataLoader.from_files(
        base_path=base_path, files=file_dic_wang[dataset], hooks=[Hook('binarize', _binarize)])

    return dataset
