import sklearn.preprocessing
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import pandas as pd
from pyspark.sql.functions import collect_list, struct, col, explode, rand
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression