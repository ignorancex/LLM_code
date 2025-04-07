from qlpdb.graph.models import Graph
from qlpdb.tdse.models import Tdse

import matplotlib.pyplot as plt

def data():
    data = Graph.objects.filter(tag="NN(2)")
    return data