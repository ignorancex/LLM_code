import numpy as np

def get_cols(array: np.ndarray):
    return (array[:,0], array[:,1])

def read_xy_data(filename, delimiter=','):
    with open(filename) as f:
        x = []
        y = []
        for line in f:
            row = line.strip('\n').split(delimiter)
            if len(row) != 2:
                raise Exception("Expected 2 columns of x,y data")
            x0, y0 = row
            x.append(float(x0))
            y.append(float(y0))
    return x,y