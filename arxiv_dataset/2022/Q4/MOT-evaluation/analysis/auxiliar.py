
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial import distance


def correlation_matrix(tb, metrics='all', dist_func=distance.correlation):
    '''
    Return a correlation matrix between combinations of detector and
    tracker. The comparable part is a vector of matrixes.

    Inputs:
        - tb : pandas DataFrame.
        - metrics : group name of metrics ('all', 'tracking', 'detection')
        - dist_func : distance or correlation function.

    Outputs:
        - correlation matrix.
    '''
    
    comb = combinations(tb)
    result_m = np.zeros((len(comb), len(comb)))
    
    if metrics == 'all':         func_metrics = all_metrics
    elif metrics == 'tracking':  func_metrics = tracking_metrics
    elif metrics == 'detection': func_metrics = detection_metrics


    for i, (tr1, dt1) in enumerate(comb):

        m1 = search(tb, tr1, dt1)
        m1 = func_metrics(m1)

        for j, (tr2, dt2) in enumerate(comb):

            m2 = search(tb, tr2, dt2)
            m2 = func_metrics(m2)

            aux_m = []

            for v1, v2 in zip(m1.values, m2.values):

                aux_m.append( dist_func(v1, v2) )


            result_m[i, j] = 1 - (sum(aux_m) / len(aux_m))
            
    return result_m


def correlation_metrics(tb, metrics):
    '''
    Return a correlation matrix between detectors. Function compares
    metrics with the Pearson product-moment correlation coefficient
    for each combination of metrics.

    Inputs:
        - tb : pandas DataFrame.
        - metrics : list of metrics.

    Outputs:
        - correlation matrix.
    '''

    result_m = np.zeros((len(metrics), len(metrics)))

    for i, m1 in enumerate(metrics):

        v1 = tb[[m1]].values.flatten()

        for j, m2 in enumerate(metrics):

            v2 = tb[[m2]].values.flatten()

            result_m[i, j] = np.corrcoef(v1, v2)[1, 0]
            
    return result_m


def combinations(tb):
    '''
    Return the possible combinations between detectors and trackers.
    (existing combinations).

    Inputs:
        - tb : pandas DataFrame.

    Outputs:
        - list of possible combinations
    '''

    trck = pd.unique(tb['Tracker'])
    detc = pd.unique(tb['Detector'])


    combin = []

    for tr in trck:

        for dc in detc:

            combin.append((tr, dc))


    return combin


def select_dataset(tb, dataset_name):
    '''
    Return a subset Data Frame with the evaluation outputs
    from the original Data Frame.

    Inputs:
        - tb : pandas DataFrame.
        - dataset_name : Name of the dataset to select.

    Outputs:
        - Data Frame with the required information.
    '''

    return tb[(tb['Dataset name'] == dataset_name)]


def search(tb, tracker, detector=None):
    '''
    Seach tuples with an specific tracker and / or a detector.

    Inputs:
        - tb : pandas DataFrame.
        - tracker : name of tracker.
        - detector : name of the detector.

    Outputs:
        - Data frame with the required information.
    '''

    if detector:

        return tb.loc[(tb['Tracker'] == tracker) & (tb['Detector'] == detector)]


    return tb.loc[(tb['Tracker'] == tracker)]



def select_list(tb, trackers, detectors):

    list_tb = []

    for trc in trackers:

        for det in detectors:

            list_tb.append( tb.loc[(tb['Tracker'] == trc) & (tb['Detector'] == det)] )


    return pd.concat(list_tb)




def tracking_metrics(tb):
    '''
    Return a Data Frame with only the tracking metrics.

    Inputs:
        - tb : pandas DataFrame.

    Outputs:
        - Data Frame with the needed columns.
    '''

    return tb.iloc[:, 11:]


def detection_metrics(tb):
    '''
    Return a Data Frame with only the detection metrics.

    Inputs:
        - tb : pandas DataFrame.

    Outputs:
        - Data Frame with the needed columns.
    '''

    return tb.iloc[:, 4:11]


def all_metrics(tb):
    '''
    Return a Data Frame with all metrics.

    Inputs:
        - tb : pandas DataFrame.

    Outputs:
        - Data Frame with the needed columns.
    '''

    return tb.iloc[:, 4:]



def plot_matrix(data, labels, plot_values=True, figsize=(12, 12), file_name=None):
    '''
    Plot a matrix.

    Inputs:
        - data : matrix with the data to plot.
        - labels : labels to plot (x and y axis must be the same).
        - plot_values : (Boolean) True: plot values in the figure.
        - figsize : size of the figure.

    Outputs:
        - (None)
    '''

    # Create figure.
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)


    # Plot matrix
    cax = ax.matshow(data, interpolation='nearest')


    # Color Bar
    bax = fig.colorbar(cax)
    bax.ax.tick_params(labelsize=20)


    # Configure and plot labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))

    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticklabels(labels, fontsize=20)

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left")


    # Plot values of the matrix
    if plot_values:

        # Loop over data dimensions and create text annotations.
        for i in range(len(labels)):
            for j in range(len(labels)):
                
                if data[i, j] > 0.8: color = 'b'
                else: color = 'w'
                
                text = ax.text(j, i, '%.2f' % (data[i, j]),
                               ha="center", va="center", color=color, fontsize=15)

    #ax.set_title("Matrix comparing detection and tracking metrics.")


    if file_name: plt.savefig('img/'+file_name, dpi=160)

    plt.show()





def plot_histogram(data, n_bins, title, xlabel="Values", ylabel="Number of Elements", figsize=(20,10), file_name=None):

    fig = plt.figure(figsize=figsize)

    plt.hist(data, bins=n_bins)

    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)

    if file_name: plt.savefig('img/'+file_name, dpi=160)

    plt.show()



def dots_graph(data, metric_x, metric_y, otype='diff', figsize=(15, 8)):
    
    x = []
    y = []
    
    fig = plt.figure(figsize=figsize)
    
    for d in data:
        
        if otype == 'diff':
            
            x.append( abs(d[0][metric_x] - d[1][metric_x]) )
            y.append( abs(d[0][metric_y] - d[1][metric_y]) )
        
        elif otype == 'all':
            
            x.append(d[0][metric_x])
            x.append(d[1][metric_x])
            y.append(d[0][metric_y])
            y.append(d[1][metric_y])
            
        else:
            
            raise Exception('oType incorrect.')
            

    plt.title('Comparing ' + metric_x + ' and ' + metric_y, fontsize=30)
    plt.xlabel(metric_x, fontsize=20)
    plt.ylabel(metric_y, fontsize=20)
    
    plt.scatter(x, y)
    plt.show()