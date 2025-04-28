import pandas as pd
import numpy as np
import joblib
from sklearn.datasets import make_regression
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures


def load_dataset(filename, trim_quantiles: list = None, qlo=0.25, qhi=0.75, plothist=False,
                 histfig: str = 'hist', format='png', n_poly: int = 1, usecols: list = None,
                 input_feature_names: list = ['x'], output_feature_indices: list = None,
                 y_col: str = None, yerr_col: str = None, subset_expr: str = None,
                 dropna_cols: list = None, comment='#', pca_model: object = None):
    """
    Loads, trims, and exports dataset to numpy arrays.

    :param filename: string
    The name of file to read from.

    :param trim_quantiles: list
    If provided, data beyond the lower and upper quantiles (qlo, qhi) of the listed column names will be trimmed.

    :param qlo: float
    The lower quantile below which to trim the data.

    :param qhi: float
    The upper quantile above which to trim the data.

    :param plothist: bolean
    Whether to plot histograms of the data in the list of columns provided by usecols.

    :param histfig: string
    Name of the figure file of the plotted histograms.

    :param format: string
    Format of the plotted histogram figure file.

    :param n_poly: positive integer
    The order of the polynomial basis of the output features. If larger than 1, polynomial features of the provided
    order will be created from the feature list given by input_feature_names.

    :param usecols: list
    List of column names to be read in from the input file.
    Passed as the usecols parameter in the argument of pandas.read_csv.

    :param input_feature_names: list
    List of the column names to be used as covariate features.

    :param output_feature_indices: list of integers
    List of the indices of the features to be used. Useful if one wants to select specific features when n_poly>1.

    :param y_col: string
    Name of the column to be considered as the target variable.

    :param yerr_col:
    Name of the column to be considered as the uncertainty on the target variable.

    :param subset_expr: string
    Expression to be used for specifying threshold rejections to be applied on one or more variables listed in usecols.
    Passed to the argument of pandas.dataframe.query().

    :param dropna_cols: list
    List of the column names for which all rows containing NaN values should be omitted.

    :param comment: string
    If a row in the input file starts with this character string, it will be trated as a comment line.
    with the exception of the first row which must start with a "#" and contain the column names.

    :param pca_model: object (scikit-learn PCA transformer object)
    PCA-transformation to be applied on the input data, must include the standardization step.
    Applied on the features in input_feature_names after the threshold rejections and
    before the polynomial transformation.

    :return:
    X: ndarray: The final data matrix, shape: (n_samples, n_features).
    y: ndarray: Values of the target variable, shape: (n_samples, )
    yw: ndarray: Uncertainties of the target variable, shape: (n_samples, )
    df: pandas dataframe object: all input data after quantile trimming threshold rejections
    feature_names: list: The names of the features in X.
    df_orig: The original data frame as read from the input file.
    """

    with open(filename) as f:
        header = f.readline()
    cols = header.strip('#').split()
    df = pd.read_csv(filename, names=cols, header=None, sep='\s+', usecols=usecols, comment=comment)
    if dropna_cols is not None:
        df.dropna(inplace=True, subset=dropna_cols)
    ndata = len(df)
    print(df.head())
    print("----------\n{} lines read from {}\n".format(ndata, filename))

    df_orig = df

    # Apply threshold rejections:
    if subset_expr is not None:
        df = df.query(subset_expr)

        ndata = len(df)
        print("{} lines after threshold rejections\n".format(ndata))

    # plot histogram for each column in original dataset
    if plothist:
        fig, ax = plt.subplots(figsize=(20, 10))
        # df.hist('ColumnName', ax=ax)
        # fig.savefig('example.png')
        _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False, color='red',
                              ax=ax)
        plt.savefig(histfig + '.' + format, format=format)

    # omit data beyond specific quantiles [qlo, qhi]
    if trim_quantiles is not None:

        dfq = df[trim_quantiles]
        quantiles = pd.DataFrame.quantile(dfq, q=[qlo, qhi], axis=0, numeric_only=True, interpolation='linear')
        print("Values at [{},{}] quantiles to be applied for data trimming:".format(qlo, qhi))
        print(quantiles.sum)
        # df_t = df[( df > df.quantile(qlo) ) & ( df < df.quantile(qhi) )]
        mask = (dfq > dfq.quantile(qlo)) & (dfq < dfq.quantile(qhi))
        # print(mask)
        mask = mask.all(axis=1)
        # print(mask.shape)
        df = pd.DataFrame.dropna(df[mask])
        ndata = len(df)
        print("\n{} lines remained after quantile rejection.\n".format(ndata))
        # plot histogram for each column in trimmed dataset
        if plothist:
            fig, ax = plt.subplots(figsize=(20, 10))
            _ = pd.DataFrame.hist(df, bins=int(np.ceil(np.cbrt(ndata) * 2)), figsize=(20, 10), grid=False,
                                  color='green', ax=ax)
            fig.savefig(histfig + "_trimmed." + format)

    # extract input features:
    X = df.loc[:, input_feature_names].to_numpy()

    # Apply PCA transformation on X using a previously trained model:
    if pca_model is not None:
        pca_model = joblib.load(pca_model)
        X = pca_model.transform(X)
        input_feature_names = ["E{}".format(i + 1) for i in range(X.shape[1])]

    # Extract column of target variable:
    if y_col is not None:
        # print(df_t[y_col])
        y = df[y_col].to_numpy()
    else:
        y = np.array([])

    # Define weights:
    if yerr_col is not None:
        df['weight'] = 1.0 / df[yerr_col] ** 2
        yw = df['weight'].to_numpy()
    else:
        yw = np.ones_like(y)

    # create polynomial features:
    if n_poly > 1:
        trans = PolynomialFeatures(degree=n_poly, include_bias=False)
        X = trans.fit_transform(X)
        input_feature_names = trans.get_feature_names(input_feature_names)

    if output_feature_indices is not None:
        feature_names = np.array(input_feature_names)[output_feature_indices]
        X = X[:, output_feature_indices]
    else:
        feature_names = input_feature_names

    return X, y, yw, df, feature_names, df_orig


def load_test_dataset(n_features=8, n_informative=4, n_samples=200, n_poly=1):
    """
    Function to load a test dataset.

    :param n_features: int
    Number of covariate features.

    :param n_informative: int
    Number of informative features.

    :param n_samples: int
    The number of data points.

    :param n_poly: int
    The order of the polynomial basis.

    :return:
    X: ndarray: The final covariate features, shape: (n_samples, n_features).
    y: ndarray: The values of the target variable, shape: (n_samples, )
    feature_names: list: The names of the features in X.
    """
    X, y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_informative,
                           effective_rank=7, tail_strength=0.7,
                           noise=2, bias=20, random_state=1)
    feature_names = ['f' + str(i + 1) for i in range(X.shape[1])]
    if n_poly is not None:
        trans = PolynomialFeatures(degree=n_poly, include_bias=False)
        X = trans.fit_transform(X)
        feature_names = trans.get_feature_names(feature_names)
    return X, y, feature_names


def plot_residual(y: np.ndarray, yhat: np.ndarray, xlabel: str = "", ylabel: str = "", highlight: np.ndarray = None,
                  plotrange: tuple = None, fname: str = None, format="png"):
    """
    Plot predicted vs ground truth values.

    :param y: 1-d ndarray
    Array with the ground truth values.

    :param yhat: 1-d ndarray
    Array with the predicted values.

    :param xlabel: string
    Label for the x axis.

    :param ylabel: string
    Label for the y axis.

    :param highlight: boolean ndarray
    Boolean array matching the shape of y. True values will be highlighted in the plot.

    :param plotrange: tuple
    Range to be applied to both the x and y axes of the plot.

    :param fname: string
    Name of the output figure file.

    :param format: string
    Format of the output figure file.
    """

    fig = plt.figure(figsize=(5, 4))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(y, yhat, 'ko')
    if highlight is not None:
        plt.plot(y[highlight], yhat[highlight], 'rx', alpha=0.5)
    minval = np.min((y.min(), yhat.min()))
    maxval = np.max((y.max(), yhat.max()))
    ax = plt.gca()
    ax.set_aspect('equal')
    if plotrange is not None:
        plt.xlim(plotrange)
        plt.ylim(plotrange)
        grid = np.linspace(plotrange[0], plotrange[1], 100)
    else:
        grid = np.linspace(minval, maxval, 100)
    plt.plot(grid, grid, 'r-')
    plt.tight_layout()
    if fname is not None:
        plt.savefig(fname + '.' + format, format=format)
        plt.close(fig)
    else:
        plt.show()
