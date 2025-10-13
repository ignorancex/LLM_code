import numpy as np
from scipy import stats
from typing import Union
import warnings
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
import util
import copy


# noinspection SpellCheckingInspection
def test_normality(data: np.ndarray, normality_level: str = 'normal') -> bool:
    """
    The normality test is performed using D'Agnostino and Pearson's test.

    :param data: If the input is a 2D array, the data should be grouped in columns;
    :param normality_level: str on requirement of normality check calculated p_value, default: 'normal', a p-value of
                           <0.05 will flag deviation from normality;
                           'low', skip normality check if data has larger than 30 samples;
                           'high', a p_value of >0.05 < 0.1 to trigger attention in the downstream;
    :return: no_attention_needed: bool on whether e.g., Levene and its derived methods to be applied;
    """
    _, p_value = stats.normaltest(data)
    no_attention_needed = True
    if isinstance(p_value, np.ndarray):
        idx_nn = [i for i, _p in enumerate(p_value) if _p <= 0.05]
        if len(idx_nn) > 0:
            if normality_level == 'low' and data.shape[0] >= 30:
                warnings.warn(f'Samples with indices of {idx_nn} are not normally distributed with p-values of '
                              f'{[p_value[i] for i in idx_nn]}. However, as there are in total {data.shape[0]} '
                              f'samples (>=30), according to central limit theorem, t-test can be still applicable!')
            else:
                no_attention_needed = False
                warnings.warn(f'Indices of non-normally distributed: {idx_nn} with corresponding p-values of '
                              f'{[p_value[i] for i in idx_nn]}')
        if normality_level == 'high':
            idx_att = [i for i, _p in enumerate(p_value) if 0.05 < _p < 0.1]
            if len(idx_att) > 0:
                no_attention_needed = False
                warnings.warn(f'Data indices need attention for normality check: {idx_att} with corresponding p-values'
                              f' of {[p_value[i] for i in idx_nn]}')

    else:
        if p_value <= 0.05:
            if normality_level == 'low' and len(data) >= 30:
                warnings.warn(f'The data are not normally distributed with p-value of {p_value}. However, as there are '
                              f'in total {len(data)} samples (>=30), according to central limit theorem, '
                              f't-test can be still applicable!')
            else:
                no_attention_needed = False
                warnings.warn(f'Normal distribution: False!')
        if normality_level == 'high':
            if 0.05 < p_value < 0.1:
                warnings.warn(f'Data has moderate normality.')
                no_attention_needed = False
    return no_attention_needed


# noinspection SpellCheckingInspection
def test_variance_equality(a: np.ndarray, b: np.ndarray, normal_sens: bool = True,
                           forced_method_flag: Union[int, None] = None):
    """
    If both input arrays are normally distributed, simple F-test is performed, i.e, flag_method = 0.
    If not, Levene or its derived method will be used instead.
    The degrees of kurtosis and skewness are measured to select the oppropriate method for the downstream tests.
    If both distributions deviate strongly from normal and have heavy tails, i.e. positive excess kurtosis (Fisher's
    definition, or Pearson's definition subtracted by 3), trimmed version of Forsythe test, derived Levene is used,
    i.e., flag_method = 3.
    If the distributions are highly skewed, i.e. absolute skew is > 1, conventional Brown and Forsythe test is used,
    i.e., derived Levene using median value instead of mean, with flag_method = 2.
    Otherwise, the original Levene method is used, i.e., flag_method = 1.

    :param a: 1d array;
    :param b: 1d array;
    :param normal_sens: bool on whether to strictly require a p_value of <0.05 to flag deviation from normality
                        or a p_value of < 0.1 to trigger attention, i.e., Levene and its derived methods to be applied;
                        However, it is important to note that if forced_method_flag is None, this parameter will
                        be ignored.
    :param forced_method_flag: int or None, if not None, it should be in [0, 1, 2, 3], corresponding to flag_methods
                               explained above;
                               Though supported, it is not recommended to set forced_method_flag to 0, as it might
                               violate the presumption that the data arrays should be normally distributed. Warnings
                               may be triggered if this is the case.
    :return: f_value, p_value and flag_method if forced_method_flag is None, or f_value and p_value.
    """
    method = {0: 'F-test', 1: 'Levene_mean', 2: 'Brown-Forsythe_median', 3: 'Brown-Forsythe_trimmed'}
    assert np.ndim(a) == np.ndim(b) == 1

    def f_test():
        _f_value = np.var(a) / np.var(b)
        _p_value = stats.f.sf(_f_value, len(a) - 1, len(b) - 1)
        return _f_value, _p_value

    if normal_sens:
        normality_level = 'normal'
    else:
        normality_level = 'high'
    if forced_method_flag is None:
        if test_normality(a, normality_level) and test_normality(b, normality_level):
            print('Both distributions are normal. F-test is applied.')
            flag_method = 0
            f_value, p_value = f_test()
        else:
            a_kt = stats.kurtosis(a)
            b_kt = stats.kurtosis(b)
            if a_kt > 0 and b_kt > 0:
                flag_method = 3
                print('Both distributions are not strictly normal and have large kurtosis. Use trimmed version of '
                      'Brown-Forsythe test for variance equality.')
            else:
                a_sk = stats.skew(a)
                b_sk = stats.skew(b)
                if abs(a_sk) > 1 and abs(b_sk) > 1:
                    flag_method = 2
                    print('Both distributions are not strictly normal and have large skewness. Use Brown-Forsythe test'
                          ' for variance equality.')
                else:
                    flag_method = 1
                    print('Both distributions are not strictly normal but have moderate or low skewness. Use Levene-'
                          'mean test for variance equality.')
            _center = method[flag_method].split('_')[1]
            f_value, p_value = stats.levene(a, b, center=_center)
        return f_value, p_value, flag_method
    else:
        print(f"{method[forced_method_flag].split('_')[0]} method is forced to apply.")
        if forced_method_flag == 0:
            if not (test_normality(a, normality_level) and test_normality(b, normality_level)):
                warnings.warn('It is not recommended to force F-test as normality requirement is not satisefied!')
            f_value, p_value = f_test()
        else:
            _center = method[forced_method_flag].split('_')[1]
            f_value, p_value = stats.levene(a, b, center=_center)
        return f_value, p_value


# noinspection SpellCheckingInspection
def ttest_with_auto_checks(a: np.ndarray, b: np.ndarray, paired: bool = False, alternative: str = 'two-sided',
                           ttest_high_tolerance: bool = False, return_f_test: bool = False,
                           ftest_normal_sens: bool = True) -> dict:
    """

    :param a: 2D or of lower dimension. If 2D, samples are in rows and variables (raters) in columns, i.e. n*k;
    :param b: array of same shape as 'a';
    :param paired: bool on whether the samples are considered independent or paired;
    :param alternative: str, optional values in ['two-sided', 'greater', 'less'], default 'two-sided';
    :param ttest_high_tolerance: bool on whether to put a low requiremnt on normality for t-test:
    :param return_f_test: bool on whether to return f_value when testing variance equality;
    :param ftest_normal_sens: bool on whether to strictly require a p_value of <0.05 to flag deviation from normality
                              for f-test;
    :return: dict_stats: dict, which contains p_value and possibly f_value;
    """

    # noinspection SpellCheckingInspection
    def _test(_a: np.ndarray, _b: np.ndarray, _flag: int, equal_var: Union[bool, None] = None):
        if _flag == 0:
            if not paired:
                words_method = ["comparable", "regular t-test"] if equal_var else ["different", "Welch's t-test"]
                print(f"The input arrays have statistically {words_method[0]} variances, {words_method[1]}"
                      f" will be performed.")
                r = stats.ttest_ind(_a, _b, equal_var=equal_var, alternative=alternative)
            else:
                print('Paried t-test without assumption of equal variances but normal distributions will be performed.')
                r = stats.ttest_rel(_a, _b, alternative=alternative)
        else:
            if not paired:
                print('For two independent non-normally distributed samples, Mann-Whitney U test will be performed')
                r = stats.mannwhitneyu(_a, _b, alternative=alternative)
            else:
                print('For the paired samples non-normally distributed Wilcoxon signed-rank test will be performed.')
                r = stats.wilcoxon(_a, _b, alternative=alternative)
        # noinspection PyUnresolvedReferences
        return r.pvalue

    assert np.ndim(a) == np.ndim(b) <= 2
    if np.ndim(a) == 2:
        k = a.shape[1]
        f_value = np.zeros(k)
        p_var = np.zeros(k)
        flag = np.zeros(k)
        for i in range(k):
            _f, _p, _flag = test_variance_equality(a[:, i], b[:, i], ftest_normal_sens)
            f_value[i] = _f
            p_var[i] = _p
            flag[i] = _flag
        # flag_uniq = np.unique(flag)
        # if len(flag_uniq) > 1:  # If 1, no need to set a forced_method_flag as the same method has been applied.
        #     flag_uniq = np.max(flag_uniq)
        #     for i in range(k):
        #         _f, _p = test_variance_equality(a[:, i], b[:, i], ftest_normal_sens, forced_method_flag=flag_uniq)
        #         f_value[i] = _f
        #         p_var[i] = _p
        # else:
        #     flag_uniq = flag_uniq[0]
        p_value = np.zeros(k)
        for i, _p in enumerate(p_var):
            _flg = 0 if (ttest_high_tolerance and len(a[:, i]) >= 30) else flag[i]
            _p_var = _test(a[:, i], b[:, i], _flg, _p >= 0.05)
            p_value[i] = _p_var
    else:
        f_value, p_var, flag = test_variance_equality(a, b, ftest_normal_sens)
        _flg = 0 if (ttest_high_tolerance and len(a) >= 30) else flag
        p_value = _test(a, b, _flg, p_var >= 0.05)

    dict_stats = util.NestedDict()
    dict_stats['p-value'] = p_value
    if return_f_test:
        method = {0: 'F-test', 1: 'Levene_mean', 2: 'Brown-Forsythe_median', 3: 'Brown-Forsythe_trimmed'}
        test_name = method[flag] if isinstance(flag, int) else [method[_f] for _f in flag]
        dict_stats['f-test'] = {'f-value': f_value, 'p-value': p_var, 'test-name': test_name}
    return dict_stats


# noinspection SpellCheckingInspection
def set_level_to_pval(p_val):
    if p_val <= 0.05:
        if p_val > 0.01:
            level = '*'
        elif 0.001 < p_val <= 0.01:
            level = '**'
        elif 0.0001 < p_val <= 0.001:
            level = '***'
        else:
            level = '****'
    else:
        level = 'ns'
    return level


# noinspection SpellCheckingInspection
def get_optimal_hist_bin_width(data, method: str = 'doane', round_to_int=False, plot_result=False):
    edges = np.histogram_bin_edges(data, bins=method)
    if plot_result:
        plt.hist(data, edges)
        plt.show()
    if round_to_int:
        bin_width = int(round(edges[1] - edges[0]))
    else:
        bin_width = edges[1] - edges[0]
    print(f'Optimal bin width: {bin_width}')
    return bin_width


def estimate_kernel_density(img_3d: np.ndarray, mask: np.ndarray, kernel='gaussian',
                            plot_result=False, orientation: Union[str, None] = None):
    assert mask.dtype == np.bool_
    data = img_3d[mask]
    edge_start = int(data.min() // 100 * 100)
    edge_end = int(np.ceil(data.max() / 100)) * 100
    bin_width = get_optimal_hist_bin_width(data, round_to_int=True)
    edges = np.array(range(edge_start, edge_end + bin_width, bin_width))
    hist = np.histogram(data, bins=edges, density=True)[0]
    kde = KernelDensity(kernel=kernel, bandwidth=0.5).fit(data.reshape(-1, 1))
    # noinspection PyUnresolvedReferences
    dens_log = kde.score_samples(edges.reshape(-1, 1))
    if plot_result:
        if orientation is None:
            orientation = 'vertical'
        # noinspection PyTypeChecker
        plt.stairs(hist, edges=edges, fill=True, alpha=0.5, orientation=orientation)
        if orientation == 'vertical':
            plt.plot(edges, np.exp(dens_log), label='Gaussian kernel')
        else:
            plt.plot(np.exp(dens_log), edges, label='Gaussian kernel')
        plt.show()

    return edges, dens_log, hist


# noinspection SpellCheckingInspection
def get_icc(data: np.ndarray, icc_type: str = '2_1'):
    """
     Calculate intra-class correlation coefficient
        ICC Formulas are based on:
        Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
        assessing rater reliability. Psychological bulletin, 86(2), 420.
        icc1:  x_ij = mu + beta_j + w_ij
        icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
        Code modifed from nipype algorithms.icc
        https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py

    :param data: 2D array, samples are in rows and variables (raters) in columns, i.e. n*k; In most cases n >> k;
    :param icc_type: Supported types: '2_1', '2_k', '3_1' or '3_k':
                     '2_1': Each subject is measured by each rater, and raters are considered representative of a
                     larger population of similar raters. The reliability is calculated from a single measurement.
                     '2_k': The difference with '2_1' is that the reliability is calculated by taking an average of the
                     k rater's measurements.
                     '3_1': Each subject is measured by each rater. But the raters are the only raters of interest. The
                      reliability is calculated from a single measurement.
                     '3_k': Reliability is calculated by taking an average of the k rater's measurements.
    :return: icc: intra-class correlation coefficient
    """
    assert np.ndim(data) == 2 and data.shape[1] >= 2
    [n, k] = data.shape
    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(data)
    SST = ((data - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))),
                                X.T), data.flatten('F'))
    residuals = data.flatten('F') - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between columns
    SSC = ((np.mean(data, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc  # / n (without n in SPSS results)

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == '2_1' or icc_type == '2_k':
        if icc_type == '2_k':
            k = 1
        icc = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == '3_1' or icc_type == '3_k':
        if icc_type == '3_k':
            k = 1
        icc = (MSR - MSE) / (MSR + (k - 1) * MSE)
    else:
        # icc1
        raise NotImplementedError("This method isn't implemented yet.")
    return icc


# noinspection SpellCheckingInspection
def get_occc(data: np.ndarray):
    """
    :param data: 2D array, samples are in rows and variables (raters) in columns, i.e. n*k; In most cases, n >> k;
    :return: occc: overall concordance correlation coefficient. Observe that for k == 2, OCCC becomes the original CCC.
    """
    k = data.shape[1]
    # Make sure that the input array is 2D.
    assert np.ndim(data) == 2 and k >= 2

    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    covs = np.cov(data, rowvar=False)
    # Generate paird indices
    ids = np.vstack(np.tril_indices(k, -1))
    ccc = np.zeros(ids.shape[1])
    ksi = np.zeros(ids.shape[1])
    for i in range(ids.shape[1]):
        ksi[i] = (means[ids[0, i]] - means[ids[1, i]]) ** 2 + stds[ids[0, i]] ** 2 + stds[ids[1, i]] ** 2
        ccc[i] = 2 * covs[ids[0, i], ids[1, i]] / ksi[i]
    if k == 2:
        occc = ccc[0]
    else:
        occc = np.sum(ksi * ccc) / np.sum(ksi)
    return occc


# noinspection SpellCheckingInspection
def compute_data_stats_in_dict(dict_results: util.NestedDict, dict_stats: dict, feature_cls: Union[str, None],
                               type_modes: list, **kwargs):
    ftest_normal_sens = kwargs.get('ftest_normal_sens', True)
    paired = kwargs.get('paired', True)
    alternative = kwargs.get('alternative', 'two-sided')
    return_f_test = kwargs.get('return_f_test', True)
    ttest_high_tolerance = kwargs.get('ttest_high_tolerance', True)
    if feature_cls is not None:
        dict_ = dict_results[feature_cls]
        dict_st = dict_stats[feature_cls]
    else:
        dict_ = dict_results
        dict_st = dict_stats
    list_fnames = list(dict_.keys())
    dict_st['features'] = list_fnames
    # Boxplots assoume that the dataset is organized in columns, while vertical stacking puts the
    # result arrays in rows. Therefore, we transpose the stacked arrays before the plot.
    a = np.vstack([dict_[_k][type_modes[0]] for _k in list_fnames]).T
    b = np.vstack([dict_[_k][type_modes[1]] for _k in list_fnames]).T
    # use simpleStats.ttest_with_auto_checks instead of scipy.stats.ttest.
    _dict_st = ttest_with_auto_checks(a, b, paired, alternative, ttest_high_tolerance, return_f_test, ftest_normal_sens)
    dict_t = dict(zip(list_fnames, _dict_st['p-value']))
    print(f"The results for t test of comparing {' and '.join(type_modes)} for feature class of {feature_cls} are: ")
    for _f_name, _pvalue in dict_t.items():
        print(f'\t{_f_name.capitalize()}: significant = {_pvalue <= 0.05}, p-value = {_pvalue}')
    dict_st['p-value'] = list(_dict_st['p-value'])
    if return_f_test:
        dict_st['f-test'] = {'test': _dict_st['f-test']['test-name'], 'median': np.median(b),
                             'var': np.var(b), 'ratio_median': np.median(b, axis=0) / np.median(a, axis=0),
                             'ratio_var': np.var(b, axis=0) / np.var(a, axis=0),
                             'p-value': _dict_st['f-test']['p-value']}
        for j, _t in enumerate(_dict_st['f-test']['test-name']):
            f_test_name = (lambda _x: (lambda _x: _x if "test" in _x else f"{_x} test")(_x.split('_')[0]))(_t)
            _pvalue = dict_st['p-value'][j]
            is_significant = 'significant' if _pvalue <= 0.05 else 'not significant'
            if feature_cls is not None:
                print(f"For {f_test_name} of comparing {' and '.join(type_modes)} for feature "
                      f"{list_fnames[j]} of class {feature_cls} is: {is_significant} with p-value of {_pvalue}")
            else:
                print(f"For {f_test_name} of comparing {' and '.join(type_modes)} for feature {list_fnames[j]} "
                      f"is: {is_significant} with p-value of {_pvalue}")


# noinspection SpellCheckingInspection
def boxplot_stats_in_dict(dict_results: util.NestedDict, dict_stats: dict, feature_cls: Union[str, None],
                          type_modes: list, return_f_test: bool = True, labels: Union[list, None] = None):
    if feature_cls is not None:
        dict_ = dict_results[feature_cls]
        dict_st = dict_stats[feature_cls]
        hatch_style = '///'
    else:
        dict_ = dict_results
        dict_st = dict_stats
        hatch_style = '..'
    fig, ax = plt.subplots(1, 1, layout='constrained')
    list_fnames = list(dict_.keys())
    num_features = len(list_fnames)
    list_colors = mpl.colormaps['turbo'](np.linspace(0, 1, num_features))
    # Append boxes to list for creating legend
    list_box_lg = list()
    for t in type_modes:
        result = np.vstack([dict_[_k][t] for _k in list_fnames]).T
        # Enable patch artist to allow setting modification; use 95% confidence interval bounds as whiskers;
        if t == type_modes[0]:
            plots = ax.boxplot(result, patch_artist=True, whis=(2.5, 97.5), showfliers=False, vert=False,
                               whiskerprops=dict(color='gray'), medianprops=dict(color='gray'))
            for _b, _c in zip(plots['boxes'], list_colors):
                _b.set(color=_c)
                _b.set_facecolor('none')
        else:
            plots = ax.boxplot(result, patch_artist=True, whis=(2.5, 97.5), showfliers=False, vert=False)
            for _b, _c in zip(plots['boxes'], list_colors):
                _b.set(hatch=hatch_style)
                _b.set_facecolor(_c[:3])
                _b.set_alpha(0.5)
        _box = copy.deepcopy(plots['boxes'][0])
        _box.set(color='gray')
        if t == 'original':
            _box.set_facecolor('none')
        list_box_lg.append(_box)
    if labels is not None:
        ax.legend(list_box_lg, labels, loc='upper left')
    else:
        ax.legend(list_box_lg, type_modes, loc='upper left')
    ax.set_yticks(np.arange(1, num_features + 1), labels=list_fnames)
    ax.set_xlabel('OCCC')
    ax.set_xticks(np.linspace(0, 1, 6))
    ax2 = ax.twinx()
    ax2.set_ylim(ax.get_ylim())
    level_t_pval = [set_level_to_pval(_p) for _p in dict_st['p-value']]
    if return_f_test:
        level_f_pval = [set_level_to_pval(_p) for _p in dict_st['f-test']['p-value']]
        ax2.set_yticks(np.arange(1, num_features + 1),
                       labels=[f'({_a} / {_b})' for _a, _b in zip(level_t_pval, level_f_pval)])
    else:
        ax2.set_yticks(np.arange(1, num_features + 1), labels=level_t_pval)
    return fig


# noinspection SpellCheckingInspection
def plot_var_median_ratio_in_dict(dict_stats: dict, feature_cls: Union[str, None]):
    from matplotlib.lines import Line2D
    from matplotlib.markers import MarkerStyle
    from matplotlib import ticker

    if feature_cls is not None:
        dict_st = dict_stats[feature_cls]
    else:
        dict_st = dict_stats
    list_fnames = dict_st['features']
    num_features = len(list_fnames)
    list_colors = mpl.colormaps['turbo'](np.linspace(0, 1, num_features))
    fig, ax = plt.subplots(1, 1, layout='constrained')
    markers_filled = list(Line2D.filled_markers)
    list_markers = util.resample_with_low_repeats(markers_filled, num_features)
    for j in range(num_features):
        ax.scatter(dict_st['f-test']['ratio_median'][j], 100 * (1 - dict_st['f-test']['ratio_var'][j]),
                      c=list_colors[j][:3].reshape(1, -1), label=list_fnames[j],
                      marker=MarkerStyle(list_markers[j], fillstyle='full'), alpha=0.8)
    ax.axvline(1.0, color='gray', linewidth=0.5, linestyle='--')
    ax.yaxis.set_major_formatter(ticker.PercentFormatter())
    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'$OCCC_{median}^{standardized}/OCCC_{median}^{original}$', fontsize=12)
    ax.set_ylabel('Reduction of OCCC variance', fontsize=12)
    ax.legend()
    return fig


# noinspection SpellCheckingInspection
def boxplot_with_sig_bar(dict_results: dict, labels: list):
    # Support only upto 3 data arrays stored in dict_results

    # noinspection SpellCheckingInspection
    def add_diff_label_to_plot(_i: int, _j: int, x_coord: np.ndarray, y_coord: np.ndarray, txt: str):
        props = {'connectionstyle': 'bar, fraction=0.05', 'arrowstyle': '-', 'shrinkA': 2, 'shrinkB': 2,
                 'linewidth': 1,
                 'color': 'gray'}
        x = (x_coord[_i] + x_coord[_j]) / 2
        y = max(y_coord[_i], y_coord[_j])
        ax.annotate(txt, xy=(x, 1.02 * y), ha='center')
        ax.annotate('', xy=(x_coord[_i], y), xytext=(x_coord[_j], y), arrowprops=props)

    methods = [_k for _k, _v in dict_results.items() if isinstance(_v, np.ndarray)]
    selected_feature = [_k for _k, _v in dict_results.items() if isinstance(_v, str)][0]
    data = np.vstack([dict_results[k] for k in methods]).T
    hatches = ['///', '..', '\\\\']
    fig, ax = plt.subplots(1, 1, layout='constrained')
    plot = ax.boxplot(data, patch_artist=True, whis=(2.5, 97.5), showfliers=False)
    for j in range(len(methods)):
        plot['boxes'][j].set(hatch=hatches[j])
        plot['boxes'][j].set_facecolor(dict_results['color'][:3])
        plot['boxes'][j].set_alpha(0.5)
    ax.set_ylabel('OCCC', fontsize=12)
    ax.set_xticks(np.arange(1, len(methods) + 1), labels=labels)
    ax.set_title(f'Selected feature: {selected_feature}')
    comb = list(zip([0]*(len(methods)-1), np.arange(1, len(methods))))
    for _idx_p in comb:
        _dict_stats = ttest_with_auto_checks(data[:, _idx_p[0]], data[:, _idx_p[1]], paired=True,
                                             ttest_high_tolerance=True, return_f_test=True)
        _pvals = [_dict_stats['p-value'], _dict_stats['f-test']['p-value']]
        _levels = [set_level_to_pval(_p) for _p in _pvals]
        add_diff_label_to_plot(_idx_p[0], _idx_p[1], np.arange(1, len(methods)+1), np.max(data, axis=0),
                               f'({_levels[0]} / {_levels[1]})')


def get_index_of_plane_with_largest_mask_area(img_mask: Union[np.ndarray, dict], dim: str = 'z',
                                              cls_map_selected: Union[dict, None] = None,
                                              aspect_ratios=(1, 1, 1)):
    import statistics as stat
    assert dim in ['x', 'y', 'z']
    if dim == 'z':
        area_axial_norm = aspect_ratios[0] * aspect_ratios[1]
    elif dim == 'y':
        area_axial_norm = aspect_ratios[0] * aspect_ratios[2]
    else:
        area_axial_norm = aspect_ratios[1] * aspect_ratios[2]

    def get_index_with_max_binary_mask_area(_mask):
        if dim == 'z':
            _area_list = [np.sum(_mask[:, :, _i])*area_axial_norm for _i in range(_mask.shape[2])]
        elif dim == 'y':
            _area_list = [np.sum(_mask[:, _i, :]) * area_axial_norm for _i in range(_mask.shape[1])]
        else:
            _area_list = [np.sum(_mask[_i, :, :]) * area_axial_norm for _i in range(_mask.shape[0])]
        idx = _area_list.index(max(_area_list))
        return idx
    if isinstance(img_mask, np.ndarray):
        if img_mask.dtype == np.bool_:
            idx_max_area = get_index_with_max_binary_mask_area(img_mask)
        else:
            if cls_map_selected is None:
                img_mask = img_mask.astype(np.bool_)
                idx_max_area = get_index_with_max_binary_mask_area(img_mask)
            else:
                labels = np.unique(img_mask[img_mask != 0])
                label_selected = list(cls_map_selected.keys())
                assert all([_l in labels for _l in label_selected])
                if dim == 'z':
                    area_arr = np.zeros((img_mask.shape[2], len(label_selected)))
                    for i, _label in enumerate(label_selected):
                        mask = img_mask == _label
                        area_arr[:, i] = np.array([np.sum(mask[:, :, _i])*area_axial_norm
                                                   for _i in range(mask.shape[2])])
                    area_list = [stat.harmonic_mean(area_arr[_i, :]) for _i in range(img_mask.shape[2])]
                elif dim == 'y':
                    area_arr = np.zeros((img_mask.shape[1], len(label_selected)))
                    for i, _label in enumerate(label_selected):
                        mask = img_mask == _label
                        area_arr[:, i] = np.array([np.sum(mask[:, _i, :]) * area_axial_norm
                                                   for _i in range(mask.shape[1])])
                    area_list = [stat.harmonic_mean(area_arr[_i, :]) for _i in range(img_mask.shape[1])]
                else:
                    area_arr = np.zeros((img_mask.shape[0], len(label_selected)))
                    for i, _label in enumerate(label_selected):
                        mask = img_mask == _label
                        area_arr[:, i] = np.array([np.sum(mask[_i, :, :]) * area_axial_norm
                                                   for _i in range(mask.shape[0])])
                    area_list = [stat.harmonic_mean(area_arr[_i, :]) for _i in range(img_mask.shape[0])]
                idx_max_area = area_list.index(max(area_list))
    else:
        masks = [v for v in img_mask.values() if isinstance(v, np.ndarray)]
        if dim == 'z':
            area_arr = np.zeros((masks[0].shape[2], len(masks)))
            for i, mask in enumerate(masks):
                area_arr[:, i] = np.array([np.sum(mask[:, :, _i]) * area_axial_norm
                                           for _i in range(mask.shape[2])])
            area_list = [stat.harmonic_mean(area_arr[_i, :]) for _i in range(masks[0].shape[2])]
        elif dim == 'y':
            area_arr = np.zeros((masks[0].shape[1], len(masks)))
            for i, mask in enumerate(masks):
                area_arr[:, i] = np.array([np.sum(mask[:, _i, :]) * area_axial_norm
                                           for _i in range(mask.shape[1])])
            area_list = [stat.harmonic_mean(area_arr[_i, :]) for _i in range(masks[0].shape[1])]
        else:
            area_arr = np.zeros((masks[0].shape[0], len(masks)))
            for i, mask in enumerate(masks):
                area_arr[:, i] = np.array([np.sum(mask[_i, :, :]) * area_axial_norm
                                           for _i in range(mask.shape[0])])
            area_list = [stat.harmonic_mean(area_arr[_i, :]) for _i in range(masks[0].shape[0])]
        idx_max_area = area_list.index(max(area_list))
    return idx_max_area


def compare_hist(a: np.ndarray, b: np.ndarray):
    # Kolmogorov-Smirnov test for comparing histograms. It also works for comparing ROC curves.
    return stats.ks_2samp(a, b)


# noinspection SpellCheckingInspection
def delong_roc_test(y: np.ndarray, y_pred_1: np.ndarray, y_pred_2: np.ndarray):
    """
    The fast version of DeLong's method for computing the covariance of unadjusted AUC.
    Reference:
     @article{sun2014fast,
       title={Fast Implementation of DeLong's Algorithm for
              Comparing the Areas Under Correlated Receiver Operating Characteristic Curves},
       author={Xu Sun and Weichao Xu},
       journal={IEEE Signal Processing Letters},
       volume={21},
       number={11},
       pages={1389--1393},
       year={2014},
       publisher={IEEE}
     }

    Computes p-value for hypothesis that two ROC AUCs are different
    Args:
       y: ground truth, an array of 0 and 1,
       y_pred_1: predictions of the first model
       y_pred_2: predictions of the second model
    """

    # noinspection SpellCheckingInspection
    def calc_pvalue(aucs, sigma):
        """Computes log(10) of p-values.
        Args:
           aucs: 1D array of AUCs
           sigma: AUC DeLong covariances
        Returns:
           pvalue
        """
        _l = np.array([[1, -1]])
        z = np.abs(np.diff(aucs)) / np.sqrt(np.dot(np.dot(_l, sigma), _l.T))
        # noinspection PyUnresolvedReferences
        return 10**(np.log10(2) + stats.norm.logsf(z, loc=0, scale=1) / np.log(10))

    def compute_ground_truth_statistics():
        assert np.array_equal(np.unique(y), [0, 1])
        _order = (-y).argsort()
        _n_cls_1 = int(y.sum())
        return _order, _n_cls_1

    order, n_cls_1 = compute_ground_truth_statistics()
    predictions_sorted_transposed = np.vstack((y_pred_1, y_pred_2))[:, order]

    # AUC comparison adapted from
    # https://github.com/Netflix/vmaf/
    def compute_mid_rank(x):
        """Computes mid_ranks.
        Args:
           x - a 1D numpy array
        Returns:
           array of mid_ranks
        """
        J = np.argsort(x)
        Z = x[J]
        N = len(x)
        T = np.zeros(N)
        i = 0
        while i < N:
            j = i
            while j < N and Z[j] == Z[i]:
                j += 1
            T[i:j] = 0.5 * (i + j - 1)
            i = j
        T2 = np.zeros(N)
        # Note "+1" is due to Python using 0-based indexing
        # instead of 1-based in the AUC formula in the paper
        T2[J] = T + 1
        return T2

    # Short variables are named as they are in the paper
    m = n_cls_1
    n = predictions_sorted_transposed.shape[1] - m
    positive_examples = predictions_sorted_transposed[:, :m]
    negative_examples = predictions_sorted_transposed[:, m:]
    k = predictions_sorted_transposed.shape[0]
    tx = np.zeros((k, m))
    ty = np.zeros((k, n))
    tz = np.zeros((k, m + n))
    for r in range(k):
        tx[r, :] = compute_mid_rank(positive_examples[r, :])
        ty[r, :] = compute_mid_rank(negative_examples[r, :])
        tz[r, :] = compute_mid_rank(predictions_sorted_transposed[r, :])
    auc_values = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delong_cov = sx / m + sy / n
    return calc_pvalue(auc_values, delong_cov)


def heatmap(data, row_labels, col_labels, label_fontsize=12, ax=None,
            cbar_kw=None, cbar_prop: Union[dict, None] = None, **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    label_fontsize: int that specifies font size
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current Axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbar_prop: dict that may contain label string and font size information with keys of 'label' and 'fontsize'.
               Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    # Create colorbar
    cbar = plt.colorbar(im, cax=cax, **cbar_kw)
    if cbar_prop is not None:
        cbar.ax.set_ylabel(cbar_prop.get('label'), rotation=-90, va='bottom',
                           fontsize=cbar_prop.get('fontsize', label_fontsize))

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels, fontsize=label_fontsize)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels, fontsize=label_fontsize)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


# noinspection PyUnresolvedReferences
def annotate_heatmap(im, data=None, val_fmt: Union[dict, None] = None,
                     text_colors=('black', 'white'), **text_kw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im:
        The AxesImage to be labeled.
    data:
        Data used to annotate.  If None, the image's data is used.  Optional.
    val_fmt:
        The format of the annotations inside the heatmap.  If None, use the string format method, "$ {x:.2f}", or it is
        provided as a dict.
    text_colors:
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    **text_kw:
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """
    from decimal import Decimal

    # noinspection PyUnresolvedReferences
    def _apply_val_fmt(_number):
        if isinstance(val_fmt, dict):
            if 'math_scale_lb' in val_fmt.keys() and _number < val_fmt['math_scale_lb']:
                return '{:.2E}'.format(Decimal(_number))
            else:
                fmt = mpl.ticker.StrMethodFormatter(val_fmt.get('str', '{x:.2f}'))
        else:
            fmt = mpl.ticker.StrMethodFormatter('{x:.2f}')
        return fmt(_number)

    # Set val_fmt and normalize the threshold to the images color range.
    if val_fmt is None:
        threshold = im.norm(np.max(data)) / 2.
    else:
        if 'txt_color_bound' in val_fmt.keys():
            threshold = val_fmt['txt_color_bound']
            if isinstance(threshold, list):
                threshold = [im.norm(_t) for _t in threshold]
            else:
                threshold = im.norm(threshold)
        else:
            threshold = im.norm(np.max(data)) / 2.

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Set default alignment to center, but allow it to be
    # overwritten by text_kw.
    kw = dict(horizontalalignment='center',
              verticalalignment='center')
    kw.update(text_kw)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if not isinstance(threshold, list):
                kw.update(color=text_colors[int(im.norm(data[i, j]) > threshold)])
            else:
                kw.update(color=text_colors[int(min(threshold) > im.norm(data[i, j]) or
                                                im.norm(data[i, j]) > max(threshold))])
            text = im.axes.text(j, i, _apply_val_fmt(data[i, j]), **kw)
            texts.append(text)
    return texts


def get_auc_ci(auc_val: float, n1: int, n2: int):
    """
    Reference: Hanley, J.A. and NcNeil, B.J. The meaning and use of the area under a receiver operating characteristic
               (ROC) curve. Radiology, vol. 148, 29-36, 1982.
    :param auc_val: a float number;
    :param n1: number of class 1 samples;
    :param n2: number of class 2 samples;
    :return:
    """

    q1 = auc_val/(2-auc_val)
    q2 = 2 * auc_val ** 2 / (1 + auc_val)
    se = np.sqrt((auc_val * (1 - auc_val) + (n1 - 1) * (q1 - auc_val ** 2) + (n2 - 1) * (q2 - auc_val ** 2))/n1/n2)
    return stats.norm.ppf(0.95) * se


def estimate_sample_size_with_mixed_model(sampled_groups: dict, df_n=None, power=0.8, sig_level=0.05, n=None,
                                          icc=0.05):
    """
    Compute sample size for linear mixed models based on power calculations.

    Parameters:
        sampled_groups: sampled groups in dict.
        df_n: Degrees of freedom for numerator (optional).
        power: Desired statistical power (default 0.8).
        sig_level: Significance level (default 0.05).
        n: Number of observations per group (optional).
        icc: Intra-class correlation coefficient (default 0.05).

    Returns:
        Dictionary with subjects per group and total sample size.
    """

    def _design_effect():
        """Compute design effect based on number of observations and ICC.
        """
        return 1 + (n - 1) * icc

    mean1 = sampled_groups['mean1']
    mean2 = sampled_groups['mean2']
    std1 = sampled_groups['std1']
    std2 = sampled_groups['std2']
    n1 = sampled_groups['n1']
    n2 = sampled_groups['n2']
    if isinstance(mean1, np.ndarray):
        n = len(mean1)
        assert len(mean2) == n
        icc = get_icc(np.vstack((mean1, mean2)).T)
        mean1 = np.mean(mean1)
        mean2 = np.mean(mean2)
        std1 = np.sqrt(np.sum(std1**2)/n)
        std2 = np.sqrt(np.sum(std2**2)/n)

    # Cohen's d as the effect size:
    eff_size = abs(mean1 - mean2) / np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2-2))
    # Compute sample size for standard design
    if df_n is None:
        # T-test calculation (approximation using standard formula)
        t_alpha = stats.norm.ppf(1 - sig_level/2)
        t_beta = stats.norm.ppf(power)
    else:
        t_alpha = stats.t.ppf(1 - sig_level / 2, df_n)  # Two-tailed test
        t_beta = stats.t.ppf(power, df_n)  # Power-related t-score

    # Calculate required sample size per group
    obs = ((t_alpha + t_beta) / eff_size) ** 2
    # If no observations per cluster, calculate based on ICC
    if n is None:
        n = (obs * (1 - icc)) / (2 - (obs * icc))
        if n < 1:
            print("Warning: Minimum required number of subjects per cluster is negative.")
            print("Adjusted to 1. Consider reducing ICC or increasing effect size.")
            n = 1
    # Adjust for design effect
    return int(round(obs * _design_effect()))



