import numpy as np
import time

# local modules:
import load_mat_params as lm
import localmin
import solve_large



UPPER = 0
LOWER = 1
MIN_ALPHAG = None
RTOL = 1e-05

params_folder="../params_for_exps/" #set to wherever the classifier output files are saved


def table_latex_str(x):
    """
    :param x:
    :return: string of latex table; all but last column should be percents
    """
    minstart = r"\textbf{"
    minend = r"}"
    return '\\\\\n'.join(f'${100 * row[0]:.2f}\%$ & ${100 * row[1]:.2f}\%$ &' +
                         ' & '.join(
                             f'${minstart if cell == min(row[2:-1]) else ""}{100 * cell:.2f}\%{minend if cell == min(row[2:-1]) else ""}$'
                             for cell in row[2:-1])
                         + f' & ${row[-1]:.2f}$'
                         for row in x)


def is_neg_zero(x):
    return (x == 0) & (np.copysign(1, x) < 0)


def calc_eta(_a, _b):
    a = _a.copy()
    b = _b.copy()
    assert (a.__class__ == np.zeros(1).__class__)
    assert (b.__class__ == np.zeros(1).__class__)
    a[np.isclose(a, 1, rtol=RTOL)] = 1
    a[np.isclose(a, 0)] = 0
    assert (a <= 1).all()
    assert (a >= 0).all()

    b[np.isclose(b, 1, rtol=RTOL)] = 1
    b[np.isclose(b, 0)] = 0

    assert (b <= 1).all()
    assert (b >= 0).all()

    assert not (is_neg_zero(a).any())
    assert not (is_neg_zero(b).any())

    old_settings = np.seterr(divide='ignore', invalid='ignore')
    try:
        result = np.array(np.maximum(1 - b / a, 1 - (1 - b) / (1 - a)))
        result[a == b] = 0
    except ValueError as e:
        print("Oops! must have wrong shapes of a and b!")
        print(e)
        raise
    finally:
        np.seterr(**old_settings)

    assert ((0 <= result).all() & (result <= 1).all())
    return result


def calc_eta_nums(a, b):
    a = np.array(a)
    b = np.array(b)
    return calc_eta(a, b)


def calc_obj_vals(wg, pig, alphag, alphayz_options, sum_axis=-2):
    """

    :param wg: (numg,1)
    :param alphayz_options: broadcastable to (..., 1, numy, numz)
    :param alphag:  (numg, numy, numz)
    :param pig: (numg, numy)
    :param sum_axis: put -2 to sum over g and get answer per y, put -1 to sum over y and get answer per g
    :return: etavals (.., numg, numy), obj_vals (..., numy, )

    """
    etavals = np.max(calc_eta(alphayz_options, alphag), axis=-1)
    obj_vals = np.sum(wg * pig * etavals, axis=sum_axis)  # sum over the g axis
    return {'etavals': etavals, 'obj_vals': obj_vals}


def calc_inflections(b1, b2, v, gamma):
    """

    :param b1: alpha_g^yy_i for all g
    :param b2: tilde{alpha}_g^{yy_{i+1}} for all g
    :param v: v_g for all g
    :param gamma: a scalar
    :return:
    """

    inflections = np.concatenate(
        (b1 / (1 - v), 1 - (1 - b1) / (1 - v), gamma - b2 / (1 - v), gamma - 1 + (1 - b2) / (1 - v),
         gamma * b1 / (b1 + b2), (b1 * (1 - gamma) - b2 + gamma) / (2 - b1 - b2),
         b1 * (1 - gamma) / (1 - b1 - b2), (gamma * (1 - b1) - b2) / (1 - b1 - b2)))
    # remove illegal inflections that can occur due to edge cases (this is instead of checking each condition):
    inflections = inflections[~np.isnan(inflections)]
    assert not (np.isnan(inflections).any())
    return inflections


class EtaObj:
    def __init__(self, wg, pigy, alphag=None, alphayz=None, pg=None):
        """Initialize with all the parameters known

        wg: ndarray(numg, 1)
        pigy: ndarray(numg, numy)
        alphayz: ndarray(1,numy,numy)
        alphahg: ndarray(numg,numy,numy)
        """
        numg = wg.shape[0]
        numy = pigy.shape[1]
        assert wg.shape == (numg, 1), "shape = {}".format(wg.shape)
        assert np.isclose(np.sum(wg), 1, rtol=RTOL)
        assert pigy.shape == (numg, numy), "shape = {}".format(pigy.shape)
        assert (pg is None) or (pg.shape == (numg, numy), "shape = {}".format(pigy.shape))
        assert np.isclose(np.sum(pigy, axis=1), 1, rtol=RTOL).all()
        assert (alphag is None) or (alphag.shape == (numg, numy, numy)), "shape = {}".format(alphag.shape)

        self.numy = numy
        self.numg = numg
        self.wg = wg
        self.pigy = pigy
        self.alphag = alphag
        self.pg = pg

        if alphag is not None:
            assert np.isclose(np.sum(alphag, axis=2), 1, rtol=RTOL).all()
            self.alphag = np.maximum(self.alphag, MIN_ALPHAG)
            self.alphag = np.minimum(self.alphag, 1-MIN_ALPHAG)
            #normalize
            self.alphag = self.alphag / np.sum(self.alphag, axis=2)[:, :, np.newaxis]
        if self.pg is not None:
            self.pg = self.pg + MIN_P
            self.pg = np.diag(1.0 / np.sum(self.pg, 1)) @ self.pg


        if alphayz is not None:
            assert alphayz.shape == (1, numy, numy), "shape = {}".format(alphayz.shape)
            assert np.isclose(np.sum(alphayz, axis=2), 1).all()
            self.alphayz = alphayz
            results = self.calc_objs(None, None, alphayz)
            self.etavals = results['etavals']
            self.obj_values = results['obj_vals']
            self.unfairness_val = sum(self.obj_values)

    def calc_error(self):
        alphag_nodiag = self.alphag.copy()
        for g in range(self.alphag.shape[0]):
            np.fill_diagonal(alphag_nodiag[g, :, :], 0)

        return np.sum(self.wg * self.pigy * np.sum(alphag_nodiag, 2))

    def calc_objs(self, y, z, alphayz_options):
        """ Calculate the eta values and objective values for each option of alphayz.

        :param alphayz_options: an array of size  (1, numy, numy)
            or of size (num_options_y, 1, 1, numy)
            of of size (num_options, 1, 1, 1)
        :param y: the index of the y that is currently handled (or None if all)
        :param z: the index of the z that is currently handled (or None if y is None)
        :return: etavals, obj_vals which are either lists or arrays based on the input
        """
        dimsalpha = alphayz_options.ndim
        assert (alphayz_options >= 0).all()
        assert (dimsalpha >= 3) and (dimsalpha <= 4)
        if dimsalpha == 3:
            assert y is None
            assert z is None
            assert alphayz_options.shape == (1, self.numy, self.numy), "shape = {}".format(alphayz_options.shape)
            assert np.isclose(np.sum(alphayz_options, axis=2), 1, rtol=RTOL).all()
        else:  # dimsalpha == 4
            assert y is not None
            if z is None:
                assert alphayz_options.shape[1:4] == (1, 1, self.numy), "shape = {}".format(alphayz_options.shape)
                assert np.isclose(np.sum(alphayz_options, axis=3), 1, rtol=RTOL).all()
            else:
                assert alphayz_options.shape[1:4] == (1, 1, 1), "shape = {}".format(alphayz_options.shape)

        if z is None:
            if y is None:
                alphag = self.alphag
            else:
                alphag = self.alphag[:, y:(y + 1), :]
        else:
            alphag = self.alphag[:, y:(y + 1), z:(z + 1)]

        pig = self.pigy if y is None else self.pigy[:, y:(y + 1)]

        results = calc_obj_vals(self.wg, pig, alphag, alphayz_options)

        return results

    def lower_bound(self):
        """" Assume the values of alphayz are unknown

        Output a lower bound based on the single-label minimization formula
        For Binary lables, the lower bound is equal to the unfairness.

        :return An array with the lower bound for each y in the sum
        """
        numy_to_test = self.numy if self.numy > 2 else 1
        obj_vals_grid = np.ndarray((self.numy, numy_to_test))

        obj_vals_search = np.ndarray((self.numy, numy_to_test))
        minimizer_search = np.ndarray((self.numy, numy_to_test))
        for i in range(self.numy):
            for j in range(numy_to_test):
                alpha_options = np.append(self.alphag[:, i, j], [0, 1])
                alpha_options = alpha_options[:, np.newaxis, np.newaxis, np.newaxis]
                obj_vals = self.calc_objs(i, j, alpha_options)['obj_vals']
                minimizer_index = np.argmin(obj_vals)
                obj_vals_search[i, j] = obj_vals[minimizer_index]
                minimizer_search[i, j] = alpha_options[minimizer_index, 0, 0, 0]
        obj_maxes_search = np.max(obj_vals_search, axis=1)


        return {'obj_vals': obj_maxes_search, 'alphas': minimizer_search}


    def upper_bound_mean(self):
        """
        A simple assignment using thea mean alphag values for alpha
        :return: an upper bound for the value of the objective for each y
        """
        alpha_assignment = np.average(self.alphag, 0, np.broadcast_to((self.wg * self.pigy)[:, :, np.newaxis],
                                                                      [self.numg, self.numy,
                                                                       self.numy]))  # a weighted average. The weight of alpha_g^yz is w_g*pi_g^y.
        obj_vals = calc_obj_vals(self.wg, self.pigy, self.alphag, alpha_assignment[np.newaxis, :, :])['obj_vals']
        return {'obj_vals': obj_vals, 'alphas': alpha_assignment}

    def upper_bound_greedy(self, num_attempts=3):
        """
        Use greedy algorithm to get a small upper bound

        :return: a smallest-as-possible upper bound for the value of the objective for each y
        """

        assert self.numy > 2

        # calculate the base case using the equivalent problem in which only two labels are active
        # this is slightly computationally wasteful but more convenient in terms of code

        alphag_diag = np.diagonal(self.alphag, axis1=1, axis2=2)
        alphag_binary = np.dstack((alphag_diag, 1 - alphag_diag,) + (np.zeros_like(alphag_diag),) * (self.numy - 2))
        binary_base = EtaObj(self.wg, self.pigy, alphag_binary)
        results = binary_base.lower_bound()
        minimizers = results['alphas']
        alpha_diag = minimizers[:, 0]  # (numy,)
        alpha_assignment = np.empty((self.numy, self.numy))
        alpha_assignment[:] = np.nan
        v_of_known_ally = calc_eta(alpha_diag[np.newaxis, :], alphag_diag)
        attempt_results_objs = np.empty((num_attempts, self.numy))
        attempt_results_objs[:] = np.nan
        attempt_results_alphas = np.empty((num_attempts, self.numy, self.numy))
        attempt_results_alphas[:] = np.nan
        for attempt in range(num_attempts):
            for y in range(self.numy):
                ys_in_order = np.concatenate(
                    ([y], np.arange(y), np.arange(self.numy - y - 1) + y + 1))  # move the y to the start
                permutation = np.random.permutation(self.numy - 1)
                ys_in_order[1:] = ys_in_order[permutation + 1]
                alpha_assignment[y, y] = alpha_diag[y]
                gamma = 1 - alpha_assignment[y, y]
                v_of_known = v_of_known_ally[:, y]
                for (i_, z) in enumerate(ys_in_order[1:-1]):
                    i = i_ + 1  # the index in ys_in_order
                    alphag_rest = np.sum(self.alphag[:, y, ys_in_order[(i + 1):]], axis=1)  # (numg,)
                    alphag_for_z = self.alphag[:, y, z]
                    inflection_points = calc_inflections(alphag_for_z, alphag_rest, v_of_known, gamma)  # the set M_i^y
                    alpha_options = np.concatenate((inflection_points, alphag_for_z, alphag_rest, [0, gamma]))
                    alpha_options = np.round(alpha_options, ROUND_NUM_DECIMALS)
                    alpha_options[alpha_options == 0] = 0  # this gets rid of negative zeros
                    alpha_options = np.unique(alpha_options)
                    alpha_options = alpha_options[(0 <= alpha_options) & (alpha_options <= gamma)]
                    alpha_options = alpha_options[:, np.newaxis]

                    etavals = np.max(np.stack((np.tile(v_of_known[np.newaxis, :], alpha_options.shape),
                                               calc_eta(alpha_options, alphag_for_z),
                                               calc_eta(gamma - alpha_options, alphag_rest))), axis=0)
                    obj_vals = np.sum(self.wg[:, 0] * self.pigy[:, y] * etavals, axis=-1)
                    minimizer = np.argmin(obj_vals, axis=0)
                    alpha_for_z = alpha_options[minimizer, 0]


                    gamma = gamma - alpha_for_z
                    v_of_known = np.maximum(v_of_known, calc_eta(np.array([alpha_for_z]), alphag_for_z))
                    alpha_assignment[y, z] = alpha_for_z
                alpha_assignment[y, ys_in_order[-1]] = gamma

            obj_vals = calc_obj_vals(self.wg, self.pigy, self.alphag, alpha_assignment[np.newaxis, :, :])
            attempt_results_objs[attempt, :] = obj_vals['obj_vals']
            attempt_results_alphas[attempt, :, :] = alpha_assignment
        best_attempt_indices = np.argmin(attempt_results_objs, axis=0)
        best_attempt_objs = np.take_along_axis(attempt_results_objs, best_attempt_indices[np.newaxis, :], axis=0)
        best_attempt_alphas = np.take_along_axis(attempt_results_alphas,
                                                 best_attempt_indices[np.newaxis, :, np.newaxis], axis=0)

        assignment = best_attempt_alphas[0, :, :]
        objectives = best_attempt_objs[0, :]

        return {'obj_vals': best_attempt_objs[0, :], 'alphas': best_attempt_alphas[0, :, :]}

    def find_local_minimum(self, alpha0):
        """ :param alpha0: an initial solution, (numy, numy) """

        solution = np.ndarray([self.numy, self.numy])
        obj = np.ndarray([self.numy])

        w_all = self.wg * self.pigy  # result is (numg, numy)
        for y in range(self.numy):
            alphas, objs = localmin.solveFairness(alpha0[y, :], np.squeeze(self.alphag[:, y, :]), w_all[:, y])
            # the objectives are calculated with a variant of the objective function so we can't use them
            # calculate the objective on my own
            true_obj_before = calc_obj_vals(self.wg, self.pigy, self.alphag, alpha0, sum_axis=-2)['obj_vals'][y]
            true_obj_after = calc_obj_vals(self.wg, self.pigy, self.alphag, alphas, sum_axis=-2)['obj_vals'][y]
            if true_obj_before < true_obj_after:
                obj[y] = true_obj_before
                solution[y, :] = alpha0[y, :]
            else:
                obj[y] = true_obj_after
                solution[y, :] = alphas

        return {'alphas': solution, 'obj_vals': obj}

    def upperbound_mindisc_unknown_alphag(self, beta):
        large_prob = solve_large.LargeProbParam(self.numg, self.numy, self.wg.reshape((self.numg,)), self.pigy, self.pg, beta)
        alpha, alphag = solve_large.getInitialGuessFairnessProblem(self.numg, self.numy)
        alpha_t, alphag_t, fobj = solve_large.solveFairness(large_prob, alpha, alphag, tol=1e-4,
                                                       alg='highs-ipm', verbose=False)

        return {'alphas': alpha_t, 'obj_val': fobj[-1]}




def generate_tests(numg, numy, num_tests=1):
    wg = np.random.random_sample((num_tests, numg, 1))
    wg = wg / np.sum(wg, axis=1)[:, np.newaxis, :]
    pigy = np.random.random_sample((num_tests, numg, numy))
    pigy = pigy / np.sum(pigy, axis=2)[:, :, np.newaxis]

    alpha_observed = np.random.random_sample((num_tests, numg, numy, numy))
    alpha_observed = alpha_observed / np.sum(alpha_observed, axis=3)[:, :, :, np.newaxis]

    return {'wg': wg, 'pigy': pigy, 'alphag': alpha_observed}


def run_one_test_type(tests, num_attempts, upper_bound_types=['greedy'], test_groups=None, dolocalmin=False):
    ### same numy for all tests
    num_tests = len(tests['wg'])
    numy = tests['pigy'][0].shape[1]
    num_upperbounds = len(upper_bound_types) * (2 if dolocalmin else 1)

    if test_groups is None:
        test_groups = [[i for i in range(num_upperbounds)]]
    num_groups = len(test_groups)

    uppery = np.ndarray((num_upperbounds, num_tests, numy))
    uppery_alphas = np.ndarray((num_upperbounds, num_tests, numy, numy))
    lowery = np.ndarray((num_tests, numy))
    error = np.ndarray(num_tests)
    lowery_alphas = np.ndarray((num_tests, numy, numy))
    minimizers = np.ndarray((num_groups, num_tests, numy), dtype='int')


    for test in range(num_tests):
        ee = EtaObj(tests['wg'][test], tests['pigy'][test], tests['alphag'][test])
        error[test] = ee.calc_error()
        lower_bound_results = ee.lower_bound()
        lowery[test, :] = lower_bound_results['obj_vals']
        lowery_alphas[test, :] = lower_bound_results['alphas']
        for upcount, upper_bound_type in enumerate(upper_bound_types):

            if upper_bound_type == 'greedy':
                upper_bound_results = ee.upper_bound_greedy(num_attempts)
            else:  # only other option is 'mean' for now
                upper_bound_results = ee.upper_bound_mean()

            uppery[upcount, test, :] = upper_bound_results['obj_vals']
            uppery_alphas[upcount, test, :] = upper_bound_results['alphas']
            if dolocalmin:
                locres = ee.find_local_minimum(uppery_alphas[upcount, test, :])
                uppery_alphas[upcount + len(upper_bound_types), test, :] = locres['alphas']
                uppery[upcount + len(upper_bound_types), test, :] = locres['obj_vals']



    uppery_groups = np.take_along_axis(uppery, minimizers, 0)
    uppery_alphas_groups = np.take_along_axis(uppery_alphas, minimizers[..., np.newaxis], 0)

    return {'bound': [np.concatenate((uppery, uppery_groups)), lowery],
            'alphas': [np.concatenate((uppery_alphas, uppery_alphas_groups)), lowery_alphas],
            'group_first_index': uppery.shape[0], 'group_minimizers': minimizers, 'error': error}


def run_one_test_type_unknown_alphag(tests, beta, upper_bound_types=['randominit'], test_groups=None):
    ### same numy for all tests

    num_tests = len(tests['wg'])
    numy = tests['pigy'][0].shape[1]
    num_upperbounds = len(upper_bound_types)

    if test_groups is None:
        test_groups = [[i for i in range(num_upperbounds)]]
    num_groups = len(test_groups)

    uppery = np.ndarray((num_upperbounds, num_tests))
    uppery_alphas = np.ndarray((num_upperbounds, num_tests, numy, numy))
    lowery = np.ndarray((num_tests,))
    lowery_alphas = np.ndarray((num_tests, numy, numy))
    minimizers = np.ndarray((num_groups, num_tests), dtype='int')


    for test in range(num_tests):
        ee = EtaObj(tests['wg'][test], tests['pigy'][test], pg=tests['pg'][test])

        for upcount, upper_bound_type in enumerate(upper_bound_types):

            if upper_bound_type == 'randominit':
                upper_bound_results = ee.upperbound_mindisc_unknown_alphag(beta)
            else:
                print('type not supported')
                exit(1)
                return
            uppery[upcount, test] = upper_bound_results['obj_val']
            uppery_alphas[upcount, test] = upper_bound_results['alphas']


    uppery_groups = np.take_along_axis(uppery, minimizers, 0)
    uppery_alphas_groups = np.take_along_axis(uppery_alphas, minimizers[..., np.newaxis, np.newaxis], 0)

    return {'bound': [np.concatenate((uppery, uppery_groups)), lowery],
            'alphas': [np.concatenate((uppery_alphas, uppery_alphas_groups)), lowery_alphas],
            'group_first_index': uppery.shape[0], 'group_minimizers': minimizers}




def run_mat_tests_unknown_alphag(mat_tests, beta):
    upper_bound_types = ['randominit']



    results = run_one_test_type_unknown_alphag(mat_tests, beta, upper_bound_types=upper_bound_types, test_groups=[])
    upper = results['bound'][UPPER]
    #lower = results['bound'][LOWER]
    #ratios = upper / lower

    print("===================== Unknown alphag results =================")
    print("upper:", upper)



def run_mat_tests_known_alphag(mat_tests):
    upper_bound_types = ['greedy', 'mean']
    mynum_attempts = 10

    results = run_one_test_type(mat_tests, mynum_attempts, upper_bound_types=upper_bound_types, dolocalmin=True,
                                test_groups=[])
    upper = results['bound'][UPPER]
    lower = results['bound'][LOWER]
    error = results['error']
    for test_with_zero in np.where(error == 0)[0]:
        print('######################## Test ', test_with_zero, ' has zero error, it should be ignored')
    ratios = np.sum(upper, 2) / np.sum(lower, 1)

    alg_upper_fortable = np.sum(upper, 2).T
    alg_upper_fortable = alg_upper_fortable[:,
                         [1, 0, 3, 2]]  # permute for presentation: average, greedy, average+minimize, greedy+minimize
    table_res = np.concatenate((error.reshape(-1, 1), np.sum(lower, 1).reshape(-1, 1), alg_upper_fortable,
                                np.min(ratios, axis=0).reshape(-1, 1)), axis=1)
    print('============ Formatted for latex: =============')

    print(table_latex_str(table_res))


def run_tests_census():
    classifier_types = ['tree', 'nn']
    for ctype in classifier_types:
        print('Now in classifier type', ctype)
        mat_tests = lm.load_mat_params(params_folder+'census_multiclass_probs_' + ctype + '.mat')
        num_labels_in_tests = [mat_tests['pigy'][i].shape[1] for i in range(mat_tests.shape[0])]
        num_labels_unique, inverse = np.unique(num_labels_in_tests, return_inverse=True)
        for num_labels_index in range(len(num_labels_unique)):
           print('=========== Now running census tests with ', num_labels_index, ' round of num labels, with ',
                  num_labels_unique[num_labels_index], ' labels')
           run_mat_tests_known_alphag(mat_tests[np.where(inverse == num_labels_index)])
        print('This was classifier type', ctype)

def run_tests_census_unknown_alphag():
    classifier_types = ['tree', 'nn']
    beta=1
    for ctype in classifier_types:
        print('Now in classifier type', ctype)
        mat_tests = lm.load_mat_params(params_folder+'census_multiclass_probs_' + ctype + '.mat')
        num_labels_in_tests = [mat_tests['pigy'][i].shape[1] for i in range(mat_tests.shape[0])]
        num_labels_unique, inverse = np.unique(num_labels_in_tests, return_inverse=True)
        for num_labels_index in range(len(num_labels_unique)):
            print('=========== Now running census tests with ', num_labels_index, ' round of num labels, with ',
                   num_labels_unique[num_labels_index], ' labels', flush=True)
            run_mat_tests_unknown_alphag(mat_tests[np.where(inverse == num_labels_index)], beta=beta)
        print('This was classifier type', ctype)


def run_tests_labor_unknown_alphag():
    classifier_types = ['decision_tree', 'naive_bayes', 'knn']

    beta = 1
    for ctype in classifier_types:
        print('======================================= Now running labor classifier, unknown alpha', ctype, '=================')
        mat_tests = lm.load_mat_params(params_folder+'labor_' + ctype + '_params.mat')
        print('beta = ', beta)

        run_mat_tests_unknown_alphag(mat_tests, beta=beta)

def run_tests_unknown_alphag(filename):
    beta = 1
    print('======================================= Now running tests from file ',filename, ', unknown alpha =================')
    mat_tests = lm.load_mat_params(filename)
    print('beta = ', beta)
    run_mat_tests_unknown_alphag(mat_tests, beta=beta)


def run_tests_labor():
    classifier_types = ['decision_tree', 'naive_bayes', 'knn']
    for ctype in classifier_types:
        print('======================================= Now running labor classifier', ctype, '=================')
        mat_tests = lm.load_mat_params(params_folder+'labor_' + ctype + '_params.mat')
        run_mat_tests_known_alphag(mat_tests)

        print('order of problems (type of sub-populations): attendant, father race, mother race, payer')
        print('======================================= That was labor classifier', ctype, '=================')



if __name__ == "__main__":
    ROUND_NUM_DECIMALS = 6  # should be less if we have very small numbers
    np.set_printoptions(linewidth=150, precision=5, suppress=True)
    epoch_time = int(time.time())
    np.random.seed(epoch_time)


    MIN_ALPHAG = 1e-5
    MIN_P = 11e-6


    run_tests_labor()
    run_tests_labor_unknown_alphag()
    run_tests_census_unknown_alphag()
    run_tests_unknown_alphag('/ukelections/uk_inputs.mat')
    run_tests_unknown_alphag('/education/education_inputs.mat')
    run_tests_census()


