import os
import sys
import time
import numpy as np
import pandas as pd
import logging
import warnings
from prettytable import PrettyTable
from cplex.exceptions import CplexError

# Logging
def setup_logging(logger, log_to_console = True, log_file = None):
    """
    Sets up logging to console and file on disk
    See https://docs.python.org/2/howto/logging-cookbook.html for details on how to customize

    Parameters
    ----------
    log_to_console  set to True to disable logging in console
    log_file        path to file for loggin

    Returns
    -------
    Logger object that prints formatted messages to log_file and console
    """

    # quick return if no logging to console or file
    if log_to_console is False and log_file is None:
        logger.disabled = True
        return logger

    log_format = logging.Formatter(fmt='%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%m-%d-%Y %I:%M:%S %p')

    # log to file
    if log_file is not None:
        fh = logging.FileHandler(filename=log_file)
        #fh.setLevel(logging.DEBUG)
        fh.setFormatter(log_format)
        logger.addHandler(fh)

    if log_to_console:
        ch = logging.StreamHandler()
        #ch.setLevel(logging.DEBUG)
        ch.setFormatter(log_format)
        logger.addHandler(ch)

    return logger

def print_log(msg, print_flag = True):
    if print_flag:
        if type(msg) is str:
            print ('%s | ' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()))) + msg
        else:
            print '%s | %r' % (time.strftime("%m/%d/%y @ %I:%M %p", time.localtime()), msg)
        sys.stdout.flush()

# Loading Settings
def get_or_set_default(settings, setting_name, default_value, type_check = False, print_flag = False):

    if setting_name in settings:
        if type_check:
            #check type match
            default_type = type(default_value)
            user_type = type(settings[setting_name])
            if user_type == default_type:
                settings[setting_name] = default_value
            else:
                print_log("type mismatch on %s: user provided type: %s and but expected type: %s" % (setting_name, user_type, default_type), print_flag)
                print_log("setting %s to its default value: %r" % (setting_name, default_value), print_flag)
                settings[setting_name] = default_value
                #else: do nothing
    else:
        print_log("setting %s to its default value: %r" % (setting_name, default_value), print_flag)
        settings[setting_name] = default_value

    return settings

# Loading and Checking Training Data
def check_data(data):
    """
    makes sure that 'data' contains training data that is suitable for binary classification problems
    throws AssertionError if

    'data' is a dictionary that must contain:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)

     data can also contain:

     - 'outcome_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    Returns
    -------
    True if data passes checks

    """
    # type checks
    assert type(data) is dict, "data should be a dict"

    assert 'X' in data, "data should contain X matrix"
    assert type(data['X']) is np.ndarray, "type(X) should be numpy.ndarray"

    assert 'Y' in data, "data should contain Y matrix"
    assert type(data['Y']) is np.ndarray, "type(Y) should be numpy.ndarray"

    assert 'variable_names' in data, "data should contain variable_names"
    assert type(data['variable_names']) is list, "variable_names should be a list"

    X = data['X']
    Y = data['Y']
    variable_names = data['variable_names']

    if 'outcome_name' in data:
        assert type(data['outcome_name']) is str, "outcome_name should be a str"

    # sizes and uniqueness
    N, P = X.shape
    assert N > 0, 'X matrix must have at least 1 row'
    assert P > 0, 'X matrix must have at least 1 column'
    assert len(Y) == N, 'dimension mismatch. Y must contain as many entries as X. Need len(Y) = N.'
    assert len(list(set(data['variable_names']))) == len(data['variable_names']), 'variable_names is not unique'
    assert len(data['variable_names']) == P, 'len(variable_names) should be same as # of cols in X'

    # feature matrix
    assert np.all(~np.isnan(X)), 'X has nan entries'
    assert np.all(~np.isinf(X)), 'X has inf entries'

    # offset in feature matrix
    if '(Intercept)' in variable_names:
        assert all(X[:, variable_names.index('(Intercept)')] == 1.0), "'(Intercept)' column should only be composed of 1s"
    else:
        warnings.warn("there is no column named '(Intercept)' in variable_names")

    # labels values
    assert all((Y == 1) | (Y == -1)), 'Need Y[i] = [-1,1] for all i.'
    if all(Y == 1):
        warnings.warn('Y does not contain any positive examples. Need Y[i] = +1 for at least 1 i.')
    if all(Y == -1):
        warnings.warn('Y does not contain any negative examples. Need Y[i] = -1 for at least 1 i.')

    if 'sample_weights' in data:
        sample_weights = data['sample_weights']
        type(sample_weights) is np.ndarray
        assert len(sample_weights) == N, 'sample_weights should contain N elements'
        assert all(sample_weights > 0), 'sample_weights[i] > 0 for all i '

        # by default, we set sample_weights as an N x 1 array of ones. if not, then sample weights is non-trivial
        if not all(sample_weights == 1):
            if len(np.unique(sample_weights)) < 2:
                warnings.warn('note: sample_weights only has <2 unique values')

    return True

def load_data_from_csv(dataset_csv_file, sample_weights_csv_file = None, fold_csv_file = None, fold_num = 0):
    """

    Parameters
    ----------
    dataset_csv_file                csv file containing the training data
                                    see /datasets/adult_data.csv for an example
                                    training data stored as a table with N+1 rows and d+1 columns
                                    column 1 is the outcome variable entries must be (-1,1) or (0,1)
                                    column 2 to d+1 are the d outcome variables
                                    row 1 contains unique names for the outcome variable, and the input vairable

    sample_weights_csv_file         csv file containing sample weights for the training data
                                    weights stored as a table with N rows and 1 column
                                    all sample weights must be non-negative

    fold_csv_file                   csv file containing indices of folds for K-fold cross validation
                                    fold indices stored as a table with N rows and 1 column
                                    folds must be integers between 1 to K
                                    if fold_csv_file is None, then we do not use folds

    fold_num                        int between 0 to K, where K is set by the fold_csv_file
                                    let fold_idx be the N x 1 index vector listed in fold_csv_file
                                    samples where fold_idx == fold_num will be used to test
                                    samples where fold_idx != fold_num will be used to train the model
                                    fold_num = 0 means use "all" of the training data (since all values of fold_idx \in [1,K])
                                    if fold_csv_file is None, then fold_num is set to 0


    Returns
    -------
    dictionary containing training data for a binary classification problem with the fields:

     - 'X' N x P matrix of features (numpy.ndarray) with a column of 1s for the '(Intercept)'
     - 'Y' N x 1 vector of labels (+1/-1) (numpy.ndarray)
     - 'variable_names' list of strings containing the names of each feature (list)
     - 'Y_name' string containing the name of the output (optional)
     - 'sample_weights' N x 1 vector of sample weights, must all be positive

    """

    if os.path.isfile(dataset_csv_file):
        df = pd.read_csv(dataset_csv_file, sep=',')
    else:
        raise IOError('could not find dataset_csv_file: %s' % dataset_csv_file)

    raw_data = df.as_matrix()
    data_headers = list(df.columns.values)
    N = raw_data.shape[0]

    # setup Y vector and Y_name
    Y_col_idx = [0]
    Y = raw_data[:, Y_col_idx]
    Y_name = data_headers[Y_col_idx[0]]
    Y[Y == 0] = -1

    # setup X and X_names
    X_col_idx = [j for j in range(raw_data.shape[1]) if j not in Y_col_idx]
    X = raw_data[:, X_col_idx]
    variable_names = [data_headers[j] for j in X_col_idx]

    # insert a column of ones to X for the intercept
    X = np.insert(arr=X, obj=0, values=np.ones(N), axis=1)
    variable_names.insert(0, '(Intercept)')

    if sample_weights_csv_file is None:
        sample_weights = np.ones(N)
    else:
        if os.path.isfile(sample_weights_csv_file):
            sample_weights = pd.read_csv(sample_weights_csv_file, sep=',', header=None)
            sample_weights = sample_weights.as_matrix()
        else:
            raise IOError('could not find sample_weights_csv_file: %s' % sample_weights_csv_file)

    data = {
        'X': X,
        'Y': Y,
        'variable_names': variable_names,
        'outcome_name': Y_name,
        'sample_weights': sample_weights,
    }

    #load folds
    if fold_csv_file is not None:
        if not os.path.isfile(fold_csv_file):
            raise IOError('could not find fold_csv_file: %s' % fold_csv_file)
        else:
            fold_idx = pd.read_csv(fold_csv_file, sep=',', header=None)
            fold_idx = fold_idx.values.flatten()
            K = max(fold_idx)
            all_fold_nums = np.sort(np.unique(fold_idx))
            assert len(fold_idx) == N, "dimension mismatch: read %r fold indices (expected N = %r)" % (len(fold_idx), N)
            assert np.all(all_fold_nums == np.arange(1, K+1)), "folds should contain indices between 1 to %r" % K
            assert fold_num in np.arange(0, K+1), "fold_num should either be 0 or an integer between 1 to %r" % K
            if fold_num >= 1:
                test_idx = fold_num == fold_idx
                train_idx = fold_num != fold_idx
                data['X'] = data['X'][train_idx,]
                data['Y'] = data['Y'][train_idx]
                data['sample_weights'] = data['sample_weights'][train_idx]

    assert check_data(data)
    return data

# Check IP Solution
def check_slim_ip_solution(slim_mip, slim_info, data):

    #TODO skip tests if there is no solution
    #TODO return true to prove that it's passed tests
    #TODO (optional) collect warnings and return those?

    #MIP related sanity checks
    assert len(slim_mip.solution.get_values()) == slim_info['n_variables']

    #setup function handles for convenient checking
    get_L0_norm = lambda x: np.sum(np.count_nonzero(x[slim_info['L0_reg_ind']]))

    #key variables
    rho = np.array(slim_mip.solution.get_values(slim_info['rho_idx']))
    alpha = np.array(slim_mip.solution.get_values(slim_info['alpha_idx']))
    beta = np.array(slim_mip.solution.get_values(slim_info['beta_idx']))
    err = np.array(slim_mip.solution.get_values(slim_info['error_idx']))

    #auxiliary variables
    total_error = np.array(slim_mip.solution.get_values(slim_info['total_error_idx']))
    total_error_pos = np.array(slim_mip.solution.get_values(slim_info['total_error_pos_idx']))
    total_error_neg = np.array(slim_mip.solution.get_values(slim_info['total_error_neg_idx']))
    total_l0_norm = np.array(slim_mip.solution.get_values(slim_info['total_l0_norm_idx']))

    # helper parameters
    L0_reg_ind = slim_info['L0_reg_ind']
    L1_reg_ind = slim_info['L1_reg_ind']
    rho_L0_reg = rho[L0_reg_ind]
    rho_L1_reg = rho[L1_reg_ind]

    # test on coefficient vector
    #assert len(rho) == len(coef_constraints), 'rho has the wrong length'
    assert all(rho <= slim_info['rho_ub']), 'rho exceeds upper bounds'
    assert all(rho >= slim_info['rho_lb']), 'rho exceeds lower bounds'

    # tests on L0 indicator variables
    assert all((alpha == 0)|(alpha == 1)), 'alpha should be binary'
    assert all(abs(rho_L0_reg[alpha == 0]) == 0.0), 'alpha = 0 should => that rho == 0'
    assert all(abs(rho_L0_reg[alpha == 1]) > 0.0), 'alpha = 1 should => that rho != 0'

    # tests on L1 helper variables
    assert all(abs(rho_L1_reg) == beta), 'beta != abs(rho)'

    # tests on L1 helper variables
    # beta_ub_reg = np.maximum(abs(slim_info['rho_ub'][L1_reg_ind]), abs(slim_info['rho_lb'][L1_reg_ind]))
    # beta_lb_reg = np.zeros_like(beta_ub_reg)
    # beta_lb_reg = np.maximum(beta_lb_reg, slim_info['rho_lb'][L1_reg_ind])
    # beta_lb_reg = -np.minimum(beta_lb_reg, slim_info['rho_ub'][L1_reg_ind])
    # beta_lb_reg = abs(beta_lb_reg)
    #assert all(beta >= beta_lb_reg), 'beta should be <= beta_ub'
    #assert all(beta <= beta_ub_reg), 'beta should be >= beta_lb'

    # L0-norm bounds
    expected_l0_norm = get_L0_norm(rho)
    assert sum(alpha) == expected_l0_norm, 'alpha should := 1[rho != 0]'
    assert total_l0_norm == expected_l0_norm
    assert sum(alpha) >= slim_info['L0_min']
    assert sum(alpha) <= slim_info['L0_max']
    assert total_l0_norm >= slim_info['L0_min']
    assert total_l0_norm <= slim_info['L0_max']
    assert expected_l0_norm >= slim_info['L0_min']
    assert expected_l0_norm <= slim_info['L0_max']

    # aggregate error measure tests
    expected_scores = (data['Y'] * data['X']).dot(rho)
    expected_err_values = expected_scores <= slim_info['epsilon']
    assert all((err == 0) | (err == 1)), 'err should be binary'
    assert all(err == expected_err_values), 'error vector is not == sign(XY.dot(rho) + epsilon)'
    assert total_error == sum(err), 'total_error should == sum(error(i))'
    assert total_error == total_error_pos + total_error_neg, 'total_error should == total_error_pos + total_error_neg'
    assert total_error_pos == sum(err[slim_info['pos_ind']])
    assert total_error_neg == sum(err[slim_info['neg_ind']])
    assert all(-expected_scores <= slim_info['M']), 'Big M is not big enough'

    # extra sanity check tests
    assert total_error <= min(slim_info['N_pos'], slim_info['N_neg']), 'total_error should be less than total_error_pos + total_error_neg'

# Print Scoring System
def print_slim_model(rho, data, show_omitted_variables = False):

    rho_values = np.copy(rho)
    rho_names = list(data['variable_names'])

    if '(Intercept)' in rho_names:
        intercept_ind = data['variable_names'].index('(Intercept)')
        intercept_val = int(rho[intercept_ind])
        rho_values = np.delete(rho_values, intercept_ind)
        rho_names.remove('(Intercept)')
    else:
        intercept_val = 0

    if data['outcome_name'] is None:
        predict_string = "PREDICT Y = +1 IF SCORE >= %d" % intercept_val
    else:
        predict_string = "PREDICT %s IF SCORE >= %d" % (data['outcome_name'][0].upper(), intercept_val)

    if not show_omitted_variables:
        selected_ind = np.flatnonzero(rho_values)
        rho_values = rho_values[selected_ind]
        rho_names = [rho_names[i] for i in selected_ind]

        #sort by most positive to most negative
        sort_ind = np.argsort(-np.array(rho_values))
        rho_values = [rho_values[j] for j in sort_ind]
        rho_names = [rho_names[j] for j in sort_ind]
        rho_values = np.array(rho_values)

    rho_values_string = [str(int(i)) + " points" for i in rho_values]
    n_variable_rows = len(rho_values)
    total_string = "ADD POINTS FROM ROWS %d to %d" % (1, n_variable_rows)

    max_name_col_length = max(len(predict_string), len(total_string), max([len(s) for s in rho_names])) + 2
    max_value_col_length = max(7, max([len(s) for s in rho_values_string]) + len("points")) + 2

    m = PrettyTable()
    m.field_names = ["Variable", "Points", "Tally"]

    m.add_row([predict_string, "", ""])
    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])

    for v in range(0, n_variable_rows):
        m.add_row([rho_names[v], rho_values_string[v], "+ ....."])

    m.add_row(['=' * max_name_col_length, "=" * max_value_col_length, "========="])
    m.add_row([total_string, "SCORE", "= ....."])
    m.header = False
    m.align["Variable"] = "l"
    m.align["Points"] = "r"
    m.align["Tally"] = "r"

    return m

def get_model_summary(rho, slim_info, data):

    #build a pretty table model
    printed_model = print_slim_model(rho, data, show_omitted_variables = False)

    #transform Y
    y = np.array(data['Y'].flatten(), dtype = np.float)
    pos_ind = y == 1
    neg_ind = ~pos_ind
    N = len(data['Y'])
    N_pos = np.sum(pos_ind)
    N_neg = N - N_pos

    #get predictions
    yhat = data['X'].dot(rho) > 0
    yhat = np.array(yhat, dtype = np.float)
    yhat[yhat == 0] = -1

    true_positives = np.sum(yhat[pos_ind] == 1)
    false_positives = np.sum(yhat[neg_ind] == 1)
    true_negatives= np.sum(yhat[neg_ind] == -1)
    false_negatives = np.sum(yhat[pos_ind] == -1)

    rho_summary = {
        'rho': rho,
        'pretty_model': printed_model,
        'mistakes': np.sum(y != yhat),
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'error_rate': (false_positives + false_negatives) / N,
        'true_positive_rate': true_positives / N_pos,
        'false_positive_rate': false_positives / N_neg,
        'L0_norm': np.sum(rho[slim_info['L0_reg_ind']]),
    }

    return rho_summary

def get_slim_summary(slim_mip, slim_info, data):

    #MIP Related Items
    slim_summary = {
        #
        # IP related information
        'solution_status_code': slim_mip.solution.get_status(),
        'solution_status': slim_mip.solution.get_status_string(slim_mip.solution.get_status()),
        'objective_value': slim_mip.solution.get_objective_value(),
        'optimality_gap': slim_mip.solution.MIP.get_best_objective(),
        'objval_lowerbound': slim_mip.solution.MIP.get_mip_relative_gap(),
        'simplex_iterations': slim_mip.solution.progress.get_num_iterations(),
        'nodes_processed': slim_mip.solution.progress.get_num_nodes_processed(),
        'nodes_remaining': slim_mip.solution.progress.get_num_nodes_remaining(),
        #
        # Solution based information (default values)
        'rho': float('nan'),
        'pretty_model': '',
        'string_model': '',
        'true_positives': float('nan'),
        'true_negatives': float('nan'),
        'false_positives': float('nan'),
        'false_negatives': float('nan'),
        'mistakes': float('nan'),
        'error_rate': float('nan'),
        'true_positive_rate': float('nan'),
        'false_positive_rate': float('nan'),
        'L0_norm': float('nan'),
    }

    #Update with Solution-Based Stats
    try:
        rho = np.array(slim_mip.solution.get_values(slim_info['rho_idx']))
        slim_summary.update(get_model_summary(rho, slim_info, data))
    except CplexError as e:
        print e

    return slim_summary

def get_accuracy_stats(model, data, error_checking = True):

    accuracy_stats = {
        'train_true_positives': float('nan'),
        'train_true_negatives':  float('nan'),
        'train_false_positives':  float('nan'),
        'train_false_negatives':  float('nan'),
        'valid_true_positives': float('nan'),
        'valid_true_negatives': float('nan'),
        'valid_false_positives': float('nan'),
        'valid_false_negatives': float('nan'),
        'test_true_positives': float('nan'),
        'test_true_negatives': float('nan'),
        'test_false_positives': float('nan'),
        'test_false_negatives': float('nan'),
    }

    def get_prediction(x, rho): return np.sign(x.dot(rho))
    def get_true_positives_from_pred(yhat, pos_ind): return np.sum(yhat[pos_ind] == 1)
    def get_false_positives_from_pred(yhat, pos_ind): return np.sum(yhat[~pos_ind] == 1)
    def get_true_negatives_from_pred(yhat, pos_ind): return np.sum(yhat[~pos_ind] != 1)
    def get_false_negatives_from_pred(yhat, pos_ind): return np.sum(yhat[pos_ind] != 1)

    model = np.array(model).reshape(data['X'].shape[1], 1)

    # training set
    data_prefix = 'train'
    X_field_name = 'X'
    Y_field_name = 'Y'
    Yhat = get_prediction(data['X'], model)
    pos_ind = data[Y_field_name] == 1

    accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
    accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

    if error_checking:
        N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                   accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                   accuracy_stats[data_prefix + '_' + 'false_positives'] +
                   accuracy_stats[data_prefix + '_' + 'false_negatives'])
        assert data[X_field_name].shape[0] == N_check

    # validation set
    data_prefix = 'valid'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_validation_set = (X_field_name in data and
                          Y_field_name in data and
                          data[X_field_name].shape[0] > 0 and
                          data[Y_field_name].shape[0] > 0)

    if has_validation_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    # test set
    data_prefix = 'test'
    X_field_name = 'X' + '_' + data_prefix
    Y_field_name = 'Y' + '_' + data_prefix
    has_test_set = (X_field_name in data and
                    Y_field_name in data and
                    data[X_field_name].shape[0] > 0 and
                    data[Y_field_name].shape[0] > 0)

    if has_test_set:

        Yhat = get_prediction(data[X_field_name], model)
        pos_ind = data[Y_field_name] == 1
        accuracy_stats[data_prefix + '_' + 'true_positives'] = get_true_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'true_negatives'] = get_true_negatives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_positives'] = get_false_positives_from_pred(Yhat, pos_ind)
        accuracy_stats[data_prefix + '_' + 'false_negatives'] = get_false_negatives_from_pred(Yhat, pos_ind)

        if error_checking:
            N_check = (accuracy_stats[data_prefix + '_' + 'true_positives'] +
                       accuracy_stats[data_prefix + '_' + 'true_negatives'] +
                       accuracy_stats[data_prefix + '_' + 'false_positives'] +
                       accuracy_stats[data_prefix + '_' + 'false_negatives'])
            assert data[X_field_name].shape[0] == N_check

    return accuracy_stats
