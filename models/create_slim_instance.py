import os
import sys
import numpy as np
import argparse
import logging
import pickle

#add '/models/' directory to search path to avoid import errors
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import slim as slim

# parse command line arguments
def setup_parser():
    """
    Create an argparse Parser object for command line arguments to create_slim_instance.
    This object determines all command line arguments, handles input
    validation and default values.

    See https://docs.python.org/3/library/argparse.html for configuration
    """

    def is_positive_integer(value):
        parsed_value = int(value)
        if parsed_value <= 0:
            raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
        return parsed_value

    def is_positive_float_or_negative_one(value):
        parsed_value = float(value)
        if not (parsed_value == -1 or parsed_value > 0.0):
            raise argparse.ArgumentTypeError("%s is an invalid value (must be -1 or > 0.00)" % value)
        return parsed_value

    def is_positive_integer_or_negative_one(value):
        parsed_value = int(value)
        if not (parsed_value == -1 or parsed_value >= 1):
            raise argparse.ArgumentTypeError("%s is an invalid value (must be -1 or >=1)" % value)
        else:
            return parsed_value

    def is_file_on_disk(file_name):
        if not os.path.isfile(file_name):
            raise argparse.ArgumentTypeError("the file %s does not exist!" % file_name)
        else:
            return file_name

    def file_choices(choices, file_name):
        ext = os.path.splitext(file_name)[1][1:]
        if ext not in choices:
            parser.error("file doesn't end with one of {}".format(choices))
        return file_name

    def is_file_of_type_on_disk(choices, file_name):
        return is_file_on_disk(file_choices(choices, file_name))


    parser = argparse.ArgumentParser(
        prog='create_slim_instance',
        description='Create a SLIM IP instance and save it as a MPS file from the command shell',
        epilog='Copyright (C) 2017 Berk Ustun',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--data_file',
                        type=lambda s: is_file_of_type_on_disk("csv", s),
                        required=True,
                        help='csv file with training data')

    parser.add_argument('--instance_file',
                        type=lambda s: file_choices("mps", s),
                        required=True,
                        help='name of instance file (must end in .mps)')

    parser.add_argument('--instance_info',
                        type=lambda s: file_choices("p", s),
                        help='name of instance information file (must be .p)')

    parser.add_argument('--max_size',
                        type = is_positive_integer_or_negative_one,
                        default=-1,
                        help='maximum number of non-zero coefficients; set as -1 for no limit')

    parser.add_argument('--max_coef',
                        type=is_positive_integer,
                        default=10,
                        help='value of upper and lower bounds for any coefficient')

    parser.add_argument('--max_offset',
                        type=is_positive_integer_or_negative_one,
                        default=-1,
                        help='value of upper and lower bound on offset parameter; set as -1 to use a conservative value')

    parser.add_argument('--c0_value',
                        type=is_positive_float_or_negative_one,
                        default=-1,
                        help='l0 regularization parameter; set as a positive float > 0.00; or -1 for smallest value')

    parser.add_argument('--log',
                        type=str,
                        help='name of the log file')

    parser.add_argument('--silent',
                        action='store_true',
                        help='flag to suppress logging to stderr')

    return parser

# create instance
def create_slim_instance(data_file, max_coef=10, c0_value=-1, max_size =-1, max_offset=-1, logger = None):

    # load dataset from csv
    data = slim.load_data_from_csv(data_file)
    N, P = data['X'].shape

    # set data-dependent parameters
    max_size = P if max_size == -1 else min(max_size, P)
    c0_value = 0.9 / (N * P) if c0_value == -1 else min(c0_value, 1.00)

    # setup coefficient constraints
    coef_constraints = slim.CoefficientSet(variable_names=data['variable_names'], ub=max_coef, lb=-max_coef)

    if max_offset == -1:
        # choose upper and lower bounds for the intercept coefficient
        # to ensure that there will be no regularization due to the intercept, choose
        #
        # intercept_ub < min_i(min_score_i)
        # intercept_lb > max_i(max_score_i)
        #
        # where min_score_i = min((Y*X) * \rho) for rho in \Lset
        # where max_score_i = max((Y*X) * \rho) for rho in \Lset
        #
        # setting intercept_ub and intercept_lb in this way ensures that we can classify every point as positive and negative
        scores_at_ub = (data['X'] * data['Y']) * coef_constraints.ub
        scores_at_lb = (data['X'] * data['Y']) * coef_constraints.lb
        non_intercept_ind = np.array([n != '(Intercept)' for n in data['variable_names']])
        scores_at_ub = scores_at_ub[:, non_intercept_ind]
        scores_at_lb = scores_at_lb[:, non_intercept_ind]
        max_scores = np.fmax(scores_at_ub, scores_at_lb)
        min_scores = np.fmin(scores_at_ub, scores_at_lb)
        max_scores = np.sum(max_scores, 1)
        min_scores = np.sum(min_scores, 1)
        intercept_ub = -min(min_scores) + 1
        intercept_lb = -max(max_scores) + 1
    else:
        intercept_ub = abs(max_offset)
        intercept_lb = -abs(max_offset)

    coef_constraints.set_field('ub', '(Intercept)', intercept_ub)
    coef_constraints.set_field('lb', '(Intercept)', intercept_lb)
    #coef_constraints.view()

    #create SLIM IP
    slim_input = {
        'X': data['X'],
        'X_names': data['variable_names'],
        'Y': data['Y'],
        'C_0': c0_value,
        'w_pos': 1.0,
        'w_neg': 1.0,
        'L0_min': 0,
        'L0_max': max_size,
        'err_min': 0,
        'err_max': 1.0,
        'pos_err_min': 0,
        'pos_err_max': 1.0,
        'neg_err_min': 0,
        'neg_err_max': 1.0,
        'coef_constraints': coef_constraints
    }

    slim_IP, slim_info = slim.create_slim_ip(slim_input)
    return slim_IP, slim_info

if __name__ == '__main__':

    parser = setup_parser()
    parsed = parser.parse_args()
    parsed_dict = vars(parsed)
    parsed_string = [key + ' : ' + str(parsed_dict[key]) + '\n' for key in parsed_dict]
    parsed_string.sort()

    # setup logging
    logger = logging.getLogger()
    logger = slim.setup_logging(logger, log_to_console=(not parsed.silent), log_file=parsed.log)
    logger.setLevel(logging.INFO)
    logger.info("running 'create_slim_instance.py'")
    logger.info("current working directory: %r" % os.getcwd())
    logger.info("parsed command line arguments:\n-%s" % '-'.join(parsed_string))

    slim_IP, slim_info = create_slim_instance(data_file=parsed.data_file,
                                              max_coef=parsed.max_coef,
                                              max_size=parsed.max_size,
                                              max_offset=parsed.max_offset,
                                              c0_value=parsed.c0_value,
                                              logger=logger)
    logger.info("generated SLIM IP")

    slim_IP.write(parsed.instance_file)
    logger.info("saved SLIM IP to file: %s" % parsed.instance_file)

    if parsed.instance_info is not None:
        with open(parsed.instance_info, 'w') as fh:
            pickle.dump(slim_info, fh)
        logger.info("saved SLIM IP information to file: %s" % parsed.instance_info)

    logger.info("quitting")
    sys.exit(0)

