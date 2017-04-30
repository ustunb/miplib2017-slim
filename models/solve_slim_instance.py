import os
import sys
import numpy as np
import argparse
import logging
import cplex as cpx
import pickle
import json

#add '/models/' directory to search path to avoid import errors
try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except:
    sys.path.append(os.getcwd()+'/models')

import slim as slim

# setup SLIM IP parameters
parsed = type('test', (), {})()
parsed.data_file = '/Users/berk/Desktop/Dropbox (MIT)/Research/SLIM/Toolboxes/miplib2017-slim/models/data/breastcancer_processed.csv'
parsed.instance_file = '/Users/berk/Desktop/Dropbox (MIT)/Research/SLIM/Toolboxes/miplib2017-slim/instances/breastcancer_max_5_features.mps'
parsed.instance_info = '/Users/berk/Desktop/Dropbox (MIT)/Research/SLIM/Toolboxes/miplib2017-slim/misc/breastcancer_max_5_features.p'
parsed.timelimit = 60

#load IP file
slim_IP = cpx.Cplex()
slim_IP.read(parsed.instance_file)

#load IP information
with open(parsed.instance_info) as fh:
    slim_info = pickle.load(fh)

#solve IP file using CPLEX
slim_IP.parameters.randomseed.set(0)
slim_IP.parameters.output.clonelog.set(0)
slim_IP.parameters.threads.set(1)
slim_IP.parameters.parallel.set(1)
slim_IP.parameters.mip.tolerances.mipgap.set(np.finfo(np.float).eps)
slim_IP.parameters.mip.tolerances.absmipgap.set(np.finfo(np.float).eps)
slim_IP.parameters.mip.tolerances.integrality.set(np.finfo(np.float).eps)
if parsed.timelimit < 0:
    slim_IP.parameters.timelimit.set(1e75)
else:
    slim_IP.parameters.timelimit.set(min(parsed.timelimit, 1e75))

# solve SLIM IP
slim_IP.solve()

# run quick and dirty tests to make sure that IP output is correct
data = slim.load_data_from_csv(parsed.data_file)
slim.check_slim_ip_solution(slim_IP, slim_info, data)

# get model results
slim_results = slim.get_slim_summary(slim_IP, slim_info, data)
print(slim_results)

# print model output to screen
print(slim_results['rho'])
print 'error_rate: %1.2f%%' % (100*slim_results['error_rate'])
print 'TPR: %1.2f%%' % (100*slim_results['true_positive_rate'])
print 'FPR: %1.2f%%' % (100*slim_results['false_positive_rate'])
print 'true_positives: %d' % slim_results['true_positives']
print 'false_positives: %d' % slim_results['false_positives']
print 'true_negatives: %d' % slim_results['true_negatives']
print 'false_negatives: %d' % slim_results['false_negatives']

# print model output to JSON
# TODO

# parsing
def setup_parser():
    """
    Create an argparse Parser object for command line arguments to create_slim_instance.
    This object determines all command line arguments, handles input
    validation and default values.

    See https://docs.python.org/3/library/argparse.html for configuration
    """

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
        description='Create a SLIM IP instance and save it as .mps file from the command shell',
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

    parser.add_argument('--timelimit',
                        type=is_positive_integer_or_negative_one,
                        default=300,
                        help='time limit on training (in seconds); set as -1 for no time limit')

    parser.add_argument('--log',
                        type=str,
                        help='name of the log file')

    parser.add_argument('--silent',
                        action='store_true',
                        help='flag to suppress logging to stderr')

    return parser

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
    logger.info("working directory: %r" % os.getcwd())
    logger.info("parsed the following variables:\n-%s" % '-'.join(parsed_string))

    logger.info("creating SLIM IP")
    slim_IP, slim_info = solve_slim_instance(data_file=parsed.data_file,
                                              max_coef=parsed.max_coef,
                                              max_size=parsed.max_size,
                                              max_offset=parsed.max_offset,
                                              c0_value=parsed.c0_value,
                                              logger=logger)
    logger.info("finished creating SLIM IP")

    logger.info("saving SLIM IP to disk")
    slim_IP.write(parsed.instance_file)
    logger.info("saved SLIM IP to file: %s" % parsed.instance_file)

    logger.info("quitting")
    sys.exit(0)



