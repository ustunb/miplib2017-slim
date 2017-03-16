
def solve_slim_ip_instance(data_file_name, instance_file_name):

    # setup SLIM IP parameters
    # see docs/usrccplex.pdf for more about these parameters
    slim_IP.parameters.timelimit.set(10.0) #set runtime here
    #TODO: add these default settings to create_slim_IP
    slim_IP.parameters.randomseed.set(0)
    slim_IP.parameters.threads.set(1)
    slim_IP.parameters.parallel.set(1)
    slim_IP.parameters.output.clonelog.set(0)
    slim_IP.parameters.mip.tolerances.mipgap.set(np.finfo(np.float).eps)
    slim_IP.parameters.mip.tolerances.absmipgap.set(np.finfo(np.float).eps)
    slim_IP.parameters.mip.tolerances.integrality.set(np.finfo(np.float).eps)
    slim_IP.parameters.emphasis.mip.set(1)


    # solve SLIM IP
    slim_IP.solve()

    # run quick and dirty tests to make sure that IP output is correct
    slim.check_slim_IP_output(slim_IP, slim_info, X, Y, coef_constraints)

    #### CHECK RESULTS ####
    slim_results = slim.get_slim_summary(slim_IP, slim_info, X, Y)
    print(slim_results)

    # print model
    print(slim_results['string_model'])

    # print coefficient vector
    print(slim_results['rho'])

    # print accuracy metrics
    print 'error_rate: %1.2f%%' % (100*slim_results['error_rate'])
    print 'TPR: %1.2f%%' % (100*slim_results['true_positive_rate'])
    print 'FPR: %1.2f%%' % (100*slim_results['false_positive_rate'])
    print 'true_positives: %d' % slim_results['true_positives']
    print 'false_positives: %d' % slim_results['false_positives']
    print 'true_negatives: %d' % slim_results['true_negatives']
    print 'false_negatives: %d' % slim_results['false_negatives']
