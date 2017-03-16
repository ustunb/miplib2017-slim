# miplib2017-slim

This repository contains code to reproduce the SLIM IP instances in our submission to [MIPLIB 2017](https://miplibsubmissions.zib.de/).

## MIPLIB Submission

Our submission includes 9 instances of the SLIM IP (3 datasets x 3 instances per dataset). We picked instances to explore solver performance with regards to different computational tasks that are relevant to machine learning applications (e.g., 0-1 loss minimization, feature selection, discrete constraints).

Each instance is stored in the ``/instances/`` directory in [MPS](https://en.wikipedia.org/wiki/MPS_(format)) format. The list of instances, in order of difficulty, include:

1. ``breastcancer_max_5_features.mps`` (easiest to solve)
2. ``breastcancer_regularized.mps``
3. ``breastcancer_best.mps``
4. ``mushroom_max_5_features.mps``
5. ``mushroom_regularized.mps``
6. ``mushroom_best.mps``
7. ``adult_max_5_features.mps``
8. ``adult_regularized.mps``
9. ``adult_best.mps`` (hardest to solve)

Based on our experience with CPLEX 12.7, we expect that a commercial MIP solver will be able to solve instances 1-4 in <10 minutes and 5-7 in <1 hour. The solution to 8-9 will take longer.  

We produced the instance files using source code from the [slim-python](https://github.com/ustunb/slim-python) package using the CPLEX Python API 12.7. To recreate the instances, simply run ``/models/create_slim_instances.py``

We note that the SLIM IP formulation may change in the future. 

## Instance Overview 

Each instance of the SLIM IP is designed to output a scoring system for a binary classification dataset with *N* samples and *d* features. For a given dataset, the SLIM IP has roughly *N + 3d* variables and *N + 5d* constraints (with a few additional variables/constraints based on the problem type). 


#### Datasets 

We chose 3 popular datasets from the [UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/).
  
| dataset      | classification task                                | samples (N) | features (d) | source                                                                                                                     |
|--------------|----------------------------------------------------|-------------|--------------|----------------------------------------------------------------------------------------------------------------------------|
| breastcancer | detect breast cancer using a biopsy                | 683         | 9            | [link](https://archive.ics.uci.edu/ml/datasets/https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)) |
| mushroom     | predict if a mushroom is poisonous                 | 8124        | 113          | [link](https://archive.ics.uci.edu/ml/datasets/Mushroom)                                                                   |
| adult        | predict if a U.S. resident earns more than $50 000 | 32561       | 36           | [link](https://archive.ics.uci.edu/ml/datasets/Adult)        

The instances were built with processed versions of the original datasets (to remove missing entries, recode categorical features and binarize some real-valued features). We include the processed datasets in ``/models/data/`` and links to the original datasets in the source column of the table above.

#### Problem Types

For each ``dataset`` in {``breastcancer``, ``mushroom``, ``adult``}, we formulated 3 instances of the SLIM IP in order to vary the underlying difficulty of the problem. The problem types include:

- ``[dataset]_best``: This outputs the most accurate scoring system with coefficients in {-10,...,10}. 
- ``[dataset]_max_5_features``: This outputs the most accurate scoring system with at most 5 non-zero coefficients in {-10,...,10} (i.e. no more than 5 features).
- ``[dataset]_regularized``:  This outputs the optimal scoring system with coefficients in {-10,...,10} for a L0-regularization penalty of C0 = 0.01 (i.e., each feature will improve training accuracy by at most 1%)

#### Difficulty

The SLIM IP involves three computationally challenging tasks: 

1. 0-1 loss minimization (harder when N is large)
2. feature selection / L0-regularization (harder when d is large)
3. minimization over integers  (harder when d is large)

Each task becomes more difficult as *N* or *d* increases. However, the difficulty of the SLIM IP also depends on the inherent difficultly of the underlying classification problem. For instance, even if *N* and *d* are very large, the SLIM IP might be easy to solve if the data are linearly separable (this is the case for ``mushroom``).

For a given problem type, we expect: ``breastcancer`` will be easier than ``mushroom`` is easier than ``adult``.  The reason why ``mushroom`` is easier than ``adult`` is because ``mushroom`` is linearly separable while ``adult`` is not. 

For a given dataset, the difficulty of the SLIM IP is governed by the feature selection. We expect that the ``max_5_features`` problem will be the easiest (well-defined limit on # of features), followed by `penalized` (implied limit on the number of features), and then `best` (no effective feature selection).  

#### SLIM

[SLIM](http://http//arxiv.org/abs/1502.04269/) is a machine learning method to build data-driven *scoring systems.* These are simple classification models let users make quick predictions by adding, subtracting and multiplying a few small numbers. SLIM scoring systems are built solving an integer program (IP). In contrast to current machine learning methods, this results in models that are fully optimized for accuracy, sparsity, and integer coefficients, and that can satisfy difficult constraints **without parameter tuning** (e.g. hard limits on model size, sensitivity, specificity).

If you are interested in learning more about SLIM, check out [our paper](http://http//arxiv.org/abs/1502.04269/):
```
@article{
    ustun2015slim,
    year = {2015},
    issn = {0885-6125},
    journal = {Machine Learning},
    doi = {10.1007/s10994-015-5528-6},
    title = {Supersparse linear integer models for optimized medical scoring systems},
    url = {http://dx.doi.org/10.1007/s10994-015-5528-6},
    publisher = { Springer US},
    author = {Ustun, Berk and Rudin, Cynthia},
    pages = {1-43},
    language = {English}
}
```

If you are interested in creating scoring systems on other datasets, check out the following repositories:

- [slim-matlab](https://github.com/ustunb/slim-matlab) (uses MATLAB + CPLEX)
- [slim-python](https://github.com/ustunb/slim-python) (uses Python + CPLEX)
- [risk-slim](https://github.com/ustunb/risk-slim) (produces models for risk-assessment; uses Python + CPLEX)


