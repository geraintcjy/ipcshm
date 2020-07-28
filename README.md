## Multi-class SVM Classifier

### Introduction

A multi-class SVM classifier implemented in Python. Uses multiprocessing library to train SVM classifier units in parallel and [Cvxopt](http://cvxopt.org) to solve the quadratic programming problem for each classifier unit.

Supports 10-fold cross validation, auto generation of  the confusion matrix and various accuracy measurements (TP, FP, Precision, Recall etc...)

This is a modified version aiming to solve problem 2 of the fitst IPC-SHM.

Original Programme: https://github.com/namoshizun/Multiclass-SVM-Classifier.git

### Run Locally

**Step 1**: Run ***toLabel.m***. This will transform original label.m document into a more readable label_r.m document.

**Step 2**: Run ***toCSV.py***. This will get you data and labels in one day, which respectively has a size of (72000, 912) and (1, 912)

**Step 3**: Run ***main.py***. Train with Multi-class SVM.

#####Dependencies: numpy, pandas, cvxopt, prettytable 

**Usage**:

```
usage: main.py [-h]
               [{linear,poly}] [{one_vs_one,one_vs_rest}] [C] [min_lagmult]
               [cross_validate] [evaluate_features] [{dev,prod}]

SVM Classifier

positional arguments:
  {linear,poly}         The kernel function to use
  {one_vs_one,one_vs_rest}
                        The strategy to implement a multiclass SVM. Choose
                        "one_vs_one" or "one_vs_rest"
  C                     The regularization parameter that trades off margin
                        size and training error
  min_lagmult           The support vector's minimum Lagrange multipliers
                        value
  cross_validate        Whether or not to cross validate SVM
  evaluate_features     Will read the cache of feature evaluation results if
                        set to False
  {dev,prod}            Reads dev data in ../input-dev/ if set to dev mode,
                        otherwise looks for datasets in ../input/

optional arguments:
  -h, --help            show this help message and exit
```

The program will load the training and testing datasets in ../input/, then saves the predictions of testing dataset into ../output/.
