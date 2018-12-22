import argparse
import os

import pandas as pd
from scipy.stats import pearsonr
from sklearn import svm

PROG_TITLE = 'Combo FF'

INDEX_COL = 'id'
SEQFF_COL = 'seqff'
TARGET_COL = 'target'


# this must be here, see https://github.com/scikit-learn/scikit-learn/issues/12250
def pearson_coeff(target_y: list, predicted_y: list) -> float:
    """
    :param target_y:
    :param predicted_y:
    :return: pearson correlation coefficient
    """
    return pearsonr(target_y, predicted_y)[0]


def create_svm(verbose: bool) -> svm.SVR:
    return svm.SVR(
        kernel='linear',
        C=1.0,
        epsilon=0.01,
        tol=0.001,
        verbose=verbose
    )


def is_file(value: str) -> str:
    if os.path.isfile(value):
        return value
    else:
        raise argparse.ArgumentTypeError("Value %s is not a file." % value)


def is_dir(value: str) -> str:
    if os.path.isdir(value):
        return value
    else:
        raise argparse.ArgumentTypeError("Value %s is not a directory." % value)

