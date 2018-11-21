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


def is_ratio(value: str) -> float:
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Value %s is not a float." % value)
    
    if value <= 0 or value >= 1:
        raise argparse.ArgumentTypeError("Ratio must be between 0 and 1.")
    
    return value


def first_rank_ids(ranking_list) -> list:
    ids = []
    for i in range(len(ranking_list)):
        if ranking_list[i] == 1:
            ids.append(i)
    
    return ids


def first_rank_df(df: pd.DataFrame, ranking_list) -> pd.DataFrame:
    """
    :param df: FL dataframe
    :param ranking_list: feature rankings by recursive feature elimination
    :return: FL dataframe without eliminated features
    """
    assert len(ranking_list) == len(df.columns)
    return df.iloc[:, first_rank_ids(ranking_list)]
