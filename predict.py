import argparse

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

from common import PROG_TITLE, is_file, first_rank_ids

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description='Predicts FF of single sample.')
    parser.add_argument('sample', type=is_file, help='fetal lengths sample with SeqFF feature as last value')
    parser.add_argument('model_fl', type=is_file, help='trained FL model')
    parser.add_argument('model_combo', type=is_file, help='trained COMBO model')
    parser.add_argument('-m', '--mean', type=is_file, required=True, help='mean of each FL training dataset feature')
    parser.add_argument('-s', '--std', type=is_file, required=True,
                        help='standard deviation of each FL trainig dataset feature')
    # parser.add_argument('-c', '--out_coeffs', type=str, help='trained model coefficients')
    parser.add_argument('-f', '--ranking', type=is_file, help='feature rankings')
    parser.add_argument('-v', '--verbose', action='store_true', help='control verbosity')
    args = parser.parse_args()
    
    sample = np.loadtxt(args.sample)
    fl_sample = sample[:-1]
    seqff_feature = sample[-1]
    
    mean_list = np.loadtxt(args.mean, dtype=float)  # type: np.ndarray
    std_list = np.loadtxt(args.std, dtype=float)  # type: np.ndarray
    
    # keep only first rank features if using recursive feature elimination
    if args.ranking is not None:
        ranking_list = np.loadtxt(args.ranking, dtype=int)  # type: np.ndarray
        ids = first_rank_ids(ranking_list)
        fl_sample = fl_sample[ids]
        mean_list = mean_list[ids]
        std_list = std_list[ids]
    
    # standardize sample
    fl_sample = fl_sample / fl_sample.sum()
    fl_sample = (fl_sample - mean_list) / std_list
    
    # print(fl_sample)
    # print(seqff_feature)
    # exit(0)
    
    # predict FF from FL model
    fl_model = joblib.load(args.model_fl)  # type: svm.SVR
    fl_feature = fl_model.predict([fl_sample])
    
    if args.verbose:
        print('FL prediction %.4f' % fl_feature)
    
    combo_model = joblib.load(args.model_combo)  # type: LinearRegression
    result = combo_model.predict([[fl_feature, seqff_feature]])
    
    if args.verbose:
        print('Combo prediction %.4f' % result)
    
    print(result)
