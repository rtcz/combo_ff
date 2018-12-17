import argparse

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

from common import PROG_TITLE, is_file, first_rank_ids

if __name__ == '__main__':
    desc = 'Predicts FF of single sample from FL model or combined model.'
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description=desc)
    desc = 'fragment length profile of a sample as tab separated values; '
    desc += 'if using combo model the last value must be a SeqFF prediction'
    parser.add_argument('sample', type=is_file, help=desc)
    parser.add_argument('fl_model', type=is_file, help='trained FL model')
    desc = 'list of means of each FL training dataset feature'
    parser.add_argument('-m', '--mean', type=is_file, required=True, help=desc)
    desc = 'list of standard deviations of each FL trainig dataset feature'
    parser.add_argument('-s', '--std', type=is_file, required=True, help=desc)
    parser.add_argument('-c', '--combo_model', type=is_file, help='trained combo model')
    parser.add_argument('-f', '--ranking', type=is_file, help='feature ranking')
    parser.add_argument('-v', '--verbose', action='store_true', help='controls verbosity')
    args = parser.parse_args()
    
    sample = np.loadtxt(args.sample)
    if args.combo_model is not None:
        fl_sample = sample[:-1]
        seqff_prediction = sample[-1]
    else:
        fl_sample = sample
        seqff_prediction = None
    
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
    
    # predict FF from FL model
    fl_model = joblib.load(args.fl_model)  # type: svm.SVR
    fl_prediction = fl_model.predict([fl_sample])
    
    if args.verbose:
        print('FL prediction %.4f' % fl_prediction)
    
    if args.combo_model is not None:
        combo_model = joblib.load(args.combo_model)  # type: LinearRegression
        combo_prediction = combo_model.predict([[fl_prediction, seqff_prediction]])
        
        if args.verbose:
            print('COMBO prediction %.4f' % combo_prediction)
    else:
        print(fl_prediction[0])
