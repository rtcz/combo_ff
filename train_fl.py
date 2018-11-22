import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.externals import joblib

from common import is_file, PROG_TITLE, create_svm, first_rank_df, INDEX_COL, TARGET_COL


def is_train_ratio(value: str) -> float:
    try:
        value = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Value %s is not a float." % value)
    
    if value <= 0 or value > 1:
        raise argparse.ArgumentTypeError("Train ratio must be between 0 and 1 included.")
    
    return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description='Trains FL (fetal length) model.')
    parser.add_argument('train', type=is_file, help='TSV file with preprocessed training fragment length profiles')
    parser.add_argument('test', type=is_file, help='TSV file with preprocessed testing fragment length profiles')
    parser.add_argument('-o', '--out_model', type=str, help='trained FL model', required=True)
    parser.add_argument('-c', '--out_coeffs', type=str, help='trained FL model coefficients')
    parser.add_argument('-f', '--ranking', type=is_file, help='feature ranking')
    parser.add_argument('-v', '--verbose', action='store_true', help='controls verbosity')
    args = parser.parse_args()
    
    train_df = pd.read_table(args.train, index_col=INDEX_COL)  # type: pd.DataFrame
    test_df = pd.read_table(args.test, index_col=INDEX_COL)  # type: pd.DataFrame
    
    train_x, train_y = train_df, train_df.pop(TARGET_COL)
    test_x, test_y = test_df, test_df.pop(TARGET_COL)
    
    if args.ranking is not None:
        ranking_list = np.loadtxt(args.ranking, dtype=int)  # type: np.ndarray
        train_x = first_rank_df(train_x, ranking_list)
        test_x = first_rank_df(test_x, ranking_list)
    
    if args.verbose:
        print('training on %d samples and %d features' % (len(train_x), len(train_x.columns)))
    
    model = create_svm(args.verbose)
    model.fit(train_x, train_y)
    
    if args.verbose:
        print('testing on %d samples' % len(test_x))
    
    train_z = model.predict(train_x)
    test_z = model.predict(test_x)
    
    print('train pearsonr %s' % str(pearsonr(train_y, train_z)))
    print('test pearsonr %s' % str(pearsonr(test_y, test_z)))
    
    # save model parameters
    if args.out_coeffs is not None:
        np.savetxt(args.out_coeffs, model.coef_[0], delimiter='\n')
    
    # save model
    joblib.dump(model, args.out_model)
