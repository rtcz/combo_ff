import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

from common import is_file, PROG_TITLE, first_rank_df, INDEX_COL, TARGET_COL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description='Trains combined model.')
    parser.add_argument('train', type=is_file, help='training FL dataset')
    parser.add_argument('test', type=is_file, help='testing FL dataset')
    parser.add_argument('train_seqff', type=is_file, help='training SeqFF feature (predicted)')
    parser.add_argument('test_seqff', type=is_file, help='testing SeqFF feature')
    parser.add_argument('fl_model', type=is_file, help='trained FL model')
    parser.add_argument('-o', '--out_model', type=str, help='trained model', required=True)
    parser.add_argument('-c', '--out_coeffs', type=str, help='trained model coefficients')
    parser.add_argument('-f', '--ranking', type=is_file, help='feature ranking')
    parser.add_argument('-v', '--verbose', action='store_true', help='control verbosity')
    args = parser.parse_args()
    
    train_seqff_feature = pd.read_table(args.train_seqff, index_col=INDEX_COL)
    test_seqff_feature = pd.read_table(args.test_seqff, index_col=INDEX_COL)
    
    train_df = pd.read_table(args.train, index_col=INDEX_COL)
    test_df = pd.read_table(args.test, index_col=INDEX_COL)
    
    fl_train_x, train_y = train_df, train_df.pop(TARGET_COL)
    fl_test_x, test_y = test_df, test_df.pop(TARGET_COL)
    
    # keep only first rank features if using recursive feature elimination
    if args.ranking is not None:
        ranking_list = np.loadtxt(args.ranking, dtype=int)  # type: np.ndarray
        fl_train_x = first_rank_df(fl_train_x, ranking_list)
        fl_test_x = first_rank_df(fl_test_x, ranking_list)
    
    fl_model = joblib.load(args.fl_model)  # type: svm.SVR
    train_fl_feature = fl_model.predict(fl_train_x)
    test_fl_feature = fl_model.predict(fl_test_x)
    
    # merge FL and SeqFF method predictions to single training and testing dataset
    train_x = pd.DataFrame(train_fl_feature, index=fl_train_x.index, columns=['fl']).join(train_seqff_feature)
    test_x = pd.DataFrame(test_fl_feature, index=fl_test_x.index, columns=['fl']).join(test_seqff_feature)
    
    if args.verbose:
        print('training on %d samples and %d features' % (len(train_x), len(train_x.columns)))
    
    model = LinearRegression()
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
