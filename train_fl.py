import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.externals import joblib

from common import is_file, PROG_TITLE, create_svm, INDEX_COL, TARGET_COL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description='Trains FL (fetal length) model.')
    parser.add_argument('train', type=is_file, help='TSV file with preprocessed training fragment length profiles')
    parser.add_argument('test', type=is_file, help='TSV file with preprocessed testing fragment length profiles')
    parser.add_argument('-o', '--out_model', type=str, help='trained FL model', required=True)
    parser.add_argument('-c', '--out_coeffs', type=str, help='trained FL model coefficients')
    parser.add_argument('-r', '--out_result', type=str, help='trained model testing results')
    parser.add_argument('-v', '--verbose', action='store_true', help='controls verbosity')
    args = parser.parse_args()
    
    train_df = pd.read_table(args.train, index_col=INDEX_COL)  # type: pd.DataFrame
    test_df = pd.read_table(args.test, index_col=INDEX_COL)  # type: pd.DataFrame
    
    train_x, train_y = train_df, train_df.pop(TARGET_COL)
    test_x, test_y = test_df, test_df.pop(TARGET_COL)
    
    if args.verbose:
        print('training on %d samples and %d features' % (len(train_x), len(train_x.columns)))
    
    model = create_svm(args.verbose)
    model.fit(train_x, train_y)
    
    if args.verbose:
        print('testing on %d samples' % len(test_x))
    
    test_z = model.predict(test_x)
    pearson = pearsonr(test_y, test_z)
    
    mae = np.mean(np.abs(test_y - test_z))
    mse = np.mean((test_y - test_z) ** 2)
    
    if args.verbose:
        print('test pearsonr %f' % pearson[0])
        print('test mae %f' % mae)
        print('test mse %f' % mse)
    
    # save model parameters
    if args.out_coeffs is not None:
        np.savetxt(args.out_coeffs, model.coef_[0], delimiter='\n')
    
    if args.out_result is not None:
        with open(args.out_result, 'w') as result_file:
            result_file.write('%f\t%f\t%f\n' % (pearson[0], mae, mse))
    
    # save model
    joblib.dump(model, args.out_model)
