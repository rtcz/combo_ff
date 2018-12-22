import argparse

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LinearRegression

from common import is_file, PROG_TITLE, INDEX_COL, TARGET_COL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description='Trains combined model.')
    parser.add_argument('train', type=is_file, help='TSV file with preprocessed training fragment length profiles')
    parser.add_argument('test', type=is_file, help='TSV file with preprocessed testing fragment length profiles')
    parser.add_argument('train_seqff', type=is_file, help='TSV file with training SeqFF feature (SeqFF estimation)')
    parser.add_argument('test_seqff', type=is_file, help='TSV file with testing SeqFF feature (SeqFF estimation)')
    parser.add_argument('fl_model', type=is_file, help='trained FL model')
    parser.add_argument('-o', '--out_model', type=str, help='trained combined model', required=True)
    parser.add_argument('-c', '--out_coeffs', type=str, help='trained combined model coefficients')
    parser.add_argument('-r', '--out_result', type=str, help='trained model testing results')
    parser.add_argument('-v', '--verbose', action='store_true', help='control verbosity')
    args = parser.parse_args()
    
    train_seqff_feature = pd.read_table(args.train_seqff, index_col=INDEX_COL)
    test_seqff_feature = pd.read_table(args.test_seqff, index_col=INDEX_COL)
    
    train_df = pd.read_table(args.train, index_col=INDEX_COL)
    test_df = pd.read_table(args.test, index_col=INDEX_COL)
    
    fl_train_x, train_y = train_df, train_df.pop(TARGET_COL)
    fl_test_x, test_y = test_df, test_df.pop(TARGET_COL)
    
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
        np.savetxt(args.out_coeffs, model.coef_, delimiter='\n')
    
    if args.out_result is not None:
        with open(args.out_result, 'w') as result_file:
            result_file.write('%f\t%f\t%f\n' % (pearson[0], mae, mse))
    
    # save model
    joblib.dump(model, args.out_model)
