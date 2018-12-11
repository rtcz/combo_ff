import argparse
import os

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

from common import PROG_TITLE, is_file, is_ratio, TARGET_COL, INDEX_COL, SEQFF_COL

if __name__ == '__main__':
    desc = 'Dataset shuffling, preprocessing and spliting into training and testing set.'
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description=desc)
    parser.add_argument('dataset', type=is_file, help='TSV file with fragment length profiles to preprocess')
    parser.add_argument('-s', '--seed', type=int, help='random state as integer')
    parser.add_argument('-r', '--ratio', type=is_ratio, default='id', required=True, help='ratio of training dataset')
    parser.add_argument('-t', '--target_col', type=str, default=TARGET_COL, help='target column label')
    parser.add_argument('-i', '--index_col', type=str, default=INDEX_COL, help='index column label')
    parser.add_argument('-q', '--seqff_col', type=str, default=SEQFF_COL, help='SeqFF prediction column label')
    parser.add_argument('-a', '--range', type=str, help='index range of fetal length features to use, e.g. "50:220"')
    parser.add_argument('-v', '--verbose', action='store_true', help='controls verbosity')
    args = parser.parse_args()
    
    dataset_df = pd.read_table(args.dataset, index_col=args.index_col)  # type: pd.DataFrame
    dataset_df = shuffle(dataset_df, random_state=args.seed)
    
    # split the dataset
    train_len = int(len(dataset_df) * args.ratio)
    train_df = dataset_df[:train_len]
    test_df = dataset_df[train_len:]
    
    # get target feature
    train_seqff = None
    test_seqff = None
    if args.seqff_col in dataset_df:
        train_seqff = train_df.pop(args.seqff_col)
        test_seqff = test_df.pop(args.seqff_col)
    
    train_target = train_df.pop(args.target_col)
    test_target = test_df.pop(args.target_col)
    
    # get subrange of fetal length fratures
    if args.range is not None:
        from_len, to_len = args.range.split(':')
        train_df = train_df.iloc[:, int(from_len):int(to_len)]
        test_df = test_df.iloc[:, int(from_len):int(to_len)]
    
    if args.verbose:
        print('training set shape: %s' % str(train_df.shape))
        print('testing set shape: %s' % str(test_df.shape))
    
    # check training dataset on zero features
    check_df = train_df.sum(axis=0)
    empty_cols = check_df[check_df == 0].index.values
    if len(empty_cols):
        raise ValueError('empty training features %s, please narrow the feature range' % empty_cols)
    
    # check testing dataset on zero features
    check_df = test_df.sum(axis=0)
    empty_cols = check_df[check_df == 0].index.values
    if len(empty_cols):
        raise ValueError('empty testing features %s, please narrow the feature range' % empty_cols)
    
    # standardize samples (sum of lengths of each sample is 1)
    train_df = train_df.div(train_df.sum(axis=1), axis=0)
    test_df = test_df.div(test_df.sum(axis=1), axis=0)
    
    mean_series = train_df.mean(axis=0)  # type: pd.Series
    std_series = train_df.std(axis=0)  # type: pd.Series
    
    # scale training dataset
    train_df = train_df.sub(mean_series, axis=1)
    train_df = train_df.div(std_series, axis=1)
    
    # scale testing dataset with parameters of training dataset
    test_df = test_df.sub(mean_series, axis=1)
    test_df = test_df.div(std_series, axis=1)
    
    # insert targets back to the dataframes
    train_df[TARGET_COL] = train_target
    test_df[TARGET_COL] = test_target
    
    dirname, basename = args.dataset.rsplit('/', maxsplit=1)
    filename, extension = os.path.splitext(basename)
    
    # save training mean and std of each fetal length feature
    np.savetxt(dirname + '/train_' + filename + '_mean', mean_series.values, delimiter='\n')
    np.savetxt(dirname + '/train_' + filename + '_std', std_series.values, delimiter='\n')
    
    # save training and testing SeqFF feature for Combo method
    if args.seqff_col in dataset_df:
        train_seqff.to_csv(dirname + '/train_seqff_' + basename, sep='\t', index_label=INDEX_COL, header=[SEQFF_COL])
        test_seqff.to_csv(dirname + '/test_seqff_' + basename, sep='\t', index_label=INDEX_COL, header=[SEQFF_COL])
    
    # save training and testing dataset for FL method
    train_df.to_csv(dirname + '/train_' + basename, sep='\t', index_label=INDEX_COL, header=True)
    test_df.to_csv(dirname + '/test_' + basename, sep='\t', index_label=INDEX_COL, header=True)
