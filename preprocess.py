import argparse
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from common import PROG_TITLE, is_file, TARGET_COL, INDEX_COL, SEQFF_COL

if __name__ == '__main__':
    desc = 'Dataset shuffling, preprocessing and spliting into training and testing folds intended for .'
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description=desc)
    parser.add_argument('dataset', type=is_file, help='TSV file with fragment length profiles to preprocess')
    parser.add_argument('-k', '--kfold', type=int, default=5, help=desc)
    desc = 'random state to shuffle the dataset, dataset is not shuffled when omitted'
    parser.add_argument('-s', '--seed', type=int, help=desc)
    desc = 'number of training and testing folds to create'
    parser.add_argument('-t', '--target_col', type=str, default=TARGET_COL, help='target column label')
    parser.add_argument('-i', '--index_col', type=str, default=INDEX_COL, help='index column label')
    parser.add_argument('-q', '--seqff_col', type=str, default=SEQFF_COL, help='SeqFF prediction column label')
    parser.add_argument('-a', '--range', type=str, help='index range of fetal length features to use, e.g. "50:220"')
    parser.add_argument('-v', '--verbose', action='store_true', help='controls verbosity')
    args = parser.parse_args()
    
    dataset_df = pd.read_table(args.dataset, index_col=args.index_col)  # type: pd.DataFrame
    
    if args.verbose:
        print('dataset shape: %s' % str(dataset_df.shape))
    
    kfold = KFold(args.kfold, shuffle=args.seed is not None, random_state=args.seed)
    counter = 0
    for train_index, test_index in kfold.split(dataset_df):
        train_df = dataset_df.iloc[train_index]
        test_df = dataset_df.iloc[test_index]
        
        if args.verbose:
            print('fold %d training set shape: %s' % (counter, str(train_df.shape)))
            print('fold %d testing set shape: %s' % (counter, str(test_df.shape)))
        
        # get target feature
        train_seqff = None
        test_seqff = None
        if args.seqff_col in dataset_df:
            train_seqff = train_df.pop(args.seqff_col)
            test_seqff = test_df.pop(args.seqff_col)
        
        train_target = train_df.pop(args.target_col)
        test_target = test_df.pop(args.target_col)
        
        # get subrange of fetal length features
        if args.range is not None:
            from_len, to_len = args.range.split(':')
            train_df = train_df.iloc[:, int(from_len):int(to_len)]
            test_df = test_df.iloc[:, int(from_len):int(to_len)]
        
        # check training dataset on zero features
        check_df = train_df.sum(axis=0)
        empty_col_names = check_df[check_df == 0].index.values
        if len(empty_col_names):
            raise ValueError('empty training features %s, please narrow the feature range' % empty_col_names)
        
        # check testing dataset on zero features
        check_df = test_df.sum(axis=0)
        empty_col_names = check_df[check_df == 0].index.values
        if len(empty_col_names):
            raise ValueError('empty testing features %s, please narrow the feature range' % empty_col_names)
        
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
        np.savetxt(
            '%s/train_%s_mean_%d' % (dirname, filename, counter),
            mean_series.values, delimiter='\n'
        )
        np.savetxt(
            '%s/train_%s_std_%d' % (dirname, filename, counter),
            std_series.values, delimiter='\n'
        )
        
        # save training and testing SeqFF feature for Combo method
        if args.seqff_col in dataset_df:
            train_seqff.to_csv(
                '%s/train_seqff_%d.tsv' % (dirname, counter)
                , sep='\t', index_label=INDEX_COL, header=[SEQFF_COL]
            )
            test_seqff.to_csv(
                '%s/test_seqff_%d.tsv' % (dirname, counter),
                sep='\t', index_label=INDEX_COL, header=[SEQFF_COL]
            )
        
        # save training and testing dataset for FL method
        train_df.to_csv(
            '%s/train_%s_%d.tsv' % (dirname, filename, counter),
            sep='\t', index_label=INDEX_COL, header=True
        )
        test_df.to_csv(
            '%s/test_%s_%d.tsv' % (dirname, filename, counter),
            sep='\t', index_label=INDEX_COL, header=True
        )
        counter += 1
