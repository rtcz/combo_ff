import argparse

import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.metrics import make_scorer

from common import is_file, PROG_TITLE, create_svm, pearson_coeff, INDEX_COL, TARGET_COL

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog=PROG_TITLE, description='Recursive feature elimination.')
    parser.add_argument('dataset', type=is_file, help='preprocessed training dataset')
    parser.add_argument('out_rankings', type=str, help='tab separated feature rankings')
    parser.add_argument('-c', '--cv', type=int, help='number of cross validations', default=10)
    parser.add_argument('-j', '--jobs', type=int, help='number of cores to use in parallel', default=-1)
    parser.add_argument('-v', '--verbose', action='store_true', help='control verbosity')
    args = parser.parse_args()
    
    dataset_df = pd.read_table(args.dataset, index_col=INDEX_COL)
    model = RFECV(
        estimator=create_svm(False),
        scoring=make_scorer(pearson_coeff),
        cv=args.cv,
        n_jobs=args.jobs,
        verbose=args.verbose
    )
    target_series = dataset_df.pop(TARGET_COL)
    model.fit(dataset_df, target_series)
    
    np.savetxt(args.out_rankings, model.ranking_, delimiter='\n', fmt='%d')

