# Combo FF - combined method for fetal fraction estimation
Combo FF is a tool for estimation of fetal fraction using genomic data obtained from mothernal blood. The tool combines two estimation methods: FL (Fragment Length) method and SeqFF method based on relative reads counts.
FL model is trained with support vector machine regressor on fragment length profiles.
Combined model is trained on fetal fraction estimations of both models using ordinary least squares linear regression.
To train the combined model, user must provide fetal fractions estimations from the SeqFF model.
For general usage it is recommended to use fragment length profile from the length of 50 to 220 since this range captures almost all of the variability between the profiles.

Dataset and single sample is provided for demonstration.
The dataset `example/dataset.tsv` has 170 features with fragment length from 50 to 220 for training the FL model.
Another feature is SeqFF fetal fraction estimation for training the combined model.
SeqFF estimation was computed with pre-trained parameters from the original study.
Prediction target is a fetal fraction computed by a reliable Y-based method.
Single sample `example/sample.tsv` is an independent sample intended for prediction from the trained model.
The bash script `example/run.sh` cotanins complete usage of the tool from data preprocessing to fetal fraction prediction from single sample.


## Preprocessing
Script shuffles and splits the dataset into number of training and testing folds intended for k-fold cross validation.
When SeqFF column label is provided, the column is separated from the original dataset and saved in dedicated training and testing datasets.
For each fold fragment length profiles of samples are scaled to sum of one and all of the fragment length features are normalized to have zero mean and unit variance with respect to the training dataset.

```
usage: preprocess.py [-h] [-k KFOLD] [-s SEED] [-o OUT] [-t TARGET_COL]
                     [-i INDEX_COL] [-q SEQFF_COL] [-a RANGE] [-v]
                     dataset

Dataset shuffling, preprocessing and spliting into training and testing folds.

positional arguments:
  dataset               TSV file with fragment length profiles to preprocess

optional arguments:
  -h, --help            show this help message and exit
  -k KFOLD, --kfold KFOLD
                        number of training and testing folds to create
  -s SEED, --seed SEED  random state to shuffle the dataset, dataset is not
                        shuffled when omitted
  -o OUT, --out OUT     output directory
  -t TARGET_COL, --target_col TARGET_COL
                        target column label
  -i INDEX_COL, --index_col INDEX_COL
                        index column label
  -q SEQFF_COL, --seqff_col SEQFF_COL
                        SeqFF prediction column label
  -a RANGE, --range RANGE
                        index range of fetal lengths to use, e.g. "50:220"
  -v, --verbose         controls verbosity
```


## Training FL model
Trains FL model with support vector machine regressor on preprocessed training set.
After the model is trained it is tested on the testing set.

```
usage: train_fl.py [-h] -o OUT_MODEL [-c OUT_COEFFS] [-r OUT_RESULT] [-v]
                   train test

Trains FL (fetal length) model.

positional arguments:
  train                 TSV file with preprocessed training fragment length
                        profiles
  test                  TSV file with preprocessed testing fragment length
                        profiles

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_MODEL, --out_model OUT_MODEL
                        trained FL model
  -c OUT_COEFFS, --out_coeffs OUT_COEFFS
                        trained FL model coefficients
  -r OUT_RESULT, --out_result OUT_RESULT
                        trained model testing results
  -v, --verbose         controls verbosity
```

## Training combined model
Combined model is trained on estimations of fetal fractions from FL model together with provided estimations from SeqFF model.
After the model is trained it is tested on the testing set.

```
combo.py -h
usage: train_combo.py [-h] -o OUT_MODEL [-c OUT_COEFFS] [-r OUT_RESULT] [-v]
                      train test train_seqff test_seqff fl_model

Trains combined model.

positional arguments:
  train                 TSV file with preprocessed training fragment length
                        profiles
  test                  TSV file with preprocessed testing fragment length
                        profiles
  train_seqff           TSV file with training SeqFF feature (SeqFF
                        estimation)
  test_seqff            TSV file with testing SeqFF feature (SeqFF estimation)
  fl_model              trained FL model

optional arguments:
  -h, --help            show this help message and exit
  -o OUT_MODEL, --out_model OUT_MODEL
                        trained combined model
  -c OUT_COEFFS, --out_coeffs OUT_COEFFS
                        trained combined model coefficients
  -r OUT_RESULT, --out_result OUT_RESULT
                        trained model testing results
  -v, --verbose         control verbosity
```

## Fetal fraction estimation
Fetal fraction of a new sample can be estimated with trained FL model or the better performing combined model.

```
usage: predict.py [-h] -m MEAN -s STD [-c COMBO_MODEL] [-f RANKING] [-v]
                  sample fl_model

Predicts FF of single sample from FL model or combined model.

positional arguments:
  sample                fragment length profile of a sample as tab separated
                        values; if using combo model the last value must be a
                        SeqFF prediction
  fl_model              trained FL model

optional arguments:
  -h, --help            show this help message and exit
  -m MEAN, --mean MEAN  list of means of each FL training dataset feature
  -s STD, --std STD     list of standard deviations of each FL trainig dataset
                        feature
  -c COMBO_MODEL, --combo_model COMBO_MODEL
                        trained combo model
  -v, --verbose         controls verbosity
```
