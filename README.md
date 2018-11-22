# Combo FF - combined method for fetal fraction estimation
Combo FF is a tool for estimation of fetal fraction using genomic data obtained from mothernal blood. The tool combines two estimation models: improved FL (Fragment Length) model and SeqFF model based on relative reads counts.
Combined model is trained on fetal fraction estimations of both models using ordinary least squares linear regression.
FL model is trained on fragment length profiles whereas estimation of fetal fraction from SeqFF model must be directly provided.

Support vector machine is used to train the FL model on fragment length profiles. It is recommended to use numbers of fragment lengths from 50 to 220 since this range contains almost all of the variability. In general, machine learning methods perform better with lower number of features. This number can be further reduced by using provided method for recursive feature elimination with cross validation.

Dataset and single sample is provided for demonstration.
The dataset `data/dataset.tsv` has 170 features with fragment length from 50 to 220 for training the FL model.
Another feature is SeqFF fetal fraction estimation for training the combined model.
SeqFF estimation was computed with pre-trained parameters from the original study.
Prediction target is a fetal fraction computed by a reliable Y-based method.
Single sample `data/sample.tsv` is intended for prediction on the trained model.
It is in the same format as a sample from the dataset.


## Preprocessing
Script shuffles and splits the dataset into training and testing set.
When SeqFF column label is provided it separates this column into dedicated training and testing set.
Fragment length profiles of samples are scaled to sum of one and all of fragment length features are normalized to have zero mean and unit variance.

```
python3 preprocess.py -h
usage: Combo FF [-h] [-s SEED] -r RATIO [-t TARGET_COL] [-i INDEX_COL]
                [-q SEQFF_COL] [-v]
                dataset

Dataset shuffling, preprocessing and spliting into trainig and testing set.

positional arguments:
  dataset               TSV file with fragment length profiles to preprocess

optional arguments:
  -h, --help            show this help message and exit
  -s SEED, --seed SEED  random state as integer
  -r RATIO, --ratio RATIO
                        ratio of training dataset
  -t TARGET_COL, --target_col TARGET_COL
                        target column label
  -i INDEX_COL, --index_col INDEX_COL
                        index column label
  -q SEQFF_COL, --seqff_col SEQFF_COL
                        SeqFF prediction column label
  -v, --verbose         controls verbosity
hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 eliminate.py -h
usage: Combo FF [-h] [-c CV] [-j JOBS] [-v] dataset out_rankings

Recursive feature elimination with cross validation.

positional arguments:
  dataset               TSV file with preprocessed training fragment length
                        profiles
  out_rankings          feature ranking

optional arguments:
  -h, --help            show this help message and exit
  -c CV, --cv CV        number of cross validations
  -j JOBS, --jobs JOBS  number of cores to use in parallel
  -v, --verbose         controls verbosity
```

```
python3 preprocess.py \
	data/dataset.tsv \
	-r 0.8 \
	-s 0
```

## Recursive feature elimination with cross validation
Optional method for ranking fragment length features.
Features with rank one contributing to the improvement of the trained model.
Output file contains ranking of all the features and ca be used in following scripts for their exclusion.

```
python3 eliminate.py -h
usage: Combo FF [-h] [-c CV] [-j JOBS] [-v] dataset out_rankings

Recursive feature elimination with cross validation.

positional arguments:
  dataset               TSV file with preprocessed training fragment length
                        profiles
  out_rankings          feature ranking

optional arguments:
  -h, --help            show this help message and exit
  -c CV, --cv CV        number of cross validations
  -j JOBS, --jobs JOBS  number of cores to use in parallel
  -v, --verbose         controls verbosity
hekel@gen-main:/data/projects/nipt-heart/ff/utility$ ^C
hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 train_fl.py 
usage: Combo FF [-h] -o OUT_MODEL [-c OUT_COEFFS] [-f RANKING] [-v] train test
Combo FF: error: the following arguments are required: train, test, -o/--out_model
hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 train_fl.py -h
usage: Combo FF [-h] -o OUT_MODEL [-c OUT_COEFFS] [-f RANKING] [-v] train test

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
  -f RANKING, --ranking RANKING
                        feature ranking
  -v, --verbose         controls verbosity
```

```
python3 eliminate.py \
	data/train_dataset.tsv \
	data/ranking
```

## Training FL model
Trains preprocessed training set with support vector machine algorithm.
After the model is trained it is tested on testing set.

```
python3 train_fl.py -h
usage: Combo FF [-h] -o OUT_MODEL [-c OUT_COEFFS] [-f RANKING] [-v] train test

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
  -f RANKING, --ranking RANKING
                        feature ranking
  -v, --verbose         controls verbosity

```

```
python3 train_fl.py \
	data/train_dataset.tsv \
	data/test_dataset.tsv \
	-o data/fl_model \
	-c data/fl_model_coeffs \
	-f data/ranking
```

## Training combined model
Combined model is trained with estimation of fetal fractions on the training dataset given by FL model together with provided estimation of SeqFF model.
The testing is done correspondingly.

```
python3 train_combo.py -h
usage: Combo FF [-h] -o OUT_MODEL [-c OUT_COEFFS] [-f RANKING] [-v]
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
  -f RANKING, --ranking RANKING
                        feature ranking
  -v, --verbose         control verbosity
```

```
python3 train_combo.py \
	data/train_dataset.tsv \
	data/test_dataset.tsv \
	data/train_seqff_dataset.tsv \
	data/test_seqff_dataset.tsv  \
	data/fl_model \
	-o data/combo_model
	-f data/ranking \
```

## Fetal fraction estimation
Fetal fraction of new sample can be estimated with trained FL model or the better performing combined model depending on the use of associated parameter.

```
python3 predict.py -h
usage: Combo FF [-h] -m MEAN -s STD [-c COMBO] [-f RANKING] [-v] sample fl

Predicts FF of single sample from FL model or combined model.

positional arguments:
  sample                fragment lengths sample with SeqFF feature as the last
                        value
  fl                    trained FL model

optional arguments:
  -h, --help            show this help message and exit
  -m MEAN, --mean MEAN  list of means of each FL training dataset feature
  -s STD, --std STD     list of standard deviations of each FL trainig dataset
                        feature
  -c COMBO, --combo COMBO
                        makes prediction by combined model using SeqFF feature
  -f RANKING, --ranking RANKING
                        feature ranking
  -v, --verbose         controls verbosity
```

```
python3 predict.py \
	sample.tsv fl_model \
	-c data/combo_model \
	-m data/train_dataset_mean \
	-s data/train_dataset_std \
	-f data/ranking
```
