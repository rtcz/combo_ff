# Combo FF - combined method for fetal fraction estimation
Combo FF is a tool for estimation of fetal fraction using genomic data obtained from mothernal blood. The tool combines two estimation models: improved FL (Fragment Length) model and SeqFF model based on relative reads counts.
Combined model is trained on fetal fraction estimations of both models using ordinary least squares linear regression.
FL model is trained on fragment length profiles whereas estimation of fetal fraction from SeqFF model must be directly provided.

Support vector machine is used to train the FL model on fragment length profiles.
It is recommended to use lengths from 50 to 220 since this range contains almost all of the fragments.

Dataset and single sample is provided for demonstration.
The dataset `data/dataset.tsv` has 170 features with fragment length from 50 to 220 for training the FL model.
Another feature is SeqFF fetal fraction estimation for training the combined model.
SeqFF estimation was computed with pre-trained parameters from the original study.
Prediction target is a fetal fraction computed by a reliable Y-based method. Single sample `data/sample.tsv` is intended for prediction on the trained model. It is in the same format as a sample from the dataset.


## Preprocessing
```
python3 preprocess.py \
	data/dataset.tsv \
	-r 0.8 \
	-s 0
```

## Recursive feature elimination with cross validation
```
python3 eliminate.py \
	data/train_dataset.tsv \
	ranking
```

## Training FL model
```
python3 train_fl.py \
	train_dataset.tsv \
	test_dataset.tsv \
	-o fl_model \
	-c fl_model_coeffs \
	-f ranking
```

## Training combined model
```
python3 train_combo.py \
	train_dataset.tsv \
	test_dataset.tsv \
	train_seqff_dataset.tsv \
	test_seqff_dataset.tsv  \
	fl_model \
	-o combo_model
	-f ranking \
```

## Fetal fraction prediction
```
python3 predict.py \
	sample.tsv fl_model \
	-c data/combo_model \
	-m train_dataset_mean \
	-s train_dataset_std \
	-f ranking
```
