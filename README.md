data/dataset.tsv lengths from length of 50 to 220


PREPROCESS
python3 preprocess.py dataset.tsv -i id -t target -q seqff -r 0.8 -s 0 -v

RFECV
python3 rfecv.py train_dataset.tsv ranking -v

TRAIN_FL
python3 train_fl.py train_dataset.tsv test_dataset.tsv -o fl_model -c fl_model_coeffs -f ranking -v

TRAIN_COMBO
python3 train_combo.py train_dataset.tsv test_dataset.tsv train_seqff_dataset.tsv test_seqff_dataset.tsv  fl_model -f ranking -o combo_model -v

PREDICT
python3 predict.py sample.tsv fl_model -c data/combo_model -m train_dataset_mean -s train_dataset_std -f ranking -v
