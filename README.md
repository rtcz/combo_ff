public/in/dataset.tsv
1000 distributions of fetal lengths from length of 50 to 220


PREPROCESS
python3 preprocess.py dataset.tsv -i id -t target -q seqff -r 0.8 -s 0 -v

RFECV
python3 rfecv.py train_dataset.tsv ranking -v

TRAIN_FL
python3 train_fl.py train_dataset.tsv test_dataset.tsv -o fl_model -c fl_model_coeffs -f ranking -v

TRAIN_COMBO
python3 train_combo.py train_dataset.tsv test_dataset.tsv train_seqff_dataset.tsv test_seqff_dataset.tsv  fl_model -f ranking -o combo_model -v

PREDICT
python3 predict.py sample.tsv fl_model data/combo_model -m train_dataset_mean -s train_dataset_std -f ranking -v





hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 test.py data/model_ranking_flen data/test_dataset.tsv -v -f data/ranking_flen
dataset set has 65 features after elimination
pearsonr: (0.8385675272376429, 4.489381227356894e-131)

hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 test.py data/model data/test_dataset.tsv -v -f data/ranking
dataset set has 162 features after elimination
pearsonr: (0.8193701649066688, 3.0121876508363485e-120)

hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 test.py data/model_ranking_bio data/test_dataset.tsv -v -f data/ranking_bio
dataset set has 143 features after elimination
pearsonr: (0.8319814308092001, 3.3159954934328635e-127)

hekel@gen-main:/data/projects/nipt-heart/ff/utility$ python3 test.py data/model_no_ranking data/test_dataset.tsv -v
pearsonr: (0.8204261966649999, 8.266499266271641e-121)

