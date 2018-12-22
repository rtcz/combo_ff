script_dir='..'
out_dir='out'
dataset='dataset.tsv'
sample='sample.tsv'
k_folds=5

mkdir -p ${out_dir}
mkdir -p ${out_dir}/data

mkdir -p ${out_dir}/fl/model
mkdir -p ${out_dir}/fl/coeffs
mkdir -p ${out_dir}/fl/result

mkdir -p ${out_dir}/combo/model
mkdir -p ${out_dir}/combo/coeffs
mkdir -p ${out_dir}/combo/result

echo "preprocessing"
# split the dataset into k traing and testing folds
python3 ${script_dir}/preprocess.py \
	${dataset} \
	-o ${out_dir}/data \
	-k ${k_folds} \
	-s 0 \
	-v

train() {
	# fl model
	python3 ${script_dir}/train_fl.py \
		${out_dir}/data/train_dataset_${1}.tsv \
		${out_dir}/data/test_dataset_${1}.tsv \
		-o ${out_dir}/fl/model/model_${1} \
		-c ${out_dir}/fl/coeffs/coeffs_${1}.txt \
		-r ${out_dir}/fl/result/result_${1}.tsv

	# combo model
	python3 ${script_dir}/train_combo.py \
		${out_dir}/data/train_dataset_${1}.tsv \
		${out_dir}/data/test_dataset_${1}.tsv \
		${out_dir}/data/train_seqff_${1}.tsv \
		${out_dir}/data/test_seqff_${1}.tsv \
		${out_dir}/fl/model/model_${1} \
		-o ${out_dir}/combo/model/model_${1} \
		-c ${out_dir}/combo/coeffs/coeffs_${1}.txt \
		-r ${out_dir}/combo/result/result_${1}.tsv
}

# train & test models
for ((i=0; i<${k_folds}; i++))
do
	train "$i" &
done

echo "waiting for training to finish"
wait

# aggregate testing results
cat ${out_dir}/fl/result/*.tsv > ${out_dir}/result_fl.tsv
cat ${out_dir}/combo/result/*.tsv > ${out_dir}/result_combo.tsv

echo "predicting"
# predict sample ff from trained models
for ((i=0; i<${k_folds}; i++))
do
	python3 ${script_dir}/predict.py \
		${sample} \
		${out_dir}/fl/model/model_${i} \
		-c ${out_dir}/combo/model/model_${i} \
		-m ${out_dir}/data/train_dataset_mean_${i}.txt \
		-s ${out_dir}/data/train_dataset_std_${i}.txt \
		-v
	echo ''
done
