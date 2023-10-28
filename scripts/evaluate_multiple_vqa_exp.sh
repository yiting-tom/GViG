#!/usr/bin/env bash

root=$(dirname "$(dirname "$(readlink -f "$0")")")
source ${root}/scripts/helper_functions.sh

# Basic Settings
export CUDA_VISIBLE_DEVICES=0,1
export GPUS_PER_NODE=2
export LOCAL_RANK=0
export MASTER_PORT=6033

# Experiment Settings
exp_tag=sample 		# experiment tag
arch=tiny    		# model architecture(define in pretrained_weights folder, e.g. huge, large, base, medium, base, tiny)
ckpt_path=${root}/pretrained_weights/ofa_${arch}.pt
trainP=1     		# training prompt id
valP=1       		# validation prompt id
testP=1             # test prompt id
test_files=(
    ${root}/datasets/official/train_sample.csv
    ${root}/datasets/official/test_public.csv
)

seed=8
beam=1
batch_size=128

# ================================================================================
# Please do not change the settings below
# ================================================================================

# Basic Settings
task=vqa_gen
# id, base64, question, answer, objectLabels
# image, question
separator=,
selected_cols=0,4

# Path Settings
folder_struc=${arch}/${exp_tag}
result_dir=${root}/results/vqa/${folder_struc}
bpe_dir=${root}/utils/BPE
user_dir=${root}/gvig_module

# check all file exists
assert_file_exists ${ckpt_path}

declare -a line_counts
total_lines=0

random=$(openssl rand -hex 16)
# set test data as the concatenated file
test_data=evaluate_vqa.${random}.csv

# Initialize an empty evaluate_vqa.tmp.csv
> ${test_data}
IFS=', '


# Process each file in test_files array
# Detect the number of lines for each file, adjusted for files starting with "image"
for file in "${test_files[@]}"; do
    assert_file_exists ${file}

    # Check if the first line starts with "image"
    if [[ $(head -n 1 "$file") == image* ]]; then
        # Subtract 1 from the line count
        lines=$(($(wc -l < "$file") - 1))
        # If yes, skip the first line and append the rest to evaluate_vqa.tmp.csv
        tail -n +2 "$file" >> ${test_data}
    else
        lines=$(wc -l < "$file")
        # Otherwise, append the entire file
        cat "$file" >> ${test_data}
    fi

    line_counts+=($lines)
    total_lines=$(($total_lines + $lines))
done

result_name=$(basename ${test_data} .csv)
display_exp_settings
echo "        Concatenating files: ${test_files[*]} -> ${test_data}"
echo "      Each file line counts: ${line_counts[*]} -> ${total_lines}"
echo "------------------------------------------------------------------------------------------"

# run evaluation
# -m torch.distributed.launch \
# --nproc_per_node=${GPUS_PER_NODE} \
# --master_port=${MASTER_PORT} \
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
python3 \
    ${root}/evaluate.py \
        ${test_data} \
        --path=${ckpt_path} \
        --user-dir=${user_dir} \
        --task=${task} \
        --bpe-dir=${bpe_dir} \
        --batch-size=${batch_size} \
        --seed=${seed} \
        --gen-subset=${result_name} \
        --results-path=${result_dir} \
        --beam=${beam} \
        --patch-image-size=512 \
        --prompt-type='none' \
        --log-format=simple --log-interval=10 \
        --fp16 \
        --zero-shot \
        --unnormalized \
        --temperature=1.0 \
        --num-workers=0 \
        --separator="${separator}" \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\",}"

echo "----------------------------------------- Output ------------------------------------------"
# Split the processed file back into individual test_files
start_line=1
for i in "${!test_files[@]}"; do
    end_line=$(($start_line + ${line_counts[$i]} - 1))
    merged_file=${result_dir}/$(basename ${test_data} .csv)_predict.csv
    target_file=${result_dir}/$(basename ${test_files[$i]})
    sed -n "$start_line,${end_line}p" ${merged_file} > ${target_file}
    echo "File generated: '${target_file}', row count: '$(wc -l < ${target_file})'"
    start_line=$(($end_line + 1))
done
rm ${test_data}
rm ${merged_file}
echo "-------------------------------------------------------------------------------------------"