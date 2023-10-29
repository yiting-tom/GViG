#!/usr/bin/env bash

root=$(dirname "$(dirname "$(readlink -f "$0")")")
source ${root}/scripts/helper_functions.sh

# Basic Settings
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export MASTER_PORT=6033
export GPUS_PER_NODE=1

# Experiment Settings
exp_tag=example 		# experiment tag
ckpt_tag=checkpoint_best.pt
arch=tiny    		# model architecture(define in pretrained_weights folder, e.g. huge, large, base, medium, base, tiny)
trainP=Instruct-2   # training prompt id
valP=Base      		# validation prompt id
testP=Instruct-2    # test prompt id

batch_size=1
seed=7
beam=1

# ================================================================================
# Please do not change the settings below
# ================================================================================

# Basic Settings
task=wsdm_vqa
separator=,
selected_cols=0,1,2,3,4,5,6,7

# Path Settings
folder_struc=${arch}/${exp_tag}/train-P${trainP}/val-P${valP}
bpe_dir=${root}/utils/BPE
user_dir=${root}/gvig_module

# Dataset Settings
data_dir=${root}/datasets/${exp_tag}          # dataset path
test_data=${data_dir}/test_private-P${testP}.csv
assert_file_exists ${test_data}
data=${test_data}

# Checkpoint Settings
ckpt_dir=${root}/checkpoints/${folder_struc} # checkpoint directory path
ckpt_path=${ckpt_dir}/${ckpt_tag}
assert_file_exists ${ckpt_path}

# Result Settings
result_dir=${root}/results/${folder_struc}   # result directory path
result_name=private_test-P${testP}

# Display Experiment Settings
display_exp_settings

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python3 \
    -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    ${root}/evaluate.py \
        ${test_data} \
        --path=${ckpt_path} \
        --user-dir=${user_dir} \
        --task=${task} \
        --batch-size=${batch_size} \
        --seed=${seed} \
        --gen-subset=${result_name} \
        --results-path=${result_dir} \
        --beam=${beam} \
        --log-format=simple --log-interval=10 \
        --min-len=4 \
        --max-len-a=0 \
        --max-len-b=4 \
        --no-repeat-ngram-size=3 \
        --use-csv \
        --fp16 \
        --num-workers=0 \
        --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
