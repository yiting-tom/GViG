#!/bin/bash

root=$(dirname "$(dirname "$(readlink -f "$0")")")
source ${root}/scripts/helper_functions.sh

# Basic Settings
export CUDA_VISIBLE_DEVICES=0,1
export LOCAL_RANK=0
export MASTER_PORT=6033
export GPUS_PER_NODE=2

# Experiment Settings
exp_tag=sample 		# experiment tag
arch=tiny    		# model architecture(define in pretrained_weights folder, e.g. huge, large, base, medium, base, tiny)
trainP=1     		# training prompt id
valP=1       		# validation prompt id

# Hyperparameter Settings
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=4
warmup_ratio=0.06
batch_size=1
update_freq=1
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=512

# ================================================================================
# Please do not change the settings below
# ================================================================================

# Basic Settings
root=$(dirname "$(dirname "$(readlink -f "$0")")")

task=wsdm_vqa
selected_cols=0,1,2,3,4,5,6,7

# Path Settings
pretrained_weights=${root}/pretrained_weights/ofa_${arch}.pt
folder_struc=${arch}/${exp_tag}/train-P${trainP}
bpe_dir=${root}/utils/BPE
user_dir=${root}/gvig_module

# Dataset Settings
data_dir=${root}/datasets/${exp_tag}          # dataset path
train_data=${data_dir}/train-P${trainP}.csv   # train data path
val_data=${data_dir}/test_public-P${valP}.csv # validation data path
assert_file_exists ${train_data}
assert_file_exists ${val_data}
train_val_files=${train_data},${val_data}

# Tensorboard Settings
tensorboard_dir=${root}/tensorboard/${folder_struc}/val-P${valP} # tensorboard log path
mkdir -p ${tensorboard_dir}

# Logging Settings
log_dir=${root}/logs/${folder_struc} # log directory path
log_path=${log_dir}/val-P${valP}.log # log file path
mkdir -p ${log_dir}

# Output Checkpoint Settings
save_dir=${root}/checkpoints/${folder_struc} # checkpoint directory path
save_path=${save_dir}/val-P${valP}           # checkpoint file path
mkdir -p ${save_dir}

# Display Experiment Settings
display_exp_settings

# Main Execution
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} \
    python3 -m torch.distributed.launch \
    --nproc_per_node=${GPUS_PER_NODE} \
    --master_port=${MASTER_PORT} \
    ${root}/train.py \
    ${train_val_files} \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${pretrained_weights} \
    --reset-optimizer --reset-dataloader --reset-meters \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=ofa_${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay --lr=${lr} \
    --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
    --log-format=simple --log-interval=10 \
    --fixed-validation-seed=7 \
    --no-epoch-checkpoints --keep-best-checkpoints=1 \
    --save-interval=1 --validate-interval=1 \
    --save-interval-updates=500 --validate-interval-updates=500 \
    --eval-acc \
    --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
    --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --fp16 \
    --fp16-scale-window=512 \
    --tensorboard-logdir=${tensorboard_dir} \
    --num-workers=0 >${log_path} 2>&1
