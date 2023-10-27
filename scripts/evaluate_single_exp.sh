#!/usr/bin/env bash

export MASTER_PORT=6036
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

# helper function
assert_file_exists () {
    if [ ! -f $1 ]; then
        echo "File '$1' not found!"
        exit
    fi
}

display_exp_settings () {
    echo -------------------------------- Experiment settings -------------------------------------
    echo Exp tag: ${exp_tag}
    echo Train P: ${trainP_list[@]}
    echo Val P : ${valP_list[@]}
    echo Will run ${#trainP_list[@]} x ${#valP_list[@]} = $(( ${#trainP_list[@]} * ${#valP_list[@]} )) jobs
    for trainP in ${trainP_list[@]}; do
        for valP in ${valP_list[@]}; do
            echo ${trainP}-${valP}
        done
    done
    echo ------------------------------------------------------------------------------------------
}
display_cur_exp () {
    echo ------------------------------- Curent Experiment settings -------------------------------
    echo Exp tag: ${exp_tag}
    echo Train P: ${trainP_list[@]}
    echo Val P : ${valP_list[@]}
    echo result file: ${result_dir}/${result_name}_predict.json
    echo ------------------------------------------------------------------------------------------
}


# basic setting
root=/home/P76104419/ICCV
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE
selected_cols=0,4,2,3
batch_size=24
beam=12


# experiment setting
exp_tag=manual
sp=manual-images
ckpt_version=checkpoint_best.pt
declare -a model_size_list=(
    tiny medium base large
)
declare -a trainP_list=(
    P5
)
declare -a valP_list=(
    P1
    # 'P4-w22' 'P4-w24' 'P4-w26' 'P4-w28' 'P4-w30' 'P4-w32' 'P4-w34' 'P4-w36' 'P4-w38' 'P4-w40'
    # 'P4-w42' 'P4-w44' 'P4-w46' 'P4-w48' 'P4-w50' 'P4-w52' 'P4-w54' 'P4-w56' 'P4-w58' 'P4-w60'
    # 'P4-w62' 'P4-w64' 'P4-w66' 'P4-w68' 'P4-w70' 'P4-w72' 'P4-w74' 'P4-w76' 'P4-w78' 'P4-w80'
    # 'P4-w82' 'P4-w84' 'P4-w86' 'P4-w88' 'P4-w90' 'P4-w92' 'P4-w94' 'P4-w96' 'P4-w98' 'P4-w100'

    # 'P5-w22' 'P5-w24' 'P5-w26' 'P5-w28' 'P5-w30' 'P5-w32' 'P5-w34' 'P5-w36' 'P5-w38' 'P5-w40'
    # 'P5-w42' 'P5-w44' 'P5-w46' 'P5-w48' 'P5-w50' 'P5-w52' 'P5-w54' 'P5-w56' 'P5-w58' 'P5-w60'
    # 'P5-w62' 'P5-w64' 'P5-w66' 'P5-w68' 'P5-w70' 'P5-w72' 'P5-w74' 'P5-w76' 'P5-w78' 'P5-w80'
    # 'P5-w82' 'P5-w84' 'P5-w86' 'P5-w88' 'P5-w90' 'P5-w92' 'P5-w94' 'P5-w96' 'P5-w98' 'P5-w100'
)



# for model_size in ${model_size_list[@]} ; do
    for trainP in ${trainP_list[@]} ; do
        for valP in ${valP_list[@]} ; do
            data=${root}/datasets/base64/vg/${exp_tag}/${sp}-${valP}.csv
            path=${root}/checkpoints/${trainP}/${ckpt_version}
            assert_file_exists ${data}
            assert_file_exists ${path}
            result_dir=${root}/results/vg/${exp_tag}/${trainP}
            result_name=${sp}-${valP}
            display_cur_exp

            CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 \
            -m torch.distributed.launch \
            --nproc_per_node=${GPUS_PER_NODE} \
            --master_port=${MASTER_PORT} \
            ${root}/evaluate.py \
                ${data} \
                --path=${path} \
                --user-dir=${user_dir} \
                --task=refcoco \
                --batch-size=${batch_size} \
                --log-format=simple --log-interval=10 \
                --seed=7 \
                --gen-subset=${result_name} \
                --results-path=${result_dir} \
                --beam=${beam} \
                --min-len=4 \
                --max-len-a=0 \
                --max-len-b=4 \
                --no-repeat-ngram-size=3 \
                --fp16 \
                --num-workers=0 \
                --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
        done
    done
# done

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
data=${train_data},${val_data}

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
    $data \
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
