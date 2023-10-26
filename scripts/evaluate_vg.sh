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