#!/usr/bin/env bash

export MASTER_PORT=8085
export CUDA_VISIBLE_DEVICES=0,1,2
export GPUS_PER_NODE=3

selected_cols=0,5,2,3,4
root=/home/P76104419/ICCV
user_dir=${root}/ofa_module
bpe_dir=${root}/utils/BPE

seed=8
batch_size=20

for split in 'train' 'test_public'; do
    for model_size in 'large'; do
        data=${root}/dataset/base64/vqa/${split}.csv
        path=${root}/pretrained_weights/ofa_${model_size}.pt
        result_path=${root}/results/vqa/${model_size}-${seed}

        CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ${root}/evaluate.py \
            ${data} \
            --path=${path} \
            --user-dir=${user_dir} \
            --task=vqa_gen \
            --selected-cols=${selected_cols} \
            --bpe-dir=${bpe_dir} \
            --patch-image-size=512 \
            --prompt-type='none' \
            --batch-size=${batch_size} \
            --log-format=simple --log-interval=10 \
            --seed=${seed} \
            --gen-subset=${split} \
            --results-path=${result_path} \
            --fp16 \
            --zero-shot \
            --beam=12 \
            --unnormalized \
            --temperature=1.0 \
            --num-workers=0
    done
done