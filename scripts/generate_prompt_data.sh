root=$(dirname "$(dirname "$(readlink -f "$0")")")
vqa_results_dir=${root}/results/vqa

raw_file=${root}/datasets/official/train_sample.csv
exp_tag=example

answer_file_keys=(
    ofa_huge
    ofa_large
)
answer_file_paths=(
    ${vqa_results_dir}/tiny/official/train_sample.csv
    ${vqa_results_dir}/tiny/official/train_sample.csv
)
prompt_name=Instruct-2
output_file=${root}/datasets/${exp_tag}/train_sample-P${prompt_name}.csv

# ================================================================================
# Please do not change the settings below
# ================================================================================
answer_file_str=""
for i in "${!answer_file_keys[@]}"; do
    answer_file_str="${answer_file_str}'${answer_file_keys[$i]}':'${answer_file_paths[$i]}', "
done

python3 ${root}/prompt.py \
    --raw_file ${raw_file} \
    --answer_files "{${answer_file_str}}" \
    --prompt_name ${prompt_name} \
    --output_file ${output_file}
