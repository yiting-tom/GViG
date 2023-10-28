root=$(dirname "$(dirname "$(readlink -f "$0")")")

raw_file=${root}/datasets/official/train_sample.csv
answer_file_keys=(
    ofa_huge
    ofa_large
)
vqa_results_dir=${root}/results/vqa
answer_file_paths=(
    ${vqa_results_dir}/tiny/sample/train_sample.csv
    ${vqa_results_dir}/tiny/sample/train_sample.csv
)
prompt_name=Base-2

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
    --prompt_name ${prompt_name}
