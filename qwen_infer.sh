# nohup bash qwen_infer.sh > qwen_infer.log  2>&1 &

datasets=("QQP")
cases_nums=(1 0)

# datasets=("BQ" "MRPC" "LCQMC")
# cases_nums=(0 1 5)



batch_size=4

for dataset in ${datasets[@]}
do
  for cases_num in ${cases_nums[@]}
  do
    python qwen_infer.py \
        --dataset $dataset \
        --cases_num $cases_num \
        --batch_size $batch_size
  done
done