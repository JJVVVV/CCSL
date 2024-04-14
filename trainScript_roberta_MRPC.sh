#!/bin/bash

# nohup ./trainScript_roberta_MRPC.sh > /dev/null 2>&1 &


seeds=(59 13 43 71 56)
seeds=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)
seeds=(30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60)

seeds=(13 14 20 37 9)
seeds=(0 1 2 3 4 5 6 7 8 10 11 12 15 16 17 18 19 21 22 23 24 25)
seeds=(30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60)

CUDA_VISIBLE_DEVICES=1/0
# CUDA_VISIBLE_DEVICES=5/6/7

# ###################################parameters#########################################
model_structure="encoder"
task_type="classify"
dashboard="None"
dataset_name="MRPC"
part="all"
text_type='ORI'
text_type='DATA_AUG_REP4'
# text_type='JUST_DATA_AUG_ORI'

min_threshold=None
alpha=None

model_type="roberta-base"
model_dir="../../pretrained/$model_type"
# model_dir="../../pretrained_models/$model_type"

model_name="Baseline_nodrop_baseline"
model_name="TIWR_nodrop_single_model"
model_name="TWR_nodrop_single_model"


fp16=True
test_in_epoch=True

accumulate_step=1
if [[ $text_type == "JUST_DATA_AUG"* ]]; then
  batch_size=64
  # batch_size=16
else
  batch_size=16
fi
batch_size_infer=64
epochs=3
max_length_input=512
learning_rate='2e-5'
weight_decay=0.1
metric='accuracy'

if [[ $model_name == *"nodrop"* ]]; then
  train_file_path="data/$dataset_name/train/qwen_with_rephrase_clean_nodrop.jsonl"
  val_file_path="data/$dataset_name/val/qwen_with_rephrase_clean_nodrop.jsonl"
  test_file_path="data/$dataset_name/test/qwen_with_rephrase_clean_nodrop.jsonl"
else
  if [[ $model_name == *"correct"* ]]; then
    train_file_path="data/$dataset_name/train/qwen_with_rephrase_clean_correct.jsonl"
  else
    train_file_path="data/$dataset_name/train/qwen_with_rephrase_clean.jsonl"
  fi
  val_file_path="data/$dataset_name/val/qwen_with_rephrase_clean.jsonl"
  test_file_path="data/$dataset_name/test/qwen_with_rephrase_clean.jsonl"
fi

warmup_ratio=0.1
# ###################################parameters#########################################

# 定义一个数组，存放可用cuda
# IFS=',' cudas=($CUDA_VISIBLE_DEVICES) IFS=' '
IFS='/' cudas=($CUDA_VISIBLE_DEVICES) IFS=' '
# 计算每个每个任务可用cuda数量
IFS=',' nproc_pre_node=(${cudas[0]}) IFS=' '
nproc_pre_node=${#nproc_pre_node[@]}
# 定义一个变量，表示最大并行数
parallel=${#cudas[@]}
# 定义一个数组，存放当前运行的进程号
pids=()
# 定义一个字典, 记录PID运行在哪个CUDA设备上
declare -A pid_cuda


# 遍历所有的种子
for seed in ${seeds[@]}
do
  # 判断有无console目录, 没有则创建
  log_file="console/$dataset_name-$text_type-$model_type-$model_name-$epochs-$batch_size-$learning_rate-$seed.ansi.log"
  log_dir=${log_file%/*}
  if [ ! -d log_dir ]; then
    mkdir -p $log_dir
  fi
  # 如果当前运行的进程数达到最大并行数，就等待任意一个进程结束: 从数组pids中删除结束进程的PID, 释放一个CUDA
  if [ ${#pids[@]} -eq $parallel ]; then
    wait -n ${pids[@]}
    # 删除已经结束的进程号, 释放一个可用的cuda
    for pid in ${pids[@]}
    do
      if ! ps -p $pid > /dev/null ; then
        # echo $pid
        finishedPID=$pid
        break
      fi
    done
    echo "finishPID: $finishedPID"
    pids=(${pids[@]/$finishedPID/})
    cudas+=(${pid_cuda[$finishedPID]})
    echo "freeCUDA: ${pid_cuda[$finishedPID]}"
    unset pid_cuda[$finishedPID]
    echo "runningProcesses: ${pids[@]}"
    echo "avaliableCUDAs: ${cudas[@]}"
    echo
  fi
  # 启动一个新训练任务: 使用一个可用的cuda,并把它的PID添加到数组pids中
  cuda=${cudas[0]}
  unset cudas[0]
  cudas=(${cudas[@]})
  # sed -i "s/seed=.*/seed=$seed/" ./trainScript.sh
  # sed -i "s/CUDA_VISIBLE_DEVICES=[0-9|,]*/CUDA_VISIBLE_DEVICES=$cuda/" ./trainScript.sh
  # ./trainScript.sh > "console//seed-$seed.log" 2>&1 &
  # ###################################训练程序#########################################
  # TORCH_DISTRIBUTED_DEBUG=INFO \
  if [ $nproc_pre_node -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$cuda \
    torchrun \
      --rdzv-backend=c10d \
      --rdzv-endpoint=localhost:0 \
      --nnodes=1 \
      --nproc-per-node=$nproc_pre_node \
      ./train_trainer.py \
        --dataset_name $dataset_name \
        --model_type $model_type \
        --model_name $model_name \
        --train_file_path $train_file_path \
        --val_file_path $val_file_path \
        --test_file_path $test_file_path \
        --seed $seed \
        --opt_lr $learning_rate \
        --epochs $epochs \
        --train_batch_size $batch_size \
        --infer_batch_size $batch_size_infer \
        --sch_warmup_ratio_steps $warmup_ratio \
        --max_length_input $max_length_input \
        --metric $metric \
        --eval_every_half_epoch $test_in_epoch \
        --gradient_accumulation_steps $accumulate_step \
        --fp16 $fp16 \
        --opt_weight_decay $weight_decay \
        --dashboard $dashboard \
        --text_type $text_type \
        --min_threshold $min_threshold \
        --alpha $alpha \
        --part $part \
        --model_dir $model_dir \
        --parallel_mode DDP \
        --save_ckpts False \
        --save_last_ckpt False \
        --show_lr False \
        --show_step False \
        --cache_dataset True \
        --record_cheat False \
        --model_structure $model_structure \
        --task_type $task_type \
        > $log_file 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$cuda \
    python ./train_trainer.py \
      --dataset_name $dataset_name \
      --model_type $model_type \
      --model_name $model_name \
      --train_file_path $train_file_path \
      --val_file_path $val_file_path \
      --test_file_path $test_file_path \
      --seed $seed \
      --opt_lr $learning_rate \
      --epochs $epochs \
      --train_batch_size $batch_size \
      --infer_batch_size $batch_size_infer \
      --sch_warmup_ratio_steps $warmup_ratio \
      --max_length_input $max_length_input \
      --metric $metric \
      --eval_every_half_epoch $test_in_epoch \
      --gradient_accumulation_steps $accumulate_step \
      --fp16 $fp16 \
      --opt_weight_decay $weight_decay \
      --dashboard $dashboard \
      --text_type $text_type \
      --min_threshold $min_threshold \
      --alpha $alpha \
      --part $part \
      --model_dir $model_dir \
      --save_ckpts False \
      --save_last_ckpt False \
      --logging_steps 1 \
      --show_lr False \
      --show_step False \
      --cache_dataset True \
      --record_cheat False \
      --model_structure $model_structure \
      --task_type $task_type \
      > $log_file 2>&1 &
  fi
    # --fp16 \
    # --batch_size $(($batch_size/$nproc_pre_node)) \
    # --pretrained_model_path $pretrained_model_path \
    # --early_stop \
    # --early_stop_metric $early_stop_metric \
    # --continue_train_more_patience \
    # --max_length $max_length \
    # --continue_train_more_epochs \
    # --continue_train_more_patience \
          # --do_lower_case \
  # ###################################训练程序#########################################
  newPID=$!
  pids+=($newPID)
  pid_cuda[$newPID]=$cuda
  echo "newPID: $newPID"
  echo "useCUDA: ${pid_cuda[$newPID]}"
  echo "runningProcesses: ${pids[@]}"
  echo "avaliableCUDAs: ${cudas[@]}"
  echo

  # while [ ! -f "console/seed-$seed.log" ]; do
  #   echo "waiting trainScript.sh to run in the background."
  #   sleep 1
  # done

done
# 等待所有剩余的进程结束
wait ${pids[@]}