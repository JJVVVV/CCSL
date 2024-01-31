#!/bin/bash

# nohup ./trainScript_roberta_MRPC_hardcase_seed.sh > /dev/null 2>&1 &
# pkill -s SIGKILL -pgn 3697090
# while kill -0 $PID 2>/dev/null; do sleep 1; done



# CUDA_VISIBLE_DEVICES=0/1/2/3/
CUDA_VISIBLE_DEVICES=0/1/2/3/4/5/6/7

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

all_times=(0.2 0.4 0.8 1 0)
# all_times=(1 0.6 0.4 0.2)
all_times=(0)
seeds_of_stage1=(59 13 43 71 56)
# seeds_of_stage1=(56)
seeds=(11 17 23 59 13 43 71 19)
seeds=(0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 11 17 23 59 13 43 71 19)
#  0 2 4 6 8 10 12 14 16 18 20 22 24 26 28 30 32 34 36 38 40 42 44 46 48 50 52 54 56 58 60 62 64 66 68 70 72 74 76 78 80 82 84 86 88 90 92 94 96 98)

for times in ${all_times[@]}
do
  # 遍历所有的种子
  for seed in ${seeds[@]}
  do
    for seed_of_stage1 in ${seeds_of_stage1[@]}
    do
      # ###################################parameters#########################################
      dashboard="None"
      dataset_name="MRPC"
      part="all"
      # text_type='ORI'
      text_type='DATA_AUG_REP4'

      min_threshold=None
      alpha=None

      model_type="roberta-base"

      # model_name="baseline_hardcases_warmboost_mix_easycases_negtimes=${times}/seed_of_stage1=$seed_of_stage1"
      # model_name="baseline_hardcases_warmboost_mix_easycases_negtimes=${times}_add_badcases/seed_of_stage1=$seed_of_stage1"

      # model_name="single_model_hardcases_warmboost_mix_easycases_negtimes=${times}_add_badcases/seed_of_stage1=$seed_of_stage1"
      # model_name="single_model_hardcases_from_baseline_warmboost_mix_easycases_totaltimes=${times}/seed_of_stage1=$seed_of_stage1"
      # model_name="single_model_hardcases_from_baseline_warmboost_mix_easycases_negtimes=${times}/seed_of_stage1=$seed_of_stage1"

      model_name="nodrop_single_model_hardcases_from_baseline_warmboost_mix_easycases_totaltimes=${times}/seed_of_stage1=$seed_of_stage1"
      model_name="nodrop_single_model_hardcases_from_baseline_warmboost_fix_num_ratio=${times}/seed_of_stage1=$seed_of_stage1"


      # model_name="s2m_multi_model_shareclassifier_hardcases_warmboost_mix_easycases_totaltimes=${times}/seed_of_stage1=$seed_of_stage1"
      # model_name="s2m_multi_model_shareclassifier_hardcases_from_baselineonly_badcases_only_badcases_warmboost_mix_easycases_totaltimes=${times}/seed_of_stage1=$seed_of_stage1"
      # model_name="s2m_multi_model_shareclassifier_hardcases_from_baseline_warmboost_mix_easycases_totaltimes=${times}/seed_of_stage1=$seed_of_stage1"


      # model_name="multi_model_shareclassifier_hardcases_warmboost_mix_easycases_negtimes=${times}/seed_of_stage1=$seed_of_stage1"
      # model_name="multi_model_shareclassifier_hardcases_warmboost_mix_easycases_negtimes=${times}_add_badcases/seed_of_stage1=$seed_of_stage1"

      # model_name="multi_model_shareclassifier_hardcases_warmboost_mix_easycases_totaltimes=${times}/seed_of_stage1=$seed_of_stage1"

      # model_name="single_model_hardcases_warmboost_mix_easycases_totaltimes=${times}_add_badcases/seed_of_stage1=$seed_of_stage1"
      # model_name="noise2_$min_threshold"
      # model_name="shift_only_$alpha"

      model_dir="../pretrained/$model_type"

      fp16=True
      test_in_epoch=True

      accumulate_step=1
      batch_size=16
      batch_size_infer=64
      epochs=1
      max_length_input=512
      learning_rate='2e-6'
      weight_decay=0.1
      metric='accuracy'

      # train_file_path="data/LCQMC/train/qwen_with_rephrase_clean_hardcases.jsonl"
      train_file_path=None
      val_file_path=None
      test_file_path=None

      warmup_ratio=0.1
      # ###################################parameters#########################################


      # if [[ $model_name == *"warmboost"* ]]; then
      #   model_dir="outputs/$dataset_name/$model_type/DATA_AUG_REP4/all/single_model/5/16/2e-05/$seed_of_stage1/optimal_checkpoint"
      # else
      #   model_dir="../pretrained/$model_type"
      # fi
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
      # HUGGINGFACE_HUB_CACHE="/data/jjwang/.cache/huggingface/hub/" TRANSFORMERS_CACHE="/data/jjwang/.cache/huggingface/hub/" \
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
            --save_last_ckpt False \
            --show_lr False \
            --show_step False \
            --cache_dataset True\
            --seed_of_stage1 $seed_of_stage1 \
            --times $times \
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
          --save_last_ckpt False \
          --logging_steps 1 \
          --show_lr False \
          --show_step False \
          --cache_dataset True \
          --seed_of_stage1 $seed_of_stage1 \
          --times $times \
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
  done
done

# 等待所有剩余的进程结束
wait ${pids[@]}