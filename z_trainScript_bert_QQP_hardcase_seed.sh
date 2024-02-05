#!/bin/bash

# nohup ./trainScript_bert_QQP_hardcase_seed.sh > /dev/null 2>&1 &
# pkill -s SIGKILL -pgn 3697090

# 定义一个数组，存放种子
# while kill -0 $PID 2>/dev/null; do sleep 1; done


# seeds_of_stage1=(36 60 98 70 95)
# all_neg_times=(4 3 2 1 0.5 0.33 0.25 None)
all_neg_times=(0.33 3)
# all_neg_times=(2)
seeds_of_stage1=(11 39 29 103 37)
#  64 77 47 41 66
seeds=(11 13 17 11 39 29 103 37)
# 31 32 33 34 35 36 37 38
for neg_times in ${all_neg_times[@]}
do
  for seed_of_stage1 in ${seeds_of_stage1[@]}
    do
    # seed_of_stage1=36
    # CUDA_VISIBLE_DEVICES=0/1/2/3/
    CUDA_VISIBLE_DEVICES=0/1/2/3/4/5/6/7

    # ###################################parameters#########################################
    dashboard="None"
    dataset_name="QQP"
    part="all"
    # text_type='ORI'
    text_type='DATA_AUG_REP4'

    min_threshold=None
    alpha=None

    model_type="bert-base-uncased"
    # model_type="chinese-macbert-base"
    # model_type="bert-large-uncased"
    # model_type='hfl/chinese-bert-wwm-ext'

    # model_name="single_model"
    # model_name="multi_model"
    model_name="{$seed_of_stage1}_single_model_hardcases_warmboost"
    model_name="{$seed_of_stage1}_single_model_hardcases"
    model_name="{$seed_of_stage1}_single_model_hardcases_warmboost_mix_easycases_negtimes=$neg_times"
    # model_name="single_model_hardcases"
    # model_name="baseline_hardcases"
    # model_name="noise2_$min_threshold"
    # model_name="shift_only_$alpha"

    fp16=True
    test_in_epoch=True

    accumulate_step=1
    batch_size=16
    batch_size_infer=64
    epochs=3
    max_length_input=512
    learning_rate='2e-5'
    weight_decay=0.01
    metric='accuracy'

    # train_file_path="data/LCQMC/train/qwen_with_rephrase_clean_hardcases.jsonl"
    train_file_path=None
    val_file_path=None
    test_file_path=None

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
      if [[ $model_name == *"warmboost"* ]]; then
        model_dir="outputs/$dataset_name/$model_type/DATA_AUG_REP4/all/single_model/3/16/2e-05/$seed_of_stage1/optimal_checkpoint"
      else
        model_dir="../pretrained/$model_type"
      fi
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
            --save_last_ckpt False \
            --show_lr False \
            --show_step False \
            --cache_dataset True\
            --seed_of_stage1 $seed_of_stage1 \
            --neg_times $neg_times \
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
          --neg_times $neg_times \
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
  done
done