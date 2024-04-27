#!/bin/bash
# nohup ./auto.sh > auto.log 2>&1 &

except_cuda=(0)

while true; do
  gpu_info=$(nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits)
  free_gpu_arr=()
  while IFS=',' read -r idx used total; do
    # echo $used
    if (( used < 5000 )) && [[ ! " ${except_cuda[@]} " =~ " $idx " ]]; then  # 将条件改为判断显存小于1000M
      free_gpu_arr+=($idx)
    fi
  done <<< "$gpu_info"
  
  if [[ -n $free_gpu_arr ]]; then
    IFS="/" free_gpu="${free_gpu_arr[*]}"
    echo "Free GPU found: $free_gpu"
    # nohup ./trainScript_roberta_QQP.sh $free_gpu > /dev/null 2>&1 &
    nohup ./trainScript_macbert_LCQMC.sh $free_gpu > /dev/null 2>&1 &
    break
  else
    echo "No free GPU available"
  fi
  
  sleep 10
done
