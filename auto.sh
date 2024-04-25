#!/bin/bash
# nohup ./auto.sh > auto.log 2>&1 &

except_cuda=(3)

while true; do
  gpu_info=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)
  free_gpu=""
  while IFS=',' read -r used total; do
    if (( used < 10000 )); then  # 将条件改为判断显存小于1000M
      free_gpu=$(nvidia-smi --query-gpu=index --format=csv,noheader)
      free_gpu_arr=($free_gpu)
      result=()
      for item in "${free_gpu_arr[@]}"
      do
          # 检查当前元素是否包含在数组 free_gpu_arr 中
          if [[ ! " ${except_cuda[@]} " =~ " $item " ]]; then
              # 如果不包含，则将元素添加到结果数组中
              result+=("$item")
          fi
      done
      IFS="/" free_gpu="${result[*]}"
      break
    fi
  done <<< "$gpu_info"
  
  if [[ -n $free_gpu ]]; then
    echo "Free GPU found: $free_gpu"
    nohup ./trainScript_macbert_LCQMC.sh $free_gpu > /dev/null 2>&1 &
    break
  else
    echo "No free GPU available"
  fi
  
  sleep 3
done
