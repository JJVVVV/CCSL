#!/bin/bash


# 结束GPU上的进程
pids=($(ps aux | grep ./[t]rain | grep -v 'killtrain.sh' | awk '{print $2}'))
for pid in ${pids[@]}
do
  # echo $(($pid))
  kill -9 $(($pid))
  if [ $? -eq 0 ]; then
    echo -e "\033[0;32m# succeed to kill $pid.\033[0m" 
  else
    echo -e "\033[0;31m# fail to kill $pid.\033[0m" 
  fi
done
exit 0