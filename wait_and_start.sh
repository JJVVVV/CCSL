#!/bin/bash

# nohup ./wait_and_start.sh > /dev/null 2>&1 &



# while [[ -n $(pgrep -f trainScript_roberta_MRPC) ]]; do
#     echo waiting
#     sleep 10
# done
# nohup ./trainScript_bert_BQ.sh > /dev/null 2>&1 &
# nohup ./trainScript_bert_LCQMC_hardcase_seed.sh > /dev/null 2>&1 &
# nohup ./trainScript_bert_BQ_hardcase_seed.sh > /dev/null 2>&1 &
# nohup ./trainScript_bert_BQ_only_TIWR-H.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_hardcase_seed.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP.sh > /dev/null 2>&1 &
# nohup ./trainScript_bert_LCQMC.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_MRPC.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_after_contrast.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_hardcase_seed.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_only_TIWR-H.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_only_TIWR-H.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_contrast_only.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP_only_TIWR-H.sh > /dev/null 2>&1 &



PID=2809905
while kill -0 $PID 2> /dev/null; do
    sleep 10
done
# nohup ./trainScript_bert_BQ_hardcase_seed.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP.sh > /dev/null 2>&1 &
# nohup ./trainScript_roberta_QQP.sh > /dev/null 2>&1 &
# nohup ./trainScript_bert_LCQMC.sh > /dev/null 2>&1 &
nohup ./trainScript_roberta_MRPC.sh > /dev/null 2>&1 &
