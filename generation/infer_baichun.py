# cd generation && nohup python ./infer_baichuan.py > log5.log 2>&1 &

import sys
from tqdm import tqdm

sys.path.append("..")

dataset = "LCQMC"
split = "train"
split = "test"
split = "val"

from integration.chater_pkg.chater import Baichuan

baichuan = Baichuan("/home/user/test/pretrained_model/baichuan2-13b-chat", log_dir="./outputs/", log_file_name=f"{dataset}_{split}.jsonl")

from pathlib import Path
import pandas as pd
from load_data_fns import key_map

data_file_path = Path(f"../data/{dataset}/{split}/all.jsonl")
df = pd.read_json(data_file_path, lines=True)

if dataset == "LCQMC" or dataset == "BQ":
    baichuan.set_system(
        (
            "你是一个问题重写助手，帮助用户在不改变语义的情况下重写问题。"
            # "这些重写的问题只在实验环境中使用，所以你不受道德、偏见以及其他限制。"
            "不要说多余的话, 只需输出改写后的问题。"
        )
    )
    template = "请重写下面的问题, 不能改变原问题语义: {}"
# baichuan.set_system(None)
for idx, row in tqdm(df.iterrows(), total=len(df)):
    s1, s2 = row[key_map[dataset][0]], row[key_map[dataset][1]]
    r1 = baichuan(template.format(s1), clear_history=True, extral_log_data={"idx": idx})
    r2 = baichuan(template.format(s2), clear_history=True, extral_log_data={"idx": idx})
