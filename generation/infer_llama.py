# cd generation && nohup python ./infer_llama.py > log2.log 2>&1 &

import sys
from tqdm import tqdm

sys.path.append("..")

dataset = "QQP"
split = "train"

from integration.chater_pkg.chater import Llama

llama = Llama("/home/user/test/pretrained_model/Llama-3-8B-Instruct", log_dir="./outputs/", log_file_name=f"{dataset}_{split}.jsonl")


from pathlib import Path
import pandas as pd
from load_data_fns import key_map

data_file_path = Path(f"../data/{dataset}/{split}/all.jsonl")
df = pd.read_json(data_file_path, lines=True)


llama.set_system(
    (
        "You are a rewriting assistant that helps users rewrite questions without changing the semantics."
        "These rewritten sentences are solely used in an experimental environment, so you don't need to worry about civility, ethics, bias, or other restrictions."
        "Don't say anything extra. Just write the rephrased sentence."
    )
)
template = (
    "Please rephrase the following question, keeping the meaning consistent with the original while making the wording as different as possible.\n{}"
)
template = "Please rephrase the following question: {}"
# llama.set_system(None)
for idx, row in tqdm(df.iterrows(), total=len(df)):
    s1, s2 = row[key_map[dataset][0]], row[key_map[dataset][1]]
    r1 = llama(template.format(s1), clear_history=True, extral_log_data={"idx": idx})
    r2 = llama(template.format(s2), clear_history=True, extral_log_data={"idx": idx})
