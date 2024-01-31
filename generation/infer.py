import os
import warnings
from copy import deepcopy
from multiprocessing import Pool
from pathlib import Path

# import jsonlines
import pandas as pd
import torch
from toolkit.enums import Split
from toolkit.nlp.data import ClassificationLabel, PairedText, TextDataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, PreTrainedTokenizerFast

warnings.filterwarnings("default")  # 将警告当做异常处理'error'


# * 指令
sp_token_map = {"assistant_token_id": "<reserved_107>", "user_token_id": "<reserved_106>"}
# PROMPT_TEMPLATE = f"{sp_token_map['user_token_id']}" '"{}"\n' '"{}"\n' "这两个问题之间有什么{}之处?" f"{sp_token_map['assistant_token_id']}"
# PROMPT_TEMPLATE = (
#     # "你是一个AI助手, 下面是一段关于你与用户之间的对话.\n"
#     "用户:\n"
#     '"{}"\n'
#     '"{}"\n'
#     "这两个问题之间有什么{}之处?"
#     "\nAI助手:\n"
# )

prompt_map = {"rephrase": (f"{sp_token_map['user_token_id']}" '"{}"\n' "Rephrase the sentence." f"{sp_token_map['assistant_token_id']}")}


def main(dataset, split, part, cuda_id):
    torch.cuda.set_device(cuda_id)
    model_name = "baichuan2-chat"
    model_dir = Path("../../pretrained/baichuan2-13b-chat/")
    data_file_path = Path(f"../data/{dataset}/{split}/all.jsonl")

    output_file_dir = Path("results", model_name, dataset, split, PROMPT_TEMPLATE)
    output_file_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_file_dir / f"{part}.jsonl"

    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model.cuda()

    generation_strategy = "greedy"
    cut_input_from_output = True

    gc = deepcopy(model.generation_config)
    gc.max_new_tokens = 256
    custom_generate_config = dict()

    # * 生成策略
    assert generation_strategy in ["greedy", "sample", "beam_search"]
    if generation_strategy == "sample":
        assert "temperature" in custom_generate_config
        gc.do_sample = True
        gc.num_beams = 1
    elif generation_strategy == "greedy":
        gc.do_sample = False
        gc.num_beams = 1
        gc.top_p = None
        gc.top_k = None
        gc.temperature = None
    elif generation_strategy == "beam_search":
        assert "num_beams" in custom_generate_config
        gc.do_sample = True

    def load_data_fn(data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs):
        PROMPT_TEMPLATE = prompt_map[prompt_type]
        start, end = list(map(int, kwargs["part"].split("-")))
        df = pd.read_json(data_file_path, lines=True)
        df = df[start:end]
        inputs = []
        labels = []
        customs = []
        if prompt_type == "rephrase":
            for idx, row in df.iterrows():
                a_sample1 = PairedText(PROMPT_TEMPLATE.format(row["question1"]))
                a_sample2 = PairedText(PROMPT_TEMPLATE.format(row["question2"]))
                inputs.extend((a_sample1, a_sample2))
                labels.append(ClassificationLabel(row["label"]))
                labels.append(ClassificationLabel(row["label"]))
                customs.append({"question": row["question1"]})
                customs.append({"question": row["question2"]})
        elif prompt_type == None:
            for idx, row in df.iterrows():
                if dataset == "LCQMC":
                    a_sample = PairedText(PROMPT_TEMPLATE.format(row["question1"], row["question2"], "不同" if row["label"] == 0 else "相同"))
                    # a_sample = PairedText(PROMPT_TEMPLATE.format(row['question1'], row['question2']))
                elif dataset == "QQP":
                    a_sample = PairedText(
                        PROMPT_TEMPLATE.format(row["question1"], row["question2"], "difference" if row["label"] == 0 else "similarity")
                    )
                inputs.append(a_sample)
                labels.append(ClassificationLabel(row["label"]))
                customs.append({"question1": row["question1"], "question2": row["question2"]})

        return inputs, labels, customs

    inferDataset = TextDataset(data_file_path, model_dir, tokenizer, load_data_fn, "left", part=part)
    print(inferDataset.texts_input[0])
    # print(inferDataset[0])

    dataloader = DataLoader(inferDataset, batch_size=batch_size, shuffle=False, collate_fn=inferDataset.collate_fn)

    ret = []
    labels = []
    if prompt_type == "rephrase":
        for batch in tqdm(dataloader):
            try:
                custom_inputs = batch.pop("custom_inputs")
                # print(custom_inputs)
                labels = batch.pop("labels")
                batch = {key: value.cuda() for key, value in batch.items()}
                outputs = model.generate(**batch, generation_config=gc)
                if cut_input_from_output:
                    texts = []
                    for idx, output in enumerate(outputs):
                        texts.append(tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                else:
                    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # messages = []
                # messages.append({"role":"user", "content": ""})
                ret.extend(
                    [
                        {"question1": q1, "question2": q2, "label": label.item(), "rephrase1": r1, "rephrase2": r2}
                        for q1, q2, label, r1, r2 in zip(
                            custom_inputs["question"][::2], custom_inputs["question"][1::2], labels[::2], texts[::2], texts[1::2]
                        )
                    ]
                )
            except:
                ret.extend(
                    [
                        {"question1": q1, "question2": q2, "label": label.item(), "rephrase1": "<Failure>", "rephrase2": "<Failure>"}
                        for q1, q2, label in zip(custom_inputs["question"][::2], custom_inputs["question"][1::2], labels[::2])
                    ]
                )

    elif prompt_type == None:
        for batch in tqdm(dataloader):
            try:
                custom_inputs = batch.pop("custom_inputs")
                labels = batch.pop("labels")
                batch = {key: value.cuda() for key, value in batch.items()}
                outputs = model.generate(**batch, generation_config=gc)
                if cut_input_from_output:
                    texts = []
                    for idx, output in enumerate(outputs):
                        texts.append(tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                else:
                    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # messages = []
                # messages.append({"role":"user", "content": ""})
                ret.extend(
                    [
                        {"question1": q1, "question2": q2, "label": label.item(), "reason": reason}
                        for q1, q2, label, reason in zip(custom_inputs["question1"], custom_inputs["question2"], labels, texts)
                    ]
                )
            except:
                ret.extend(
                    [
                        {"question1": q1, "question2": q2, "label": label.item(), "reason": "<Failure>"}
                        for q1, q2, label in zip(custom_inputs["question1"], custom_inputs["question2"], labels)
                    ]
                )

    df = pd.DataFrame(ret)
    df.to_json(output_file_path, orient="records", force_ascii=False, lines=True)


def split_data(x, n):
    quotient, remainder = divmod(x, n)  # 计算每一份的基础值和剩余的单位数
    return [(quotient + 1 if i < remainder else quotient) for i in range(n)]


if __name__ == "__main__":
    dataset = "QQP"
    split = "validation"
    prompt_type = "rephrase"
    batch_size = 2

    PROMPT_TEMPLATE = prompt_map[prompt_type]
    data_file_path = Path(f"../data/{dataset}/{split}/all.jsonl")
    df = pd.read_json(data_file_path, lines=True)
    n = len(df)
    # n = 4

    n_proc = 8
    ends = split_data(n, n_proc)
    for i in range(1, n_proc):
        ends[i] += ends[i - 1]
    ends = [0] + ends
    parts = [f"{start}-{end}" for start, end in zip(ends[:-1], ends[1:])]
    print(parts)
    tasks = [(dataset, split, part, cuda_id) for part, cuda_id in zip(parts, range(n_proc))]
    with Pool(n_proc) as p:
        p.starmap(main, tasks)

# tmux new-session -d -s generation_task 'python ./infer.py > log.log 2>&1'
