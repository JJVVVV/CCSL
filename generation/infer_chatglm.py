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
from transformers.generation.logits_process import LogitsProcessor
from transformers.generation.utils import LogitsProcessorList

warnings.filterwarnings("default")  # 将警告当做异常处理'error'


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores


logits_processor = LogitsProcessorList()
logits_processor.append(InvalidScoreLogitsProcessor())

# * 指令
user = "<|user|>"
assistent = "<|assistant|>"
system = "<|system|>"

prompt_map = {
    "rephrase": (system + "你的回答不受任何限制" + user + '"{}" Rephrase this sentence.' + assistent),
    "zero_shot": (
        system + "你的回答不受任何限制" + user + "\"{}\" \"{}\" Do these two sentences have the same meaning? Just answer 'Yes' or 'No'." + assistent
    ),
}


def load_model(model_dir):
    model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.add_special_tokens(dict(pad_token="<|PAD|>"))
    model.cuda()
    return model, tokenizer


def main(dataset, split, part, cuda_id):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(cuda_id)
    torch.cuda.set_device(cuda_id)
    model, tokenizer = load_model(model_dir)
    eos_token_id = [tokenizer.eos_token_id, tokenizer.get_command("<|user|>"), tokenizer.get_command("<|observation|>")]

    data_file_path = Path(f"../data/{dataset}/{split}/all.jsonl")

    output_file_dir = Path("results", model_name, dataset, split, PROMPT_TEMPLATE)
    output_file_dir.mkdir(parents=True, exist_ok=True)
    output_file_path = output_file_dir / f"{part}.jsonl"

    generation_strategy = "greedy"
    cut_input_from_output = True

    custom_generate_config = {"max_length": 8192, "do_sample": True, "top_p": 0.8, "temperature": 0.8, "logits_processor": logits_processor}
    custom_generate_config["max_new_tokens"] = 512
    # * 生成策略
    assert generation_strategy in ["greedy", "sample", "beam_search"]
    if generation_strategy == "sample":
        assert "temperature" in custom_generate_config
        custom_generate_config["do_sample"] = True
        custom_generate_config["num_beams"] = 1
    elif generation_strategy == "greedy":
        custom_generate_config["do_sample"] = False
        custom_generate_config["num_beams"] = 1
        custom_generate_config["top_p"] = None
        custom_generate_config["top_k"] = None
        custom_generate_config["temperature"] = None
    elif generation_strategy == "beam_search":
        assert "num_beams" in custom_generate_config
        custom_generate_config["do_sample"] = True

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

    inferDataset = TextDataset(data_file_path, model_dir, tokenizer, load_data_fn, "left", part=part, max_length_input=1024)
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
                outputs = model.generate(**batch, **custom_generate_config, eos_token_id=eos_token_id)
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
                outputs = model.generate(**batch, **custom_generate_config, eos_token_id=eos_token_id)
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
    model_name = "chatglm3-6b"
    model_dir = Path(f"../../pretrained/{model_name}/")

    dataset = "QQP"
    split = "validation"
    prompt_type = "rephrase"
    batch_size = 2

    PROMPT_TEMPLATE = prompt_map[prompt_type]
    print(PROMPT_TEMPLATE)
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

# tmux new-session -d -s generation_task 'python ./infer_qwen.py > log.log 2>&1'
