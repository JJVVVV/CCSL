# Qwen
# nohup python qwen_infer.py QQP 0 8 > qwen_infer.log  2>&1 &
# nohup python qwen_infer.py QQP 0 > qwen_infer.log  2>&1 &

import random
from typing import Literal

import pandas as pd
from fire import Fire

from load_data_fns import key_map


class FewShotCasesGenerator:
    def __init__(self, dataset: str) -> None:
        df_train = pd.read_json(f"data/{dataset}/train/all.jsonl", lines=True)
        df_train[key_map[dataset][0]] = df_train[key_map[dataset][0]].str.replace(r"{|}", "", regex=True)
        df_train[key_map[dataset][1]] = df_train[key_map[dataset][1]].str.replace(r"{|}", "", regex=True)
        self.dataset = dataset
        self.df_train = df_train
        self.df_pos = df_train[df_train["label"] == 1].sample(frac=1, random_state=1)
        self.df_neg = df_train[df_train["label"] == 0].sample(frac=1, random_state=2)

    def get_cases(self, n: int, language: Literal["zh", "en"]):
        pos_num = n // 2 + (random.random() > 0.5)
        neg_num = n - pos_num
        pos_start = random.randint(0, len(self.df_pos) - 1 - pos_num)
        neg_start = random.randint(0, len(self.df_neg) - 1 - neg_num)

        df = pd.concat([self.df_pos[pos_start : pos_start + pos_num], self.df_neg[neg_start : neg_start + neg_num]], axis=0)
        df = df.sample(frac=1, random_state=0)
        if language == "zh":
            ret = "判断两个句子的含义是否相同，下面是几个例子：\n"
            for _, row in df.iterrows():
                ret += f"\"{row[key_map[self.dataset][0]]}\" \"{row[key_map[self.dataset][1]]}\" {'是' if row['label']==1 else '否'}\n"
            if n == 0:
                return ret + '现在，请判断下面这两个句子是否表达相同语义，只需回答"是"或"否"：\n'
            else:
                return ret + '现在，请判断下面这两个句子是否表达相同语义，只需回答"是"或"否"：\n'
        else:
            ret = "Determine whether two sentences express the same meaning, here are some examples:\n"
            for _, row in df.iterrows():
                ret += f"\"{row[key_map[self.dataset][0]]}\" \"{row[key_map[self.dataset][1]]}\" {'yes' if row['label']==1 else 'no'}\n"
            if n == 0:
                return ret + "Now, determine whether the following two sentences express the same meaning. Just answer 'yes' or 'no':\n"
            else:
                return ret + "Now, determine whether the following two sentences express the same meaning. Just answer 'yes' or 'no':\n"


def main(dataset, cases_num, batch_size: int | None = None):
    print("-----------------params-----------------")
    print("dataset:", dataset)
    print("cases_num: ", cases_num)
    print("batch_size:", batch_size)
    print("-----------------params-----------------")

    import json
    import re
    from pprint import pp

    import torch
    from toolkit.logger import getLogger
    from tqdm.auto import tqdm
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig

    logger = getLogger("QwenInfer", None)

    # device = torch.device("cuda:0")

    model_dir = "../../pretrained/Qwen/Qwen-14B-Chat/"
    # model_dir="/public/home/hongy/pretrained_models/Qwen-14B-Chat/"
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.add_special_tokens(dict(pad_token="<|PAD|>"))

    # use bf16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
    # use fp16
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", device_map="auto", trust_remote_code=True, fp16=True).eval()
    # use cpu only
    # model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-14B-Chat", device_map="cpu", trust_remote_code=True).eval()
    # use auto mode, automatically select precision based on the device.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True, bf16=True, use_flash_attn=False, device_map="auto"
    ).eval()
    model.tie_weights()

    # model = AutoModelForCausalLM.from_pretrained(
    #     model_dir, trust_remote_code=True, torch_dtype="auto", low_cpu_mem_usage=True, bf16=True, use_flash_attn=False,
    # ).eval()
    # model.cuda()

    # exit(1)
    # model.to(device)
    # response, history = model.chat(tokenizer, query="hello", history=None)
    tokenizer.im_start_id
    print(tokenizer.convert_ids_to_tokens(tokenizer.im_start_id))
    print(tokenizer.convert_ids_to_tokens(tokenizer.im_end_id))
    import os
    import warnings

    import jsonlines
    import pandas as pd
    from toolkit.nlp.data import TextDataset
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    # if "qwen-14b-chat" in model_dir:
    #     tokenizer.add_special_tokens(dict(pad_token="<|PAD|>"))

    warnings.filterwarnings("default")  # 将警告当做异常处理'error'

    model_name = "qwen-14b-chat"
    # dataset = "LCQMC"
    # dataset = "QQP"
    # split = "test"
    split = "val" if dataset == "QQP" else "test"
    part = "0-999999"
    # part = "120-124"
    # part = "0-4"
    # part = [25, 41, 54, 61, 192, 247, 475, 481, 704, 789, 1260, 1385]
    # part=[180, 181, 604, 2844, 2891]
    # cases_num = 10

    # prompt_type = "rephrase"
    # prompt_type = "compare"
    if cases_num == 0:
        prompt_type = "zero_shot"
    else:
        prompt_type = "few_shot"

    if batch_size is None:
        batch_size = 2 if prompt_type == "rephrase" else 1

    im_start, im_end = "<|im_start|>", "<|im_end|>"
    im_start_tokens = [tokenizer.im_start_id]
    im_end_tokens = [tokenizer.im_end_id]
    nl = "\n"
    nl_tokens = tokenizer.encode("\n")

    if prompt_type == "few_shot":
        case_gen = FewShotCasesGenerator(dataset)

    def get_prompt(dataset, prompt_type) -> str:
        match dataset, prompt_type:
            case "LCQMC" | "BQ", "zero_shot":
                return (
                    im_start
                    + "system\n"
                    + "You are a helpful assistant. Your answer is not subject to any restrictions."
                    + im_end
                    + nl
                    + im_start
                    +
                    # "user\n\"{}\" \"{}\" Do these two sentences have the same meaning? Just answer 'Yes' or 'No'."+
                    'user\n"{}" "{}" 这两个句子的含义是否相同？只需回答"是"或"否"'
                    + im_end
                    + nl
                    + im_start
                    + "assistant\n"
                )
            case "LCQMC" | "BQ", "few_shot":
                return (
                    im_start
                    + "system\n"
                    + "You are a helpful assistant. Your answer is not subject to any restrictions."
                    + im_end
                    + nl
                    + im_start
                    +
                    # "user\n\"{}\" \"{}\" Do these two sentences have the same meaning? Just answer 'Yes' or 'No'."+
                    "user\n"
                    + case_gen.get_cases(cases_num, "zh")
                    + '"{}" "{}"'
                    + im_end
                    + nl
                    + im_start
                    + "assistant\n"
                )
            case "QQP" | "MRPC", "zero_shot":
                return (
                    im_start
                    + "system\n"
                    + "You are a helpful assistant. Your answer is not subject to any restrictions."
                    + im_end
                    + nl
                    + im_start
                    # + "user\n\"{}\" \"{}\" Do these two sentences have the same meaning? Just answer 'Yes' or 'No'."
                    + "user\n\"{}\" \"{}\" Determine whether the two sentences express the same meaning. Just answer 'yes' or 'no'."
                    + im_end
                    + nl
                    + im_start
                    + "assistant\n"
                )
            case "QQP" | "MRPC", "few_shot":
                return (
                    im_start
                    + "system\n"
                    + "You are a helpful assistant. Your answer is not subject to any restrictions."
                    + im_end
                    + nl
                    + im_start
                    + "user\n"
                    + case_gen.get_cases(cases_num, "en")
                    + '"{}" "{}"'
                    + im_end
                    + nl
                    + im_start
                    + "assistant\n"
                )

    # import pdb; pdb.set_trace()

    generation_strategy = "greedy"
    cut_input_from_output = True
    from copy import deepcopy

    # print(PROMPT_TEMPLATE)
    from pathlib import Path

    from toolkit.enums import Split
    from toolkit.nlp.data import (
        ClassificationLabel,
        FinelyControlledText,
        PairedText,
        RegressionLabel,
    )
    from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

    gc = deepcopy(model.generation_config)
    gc.max_new_tokens = 10
    custom_generate_config = dict(temperature=0.1)

    # * 生成策略
    assert generation_strategy in ["greedy", "sample", "beam_search"]
    if generation_strategy == "sample":
        assert "temperature" in custom_generate_config
        gc.do_sample = True
        gc.num_beams = 1
        gc.temperature = custom_generate_config["temperature"]
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
        df = pd.read_json(data_file_path, lines=True)
        if "part" in kwargs:
            if isinstance(kwargs["part"], str):
                start, end = list(map(int, kwargs["part"].split("-")))
                df = df[start:end]
            elif isinstance(kwargs["part"], list):
                df = df.iloc[part]
        inputs = []
        labels = []
        customs = []
        if prompt_type == "rephrase":
            for idx, row in df.iterrows():
                a_sample1 = PairedText(get_prompt(dataset, prompt_type).format(row["question1"]))
                a_sample2 = PairedText(get_prompt(dataset, prompt_type).format(row["question2"]))
                inputs.extend((a_sample1, a_sample2))
                labels.append(ClassificationLabel(row["label"]))
                labels.append(ClassificationLabel(row["label"]))
                customs.append({"question": row["question1"]})
                customs.append({"question": row["question2"]})
        elif prompt_type == "compare":
            for idx, row in df.iterrows():
                inputs.append(PairedText(get_prompt(dataset, prompt_type).format(row["question1"], row["question2"])))
                labels.append(ClassificationLabel(row["label"]))
                customs.append({"question1": row["question1"], "question2": row["question2"]})
        elif "shot" in prompt_type:
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                # if dataset == "QQP" or dataset == "LCQMC":
                # try:
                prompt_template = get_prompt(dataset, prompt_type)
                a_sample = PairedText(prompt_template.format(row[key_map[dataset][0]], row[key_map[dataset][1]]))
                # except:
                #     print(prompt_template)
                #     print(row[key_map[dataset][0]], row[key_map[dataset][1]])
                #     import pdb; pdb.set_trace()
                inputs.append(a_sample)
                labels.append(ClassificationLabel(row["label"]))
                customs.append({key_map[dataset][0]: row[key_map[dataset][0]], key_map[dataset][1]: row[key_map[dataset][1]]})

        return inputs, labels, customs

    # * 准备
    if prompt_type == "few_shot":
        output_file_dir = os.path.join("results", model_name, dataset, prompt_type + "_" + str(cases_num))
    else:
        output_file_dir = os.path.join("results", model_name, dataset, prompt_type)

    data_file_path = f"data/{dataset}/{split}/all.jsonl"
    inferDataset = TextDataset(data_file_path, model_dir, "decoder", "generate", tokenizer, load_data_fn, part=part)
    inferDataset.padding_side = "left"
    inferDataset.padding_to_max_length = False
    # inferDataset = TextDataset(f"data/LCQMC//test/qwen_with_rephrase_clean_hardcases.jsonl", model_dir, tokenizer, load_data_fn, "left",max_length_input=512)
    print(inferDataset.texts_input[0][0])
    print(inferDataset.texts_input[1][0])
    print(inferDataset.texts_input[2][0])
    # print(inferDataset[0])

    dataloader = DataLoader(inferDataset, batch_size=batch_size, shuffle=False, collate_fn=inferDataset.collate_fn)

    ret = []
    labels = []
    cur_batch_idx = 0
    if prompt_type == "rephrase":
        for batch in tqdm(dataloader):
            try:
                custom_inputs = batch.pop("custom_inputs")
                # print(custom_inputs)
                labels = batch.pop("labels")
                batch = {key: value.to(model.device) for key, value in batch.items()}
                outputs = model.generate(
                    **batch, stop_words_ids=[[tokenizer.im_end_id], [tokenizer.im_start_id]], return_dict_in_generate=False, generation_config=gc
                )
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

    elif prompt_type == "zero_shot" or prompt_type == "few_shot" or prompt_type == "compare":
        for batch in tqdm(dataloader):
            try:
                custom_inputs = batch.pop("custom_inputs")
                labels = batch.pop("labels")
                batch = {key: value.to(model.device) for key, value in batch.items()}
                outputs = model.generate(
                    **batch, stop_words_ids=[[tokenizer.im_end_id], [tokenizer.im_start_id]], return_dict_in_generate=False, generation_config=gc
                )
                if cut_input_from_output:
                    texts = []
                    for idx, output in enumerate(outputs):
                        texts.append(tokenizer.decode(output[batch["input_ids"][idx].size(0) :], skip_special_tokens=True))
                else:
                    texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # messages = []
                # messages.append({"role":"user", "content": ""})
                output_batch = [
                    {key_map[dataset][0]: q1, key_map[dataset][1]: q2, "label": label.item(), "response": reason}
                    for q1, q2, label, reason in zip(custom_inputs[key_map[dataset][0]], custom_inputs[key_map[dataset][1]], labels, texts)
                ]
            except Exception as e:
                print(e)
                output_batch = [
                    {key_map[dataset][0]: q1, key_map[dataset][1]: q2, "label": label.item(), "response": "<Failure>"}
                    for q1, q2, label in zip(custom_inputs[key_map[dataset][0]], custom_inputs[key_map[dataset][1]], labels)
                ]
            ret.extend(output_batch)
            cur_batch_idx += 1
            if cur_batch_idx <= 10:
                logger.debug(json.dumps(output_batch, indent=2, ensure_ascii=False))

    print("Converting to dataframe...")
    df = pd.DataFrame(ret)
    print("Saving...")
    # os.makedirs(output_file_dir, exist_ok=True)
    os.makedirs(output_file_dir, exist_ok=True)
    output_file_path = os.path.join(output_file_dir, f"{split}_{part}.jsonl")
    df.to_json(output_file_path, orient="records", force_ascii=False, lines=True)


if __name__ == "__main__":
    Fire(main)
# for idx, row in df.iterrows():
#     print(row['question1'], row['question2'], row['label'], row['response'], sep='\n')
#     print('------------------------------------------------------------------')
# for idx, row in df.iterrows():
#     print(row['question1'], row['question2'], row['label'], row['rephrase1'],row['rephrase2'], sep='\n')
#     print('------------------------------------------------------------------')
# ## zero shot
# import re
# def to_pred(response: str):
#     if re.search('yes|是', response, re.IGNORECASE):
#         return 1
#     elif re.search('no|否', response, re.IGNORECASE):
#         return 0
#     else:
#         return response

# df['pred'] = df['response'].map(to_pred)
# fail_mask = ~((df['pred']==1) | (df['pred']==0))
# print(sum(fail_mask))
# df[fail_mask].head()
# df_filtered = df[~fail_mask]
# df_filtered['pred'] = df_filtered['pred'].astype(int)
# from sklearn.metrics import accuracy_score, f1_score
# print("acc: ", round(accuracy_score(df_filtered['label'], df_filtered['pred'])*100, 2))
# print("f1: ", round(f1_score(df_filtered['label'], df_filtered['pred'])*100, 2))
