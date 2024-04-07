import fcntl
from enum import Enum, auto
from pathlib import Path

import numpy as np
import pandas as pd
from toolkit.enums import Split
from toolkit.nlp.data import ClassificationLabel, PairedText
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

key_map = {
    "LCQMC": ("question1", "question2"),
    "BQ": ("question1", "question2"),
    "QQP": ("question1", "question2"),
    "MRPC": ("sentence1", "sentence2"),
    "RTE": ("sentence1", "sentence2"),
    "QNLI": ("question", "sentence"),
}


class DatasetName(Enum):
    QQP = auto()
    SST2 = auto()
    MNLI = auto()
    QNLI = auto()
    MRPC = auto()
    RTE = auto()
    LCQMC = auto()
    BQ = auto()


class TextType(Enum):
    ANS = auto()
    DIS = auto()
    QUE_DIS = auto()
    QUE_ANS = auto()
    QUE_PSEUDO = auto()

    ORI = auto()
    JUST_DATA_AUG6 = auto()
    DATA_AUG_REP2 = auto()
    DATA_AUG_REP4 = auto()
    JUST_DATA_AUG_REP4 = auto()
    JUST_DATA_AUG_ORI = auto()
    DATA_AUG_REP4_FUSED = auto()
    DATA_AUG_REP4_CLOSED = auto()
    DATA_AUG_REP6 = auto()
    GAUSSIAN_LABEL = auto()
    SORTED_DATA = auto()
    CHECK_HAL = auto()
    CHECK_HAL2 = auto()
    # ONLY_POS = auto()


DATASET_CLASSNUM_MAP = {DatasetName.QQP: 2, DatasetName.MRPC: 2, DatasetName.LCQMC: 2, DatasetName.BQ: 2}


def get_sep_token_num(model_type):
    if "roberta" in model_type:
        return 2
    else:
        return 1


def gaussian(x, mu, sigma):
    return np.exp(-1 * ((x - mu) ** 2) / (2 * (sigma**2))) / (np.sqrt(2 * np.pi) * sigma)
    # return np.exp(-(x-mu)**2/(2*sigma**2))/(sigma*np.sqrt(2*np.pi))


def get_gaussian_label(mu, start, end, num, sigam=0.1):
    x = np.linspace(start, end, num)
    y = gaussian(x, mu, sigam)
    y = y * 1 / num
    y = y / y.sum()
    return y


def get_soft_label(label, sigam=0.1):
    if label == 0:
        soft_label = np.random.normal(0.25, sigam)
        if soft_label >= 0.5:
            soft_label -= 0.5
        elif soft_label < 0:
            soft_label += 0.5
    elif label == 1:
        soft_label = np.random.normal(0.75, sigam)
        if soft_label > 1:
            soft_label -= 0.5
        elif soft_label <= 0.5:
            soft_label += 0.5
    return soft_label


def load_data_fn_qqp_lcqmc_bq_mrpc(
    data_file_path: Path | str, model_type: str, tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast, split: Split, **kwargs
):
    special_tokens_map = tokenizer.special_tokens_map
    BOS = special_tokens_map["bos_token"] if "bos_token" in special_tokens_map.keys() else None
    EOS = special_tokens_map["eos_token"] if "eos_token" in special_tokens_map.keys() else None
    SEP = special_tokens_map["sep_token"] if "sep_token" in special_tokens_map.keys() else None
    MASK = special_tokens_map["mask_token"] if "mask_token" in special_tokens_map.keys() else None
    CLS = special_tokens_map["cls_token"] if "cls_token" in special_tokens_map.keys() else None
    sep_num = get_sep_token_num(model_type)
    text_type = kwargs["text_type"]
    key1, key2 = key_map[kwargs["dataset_name"].name]

    if isinstance(data_file_path, Path | str):
        with open(data_file_path) as f:
            fcntl.flock(f, fcntl.LOCK_SH)
            if text_type == TextType.SORTED_DATA and split == Split.TRAINING:
                sorted_df = pd.read_json(data_file_path)
                fcntl.flock(f, fcntl.LOCK_UN)
                sorted_df = sorted_df.reset_index(drop=True)
                # df = df.iloc[df["label"].to_list().index(1) :]
                # df = sorted_df.iloc[sorted_df["label"].to_list().index(1) : sorted_df[sorted_df["dif"] > 0.3].index[0]]
                # df = sorted_df.iloc[sorted_df[sorted_df["dif"] > 0.1].index[0] : sorted_df[sorted_df["dif"] > 0.4].index[0]]
                # df = sorted_df.iloc[sorted_df[sorted_df["dif"] > 0.2].index[0] : sorted_df[sorted_df["dif"] > 0.5].index[0]]
                # df = sorted_df.iloc[sorted_df[sorted_df["dif"] > 0.05].index[0] : sorted_df[sorted_df["dif"] > 0.45].index[0]]
                df = sorted_df.iloc[sorted_df[sorted_df["dif"] > 0.01].index[0] : sorted_df[sorted_df["dif"] > 0.5].index[0]]  # 2
            else:
                df = pd.read_json(data_file_path, lines=True)
                fcntl.flock(f, fcntl.LOCK_UN)
                # with jsonlines.open(data_file_path, "r") as jlReader:
                #     dict_objs = list(jlReader)
                #     if isinstance(dict_objs[0], str):
                #         dict_objs = dict_objs[1:]
                # df = pd.DataFrame(dict_objs)
    else:
        df = data_file_path

    inputs = []
    labels = []
    for _, row in df.iterrows():
        # input
        match text_type:
            case TextType.CHECK_HAL:
                inputs.append(PairedText(row[key1], row["rephrase1"]))
                inputs.append(PairedText(row[key2], row["rephrase2"]))
            case TextType.CHECK_HAL2:
                inputs.append(PairedText(row["rephrase1"], row["rephrase2"]))
            case TextType.ORI | TextType.GAUSSIAN_LABEL | TextType.SORTED_DATA:
                inputs.append(PairedText(row[key1], row[key2]))
                # inputs.append(((False, CLS), (True, dict_obj[key1]), (False, SEP), (True, dict_obj[key2]), (False, SEP)))
            case TextType.JUST_DATA_AUG_ORI:
                if split == Split.TRAINING:
                    inputs.extend(
                        [
                            PairedText(t1, t2)
                            for t1, t2 in zip(
                                [row[key1], row[key1], row["rephrase1"], row["rephrase1"]], [row[key2], row["rephrase2"], row[key2], row["rephrase2"]]
                            )
                        ]
                    )
                else:
                    a_sample = PairedText(row[key1], row[key2])
                    inputs.append(a_sample)
            case TextType.JUST_DATA_AUG_REP4:
                if split == Split.TRAINING:
                    inputs.extend(
                        [
                            PairedText(t1, t2)
                            for t1, t2 in zip(
                                [row[key1], row[key1], row["rephrase1"], row["rephrase1"]], [row[key2], row["rephrase2"], row[key2], row["rephrase2"]]
                            )
                        ]
                    )
                else:
                    a_sample = PairedText(
                        [row[key1], row[key1], row["rephrase1"], row["rephrase1"]], [row[key2], row["rephrase2"], row[key2], row["rephrase2"]]
                    )
                    inputs.append(a_sample)
            case TextType.JUST_DATA_AUG6:
                if split == Split.TRAINING:
                    inputs.extend(
                        [
                            PairedText(t1, t2)
                            for t1, t2 in zip(
                                [row[key1], row[key1], row["rephrase1"], row["rephrase1"], row[key1], row[key2]],
                                [row[key2], row["rephrase2"], row[key2], row["rephrase2"], row["rephrase1"], row["rephrase2"]],
                            )
                        ]
                    )
                else:
                    inputs.append(PairedText(row[key1], row[key2]))
            case TextType.DATA_AUG_REP2:
                tmp = ([row[key1], row["rephrase1"]], [row[key2], row["rephrase2"]])
                inputs.append(tmp)
            case TextType.DATA_AUG_REP4 | TextType.DATA_AUG_REP4_FUSED | TextType.DATA_AUG_REP4_CLOSED:
                a_sample = PairedText(
                    [row[key1], row[key1], row["rephrase1"], row["rephrase1"]], [row[key2], row["rephrase2"], row[key2], row["rephrase2"]]
                )
                inputs.append(a_sample)
            case TextType.DATA_AUG_REP6:
                tmp = (
                    [row[key1], row[key1], row["rephrase1"], row["rephrase1"], row[key1], row[key2]],
                    [row[key2], row["rephrase2"], row[key2], row["rephrase2"], row["rephrase1"], row["rephrase2"]],
                )
                inputs.append(tmp)
        # label
        match text_type, split:
            case (TextType.GAUSSIAN_LABEL, Split.TRAINING):
                soft_label = get_soft_label(label=row["label"], sigam=0.1)
                labels.append(get_gaussian_label(soft_label, start=0, end=1, num=100, sigam=0.1))
            case (TextType.JUST_DATA_AUG6, Split.TRAINING):
                labels.extend([ClassificationLabel(row["label"])] * 4 + [ClassificationLabel(1)] * 2)
            case (TextType.JUST_DATA_AUG_REP4 | TextType.JUST_DATA_AUG_ORI, Split.TRAINING):
                labels.extend([ClassificationLabel(row["label"]) for _ in range(4)])
            case (TextType.CHECK_HAL, _):
                labels.append([1])
                labels.append([1])
            case _:
                labels.append([row["label"]])

    return inputs, labels


LOAD_DATA_FNS = {
    DatasetName.QQP: load_data_fn_qqp_lcqmc_bq_mrpc,
    DatasetName.MRPC: load_data_fn_qqp_lcqmc_bq_mrpc,
    DatasetName.LCQMC: load_data_fn_qqp_lcqmc_bq_mrpc,
    DatasetName.BQ: load_data_fn_qqp_lcqmc_bq_mrpc,
}
