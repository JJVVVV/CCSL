import fcntl
import json
import re
import time
from collections import OrderedDict
from pathlib import Path

import deepspeed
import hjson
import numpy as np
import pandas as pd
import toolkit
import torch

# import wandb
from fire import Fire
from load_data_fns import DATASET_CLASSNUM_MAP, LOAD_DATA_FNS, DatasetName, TextType, key_map
from model.MatchModel_binary_classification import (  # BertModelFirst,; RobertaModel_4times_4classifier_bi,; RobertaModel_6times_bi,
    BertModel_binary_classify,
    BertModel_rephrase,
    BertModel_rephrase_auxloss,
    BertModel_rephrase_auxloss_sep,
    BertModel_rephrase_contrast_only,
    BertModel_rephrase_IWR,
    BertModel_rephrase_just_data_aug,
    BertMultiModel_rephrase,
    BertMultiModel_rephrase_share_classifier,
    BertMultiModel_rephrase_share_classifier_auxloss,
    BertMultiModel_rephrase_share_classifier_contrast_only,
    RobertaModel_binary_classify,
    RobertaModel_rephrase,
    RobertaModel_rephrase_auxloss,
    RobertaModel_rephrase_close,
    RobertaModel_rephrase_contrast_only,
    RobertaModel_rephrase_IWR,
    RobertaModel_rephrase_just_data_aug,
    RobertaModel_rephrase_zh,
    RobertaMultiModel_rephrase,
    RobertaMultiModel_rephrase_fused,
    RobertaMultiModel_rephrase_share_classifier,
    RobertaMultiModel_rephrase_withfused,
)
from model.MatchModel_multi_classification import BertModel_multi_classify, RobertaModel_multi_classify
from sklearn.metrics import accuracy_score, f1_score
from toolkit import getLogger
from toolkit.enums import Split
from toolkit.metric import MetricDict
from toolkit.nlp import NLPTrainingConfig, TextDataset
from toolkit.training import CheckpointManager, Evaluator, Trainer, WatchDog
from toolkit.training.initializer import allocate_gpu_memory, initialize
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast
from utils.evaluate import Evaluator1


def get_contro_confused_definite_cases(split="TEST"):
    seed_of_stage1 = configs.seed if not hasattr(configs, "seed_of_stage1") else configs.seed_of_stage1
    if "hardcases_from_baseline" in configs.model_name and split == "TRAINING":
        step = WatchDog.load(baseline_model_dir / str(seed_of_stage1) / "optimal_checkpoint").best_checkpoint[1]
        results_dir = baseline_model_dir / str(seed_of_stage1) / "evaluator" / f"step={step}" / f"{split if split!='TRAINING' else 'ANY'}.json"
    else:
        step = WatchDog.load(stage1_model_dir / str(seed_of_stage1) / "optimal_checkpoint").best_checkpoint[1]
        results_dir = stage1_model_dir / str(seed_of_stage1) / "evaluator" / f"step={step}" / f"{split if split!='TRAINING' else 'ANY'}.json"
    logger.info("results_dir: " + str(results_dir))
    if not results_dir.exists():
        if split == "TRAINING":
            # split = "train"
            # model_path = f"outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/single_model/5/16/2e-05/{seed}/optimal_checkpoint"
            # model_path = "outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/multi_model/5/16/2e-05/4/optimal_checkpoint"
            model_infer_hardcases = baseline_model_dir if "hardcases_from_baseline" in configs.model_name else stage1_model_dir
            model_path = str(model_infer_hardcases / str(seed_of_stage1) / "optimal_checkpoint")
            # model_type = model_path.split("/")[2]
            # dataset_name = model_path.split("/")[1]
            # data_file_path = 'data/QQP/validation/vicuna/Rephrase_the_following_question_and_keep_the_meaning_the_same:/all.jsonl'
            config = NLPTrainingConfig.load(model_path)
            config.text_type = "DATA_AUG_REP4"
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            # dataset = TextDataset(data_file_path, model_type, tokenizer, LOAD_DATA_FNS[DatasetName.QQP], text_type=TextType.DATA_AUG_REP4)
            dataset = TextDataset.from_file(
                config.train_file_path,
                tokenizer,
                split=Split.TRAINING,
                configs=config,
                load_data_fn=LOAD_DATA_FNS[DatasetName[config.dataset_name]],
                text_type=TextType[config.text_type],
                dataset_name=DatasetName[config.dataset_name],
            )
            if "multi" in model_path:
                if "roberta" in model_path:
                    if DATASETNAME.name in ["LCQMC", "BQ"]:
                        pass
                    if "shareclassifier" in model_path:
                        model = RobertaMultiModel_rephrase_share_classifier.from_pretrained(model_path, False)
                    elif "withfused" in model_path:
                        model = RobertaMultiModel_rephrase_withfused.from_pretrained(model_path, False)
                    else:
                        model = RobertaMultiModel_rephrase.from_pretrained(model_path, False)
                else:
                    if "shareclassifier" in model_path:
                        model = BertMultiModel_rephrase_share_classifier.from_pretrained(model_path, False)
                    elif "withfused" in model_path:
                        model = BertMultiModel_rephrase_withfused.from_pretrained(model_path, False)
                    else:
                        model = BertMultiModel_rephrase.from_pretrained(model_path, False)
            elif "single" in model_path or "baseline" in model_path:
                if "roberta" in model_path:
                    model = RobertaModel_rephrase.from_pretrained(model_path)
                else:
                    model = BertModel_rephrase.from_pretrained(model_path)
            config.batch_size_infer = 100
            Evaluator1.confused_use_ot = False
            Evaluator1.save_results = True
            evaluator = Evaluator1(
                "classify", Split.ANY, config=config, model=model, tokenizer=tokenizer, dataset=dataset, extral_args_evaluation={"is_train": False}
            )
            metric_dict, bad_cases, good_cases_idxs, controversial_cases, confused_cases, definite_cases, all_logits, all_labels = evaluator.eval()
            return controversial_cases, confused_cases, definite_cases
        else:
            return None
    with open(results_dir, "r") as f:
        metric_dict, bad_cases, good_cases_idxs, controversial_cases, confused_cases, definite_cases = json.load(f)
    return controversial_cases, confused_cases, definite_cases


def create_hardcases_data_file(split="TEST"):
    seed_of_stage1 = configs.seed if not hasattr(configs, "seed_of_stage1") else configs.seed_of_stage1
    if split == "TRAINING" and "mix_easycases" in configs.model_name:
        data_file_dir = stage1_model_dir / str(seed_of_stage1) / "optimal_checkpoint" / "hardcases" / f"{split}_mix_easycases_{configs.times}.jsonl"
    else:
        data_file_dir = stage1_model_dir / str(seed_of_stage1) / "optimal_checkpoint" / "hardcases" / f"{split}.jsonl"
    print(data_file_dir, data_file_dir.is_file())
    # if data_file_dir.is_file():
    #     return data_file_dir

    if (seperate_cases := get_contro_confused_definite_cases(split)) is not None:
        controversial_cases, confused_cases, definite_cases = seperate_cases
    else:
        logger.debug(f"No {split} data.")
        return None

    hard_cases = dict()
    hard_cases.update(controversial_cases)
    # if not (split == "TRAINING" and "hardcases_from_baseline" in configs.model_name):
    hard_cases.update(confused_cases)
    hard_cases = OrderedDict(sorted(hard_cases.items(), key=lambda x: x[0]))
    df = []
    idxs = []
    for key, value in hard_cases.items():
        idxs.append(key)
        q1 = value["text"][0][0]
        q2 = value["text"][1][0]
        r1 = value["text"][0][-1]
        r2 = value["text"][1][-1]
        label = value["labels"]
        pred = value["pred"]
        df.append(
            {key_map[DATASETNAME.name][0]: q1, key_map[DATASETNAME.name][1]: q2, "rephrase1": r1, "rephrase2": r2, "label": label, "pred": pred}
        )
    df = pd.DataFrame(df)
    df.index = idxs
    if split == "TRAINING" and "hardcases_from_baseline" in configs.model_name:
        if "only_badcases" in configs.model_name:
            print("Only use the badcases of all hardcases")
            df = df[df["label"] != df["pred"]]
        else:
            print("Balance the hardcases because they are from baseline ")
            count = df["label"].value_counts()
            neg_num, pos_num = count[0], count[1]
            t = min(neg_num, pos_num)
            cut_df = pd.concat(
                [
                    df[df["label"] == 1].sample(t, random_state=configs.seed_of_stage1),
                    df[df["label"] == 0].sample(t, random_state=configs.seed_of_stage1),
                ],
                axis=0,
            )
            df = cut_df
    print(f"All hardcases used to {split.lower()}: ")
    print(count := df["label"].value_counts())
    neg_num, pos_num = count[0], count[1]
    hardcases_num = len(df)
    print(round(count[1] / len(df) * 100), "%  : ", round(count[0] / len(df) * 100), "%")

    # if split=="TRAINING" and "correct_"
    if split == "TRAINING":
        if "mix_easycases" in configs.model_name or "add_badcases" in configs.model_name or "fix_num" in configs.model_name:
            definite_cases = OrderedDict(sorted(definite_cases.items(), key=lambda x: x[0]))
            df_easy = []
            df_bad = []
            idxs_easy = []
            idxs_bad = []

            for key, value in definite_cases.items():
                q1 = value["text"][0][0]
                q2 = value["text"][1][0]
                r1 = value["text"][0][-1]
                r2 = value["text"][1][-1]
                label = value["labels"]
                if label == value["pred"]:
                    idxs_easy.append(key)
                    df_easy.append(
                        {key_map[DATASETNAME.name][0]: q1, key_map[DATASETNAME.name][1]: q2, "rephrase1": r1, "rephrase2": r2, "label": label}
                    )
                else:
                    idxs_bad.append(key)
                    df_bad.append(
                        {key_map[DATASETNAME.name][0]: q1, key_map[DATASETNAME.name][1]: q2, "rephrase1": r1, "rephrase2": r2, "label": label}
                    )
            df_easy = pd.DataFrame(df_easy)
            df_easy.index = idxs_easy
            df_bad = pd.DataFrame(df_bad)
            df_bad.index = idxs_bad

        if "add_badcases" in configs.model_name:
            df_withbad = pd.concat([df, df_bad], axis=0)
            df_withbad.sort_index(inplace=True)
            df = df_withbad
            print("After add badcases: ")
            print(count := df["label"].value_counts())
            print(round(count[1] / len(df) * 100), "%  : ", round(count[0] / len(df) * 100), "%")

        if "mix_easycases" in configs.model_name and configs.times is not None:
            times = configs.times
            if "negtimes" in configs.model_name:
                count = df_easy["label"].value_counts()
                max_neg_num, max_pos_num = count[0], count[1]
                target_neg_num = pos_num * times
                dif_num = target_neg_num - neg_num
                extral_neg = min(dif_num if dif_num >= 0 else 0, max_neg_num)
                extral_pos = min(int(abs(neg_num // times - pos_num)) if dif_num < 0 else 0, max_pos_num)
                df_mix = pd.concat(
                    [
                        df,
                        df_easy[df_easy["label"] == 1].sample(extral_pos, random_state=configs.seed, replace=False),
                        df_easy[df_easy["label"] == 0].sample(extral_neg, random_state=configs.seed, replace=False),
                    ],
                    axis=0,
                )
                df_mix = df_mix.sample(frac=1, random_state=0)
            elif "totaltimes" in configs.model_name and configs.times is not None:
                easy_case_num = min(round(times * hardcases_num), len(df_easy))
                if easy_case_num > 0:
                    df_mix = pd.concat([df, df_easy.sample(easy_case_num, random_state=configs.seed, replace=False)], axis=0)
                else:
                    df_mix = df
            df = df_mix
            print("After add easycases: ")
            print(count := df["label"].value_counts())
            print(round(count[1] / len(df) * 100), "%  : ", round(count[0] / len(df) * 100), "%")

        if "fix_num" in configs.model_name and configs.times is not None:
            total = len(df)
            num_hard = int(total * configs.times)
            num_easy = min(total - num_hard, len(df_easy))
            df = pd.concat(
                [df.sample(num_hard, random_state=configs.seed, replace=False), df_easy.sample(num_easy, random_state=configs.seed, replace=False)],
                axis=0,
            )
            print("After mix part of hardcases and easycases: ")
            print(count := df["label"].value_counts())
            print(round(count[1] / len(df) * 100), "%  : ", round(count[0] / len(df) * 100), "%")
    return df
    # logger.debug(f"Saving {split} data file ...")
    # data_file_dir.parent.mkdir(parents=True, exist_ok=True)
    # with open(data_file_dir, "w") as f:
    #     fcntl.flock(f, fcntl.LOCK_EX)
    #     df.to_json(data_file_dir, force_ascii=False, lines=True, orient="records")
    #     f.flush()
    #     fcntl.flock(f, fcntl.LOCK_UN)
    # logger.debug(f"Saved.")
    # return data_file_dir


def load_dataset(tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast) -> tuple:
    # * Load training data, development data and test data
    # if "hardcases" in configs.model_name:
    #     configs.train_file_path = create_hardcases_data_file("TRAINING")
    #     configs.val_file_path = create_hardcases_data_file("VALIDATION")
    #     configs.test_file_path = create_hardcases_data_file("TEST")

    train_dataset = TextDataset.from_file(
        configs.train_file_path if "hardcases" not in configs.model_name else create_hardcases_data_file("TRAINING"),
        tokenizer,
        split=Split.TRAINING,
        configs=configs,
        load_data_fn=LOAD_DATA_FNS[DATASETNAME],
        text_type=TEXTTYPE,
        dataset_name=DATASETNAME,
        use_cache=("hardcases" not in configs.model_name),
    )
    # print(train_dataset[0]['model_input']['input_ids'].shape)
    # print(train_dataset[0]['model_input']['attention_mask'].shape)
    # print(train_dataset[0]['model_input']['token_type_ids'].shape)
    try:
        val_dataset = TextDataset.from_file(
            (
                configs.val_file_path
                if ("hardcases" not in configs.model_name or "TIWR-H" in configs.model_name)
                else create_hardcases_data_file("VALIDATION")
            ),
            tokenizer,
            split=Split.VALIDATION,
            configs=configs,
            load_data_fn=LOAD_DATA_FNS[DATASETNAME],
            text_type=TEXTTYPE,
            dataset_name=DATASETNAME,
            use_cache=("hardcases" not in configs.model_name or "TIWR-H" in configs.model_name),
        )
    except TypeError as e:
        if local_rank == 0:
            logger.warning(e)
        val_dataset = None
    try:
        test_dataset = TextDataset.from_file(
            (
                configs.test_file_path
                if ("hardcases" not in configs.model_name or "TIWR-H" in configs.model_name)
                else create_hardcases_data_file("TEST")
            ),
            tokenizer,
            split=Split.TEST,
            configs=configs,
            load_data_fn=LOAD_DATA_FNS[DATASETNAME],
            text_type=TEXTTYPE,
            dataset_name=DATASETNAME,
            use_cache=("hardcases" not in configs.model_name or "TIWR-H" in configs.model_name),
        )
    except TypeError as e:
        if local_rank == 0:
            logger.warning(e)
        test_dataset = None
    return train_dataset, val_dataset, test_dataset


class _Evaluator1(Evaluator):
    def calculate_metric_callback(self, all_labels: list, all_logits: list, mean_loss: float) -> MetricDict:
        # print(all_logits)
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)

        if "contrast_only" in configs.model_name or "auxloss_sep" in configs.model_name:
            return MetricDict(loss=mean_loss)

        match DATASET_CLASSNUM_MAP[DATASETNAME]:
            case 2:
                if "DATA_AUG_REP" in TEXTTYPE.name and "DATA_AUG_REP4_FUSED" != TEXTTYPE.name:
                    # all_logits: (num, 4)
                    all_ori_preds = (all_logits > 0).astype(int)
                    threshold = all_ori_preds.shape[1] >> 1
                    vote_pos = all_ori_preds.sum(axis=1)
                    all_preds = np.zeros_like(vote_pos)
                    pos_mask = vote_pos > threshold
                    neg_mast = ~pos_mask
                    controversial_mask = np.zeros_like(pos_mask).astype(bool) if all_ori_preds.shape[1] & 1 else vote_pos == threshold
                    all_preds[pos_mask] = 1
                    all_preds[neg_mast] = 0
                    # if controversial, then use original text
                    all_preds[controversial_mask] = all_ori_preds[controversial_mask][:, 0]
                    definite_mask = (vote_pos == all_ori_preds.shape[1]) | (vote_pos == 0)
                    confused_mask = ~(definite_mask | controversial_mask)
                    # # if confused, then use original text
                    # all_preds[confused_mask] = all_ori_preds[confused_mask][:, 0]
                elif TEXTTYPE == TextType.GAUSSIAN_LABEL:
                    # all_logtis: (num, 100)
                    all_preds = (np.argmax(all_logits, axis=1, keepdims=True) >= 50).astype(int)
                else:
                    # all_logtis: (num, 1)
                    all_preds = (all_logits > 0).astype(int)
            case _:
                all_preds = np.argmax(all_logits, axis=1, keepdims=True)

        if DATASET_CLASSNUM_MAP[DATASETNAME] == 2:
            if "hardcases" in self.config.model_name:
                controversial_cases, confused_cases, definite_cases = get_contro_confused_definite_cases(self.split.name)
                if "TIWR-H" in configs.model_name:
                    labels = all_labels.reshape(-1)
                    preds = all_preds.reshape(-1)
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average="binary")
                else:
                    labels = [d["labels"] for d in definite_cases.values()]
                    preds = [d["pred"] for d in definite_cases.values()]
                    logger.debug(
                        f"all_labels={len(all_labels)}, controversial_cases={len(controversial_cases)}, confused_cases={len(confused_cases)}"
                    )
                    logger.debug(len(controversial_cases) + len(confused_cases))
                    assert len(all_labels) == len(confused_cases) + len(controversial_cases)

                    labels.extend(all_labels.reshape(-1))
                    preds.extend(all_preds.reshape(-1))
                    acc = accuracy_score(labels, preds)
                    f1 = f1_score(labels, preds, average="binary")

                    stage1_hardcases_labels = [d["labels"] for d in controversial_cases.values()] + [d["labels"] for d in confused_cases.values()]
                    stage1_hardcases_preds = [d["pred"] for d in controversial_cases.values()] + [d["pred"] for d in confused_cases.values()]
                    stage1_all_labels = stage1_hardcases_labels + [d["labels"] for d in definite_cases.values()]
                    stage1_all_preds = stage1_hardcases_preds + [d["pred"] for d in definite_cases.values()]
                    logger.info(f"{self.split.name} hardcases acc stage1: {accuracy_score(stage1_hardcases_labels, stage1_hardcases_preds)*100:.2f}")
                    logger.info(f"{self.split.name} hardcases acc stage2: {accuracy_score(all_labels, all_preds)*100:.2f}")
                    logger.info(
                        f"{self.split.name} hardcases f1 stage1: {f1_score(stage1_hardcases_labels, stage1_hardcases_preds, average='binary')*100:.2f}"
                    )
                    logger.info(f"{self.split.name} hardcases f1 stage2: {f1_score(all_labels, all_preds, average='binary')*100:.2f}")

                    logger.info(f"{self.split.name} acc stage1: {accuracy_score(stage1_all_labels, stage1_all_preds)*100:.2f}")
                    logger.info(f"{self.split.name} acc stage2: {acc*100:.2f}")
                    logger.info(f"{self.split.name} f1 stage1: {f1_score(stage1_all_labels, stage1_all_preds, average='binary')*100:.2f}")
                    logger.info(f"{self.split.name} f1 stage2: {f1*100:.2f}")
            else:
                acc = accuracy_score(all_labels, all_preds)
                f1 = f1_score(all_labels, all_preds, average="binary")
        else:
            acc = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds, average="micro")

        if "DATA_AUG_REP" in TEXTTYPE.name and "DATA_AUG_REP4_FUSED" != TEXTTYPE.name:
            logger.info(f"--------------{self.split.name}---------------------")
            logger.info("Consistent:")
            mask = definite_mask
            logger.info(f"{mask.sum()}/{mask.size}  {mask.sum()/mask.size*100:.2f}%")
            part_acc = accuracy_score(y_true=all_labels[mask], y_pred=all_preds[mask])
            part_f1 = f1_score(y_true=all_labels[mask], y_pred=all_preds[mask])
            logger.info(f"acc: {part_acc * 100:.2f}%\tf1: {part_f1 * 100:.2f}%\n")
            logger.info("Inconsistent: ")
            mask = controversial_mask | confused_mask
            logger.info(f"{mask.sum()}/{mask.size}  {mask.sum()/mask.size*100:.2f}%")
            part_acc = accuracy_score(y_true=all_labels[mask], y_pred=all_preds[mask])
            part_f1 = f1_score(y_true=all_labels[mask], y_pred=all_preds[mask])
            logger.info(f"acc: {part_acc * 100:.2f}%\tf1: {part_f1 * 100:.2f}%\n")
            logger.info("----------------------------------------------------------------")
            for des, mask in zip(("controversial", "confused", "definite"), [controversial_mask, confused_mask, definite_mask]):
                if mask.sum() == 0:
                    continue
                logger.info(des + ":")
                logger.info(f"{mask.sum()}/{mask.size}  {mask.sum()/mask.size*100:.2f}%")
                part_acc = accuracy_score(y_true=all_labels[mask], y_pred=all_preds[mask])
                part_f1 = f1_score(y_true=all_labels[mask], y_pred=all_preds[mask])
                logger.info(f"acc: {part_acc * 100:.2f}%\tf1: {part_f1 * 100:.2f}%\n")
            logger.info(f"+++++++++++++++{self.split.name}++++++++++++++++++++")

            bad_cases = dict()
            controversial_cases = dict()
            definite_cases = dict()
            confused_cases = dict()
            text_type = TextType[self.config.text_type]

            for i in range(len(self.dataset)):
                if all_preds[i] != all_labels[i]:
                    bad_cases[i] = {"text": self.dataset.texts_input[i].tolist(), "pred": all_preds[i].item(), "labels": all_labels[i].item()}
                if "DATA_AUG_REP" in text_type.name:
                    if controversial_mask.size and controversial_mask[i]:
                        controversial_cases[i] = {
                            "text": self.dataset.texts_input[i].tolist(),
                            "pred": all_preds[i].item(),
                            "labels": all_labels[i].item(),
                            "ori_labels": all_ori_preds[i].tolist(),
                        }
                    if confused_mask.size and confused_mask[i]:
                        confused_cases[i] = {
                            "text": self.dataset.texts_input[i].tolist(),
                            "pred": all_preds[i].item(),
                            "labels": all_labels[i].item(),
                            "ori_labels": all_ori_preds[i].tolist(),
                        }
                    if definite_mask.size and definite_mask[i]:
                        definite_cases[i] = {
                            "text": self.dataset.texts_input[i].tolist(),
                            "pred": all_preds[i].item(),
                            "labels": all_labels[i].item(),
                            "ori_labels": all_ori_preds[i].tolist(),
                        }
            good_cases_idxs = set(range(len(self.dataset))) - set(bad_cases.keys())
            metric_dict = MetricDict({"accuracy": acc * 100, "F1-score": f1 * 100, "loss": mean_loss})

            file_path: Path = self.config.save_dir / "evaluator" / f"step={self.config.training_runtime['cur_step']}" / (self.split.name + ".json")
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as f:
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    data = json.dumps(
                        [dict(metric_dict), bad_cases, list(good_cases_idxs), controversial_cases, confused_cases, definite_cases], ensure_ascii=False
                    )
                    f.write(data)
                    f.flush()
                    fcntl.flock(f, fcntl.LOCK_UN)
                except IOError:
                    print("⚠️ Skip this operation because other programs are writing files ...")
            return metric_dict
        metric_dict = MetricDict({"accuracy": acc * 100, "F1-score": f1 * 100, "loss": mean_loss})
        return metric_dict


def load_model() -> tuple[PreTrainedModel | DDP, PreTrainedTokenizer | PreTrainedTokenizerFast, int]:
    # * Determine the model architecture
    global DATASETNAME, TEXTTYPE
    match DATASETNAME:
        case DatasetName.QQP | DatasetName.MRPC | DatasetName.LCQMC | DatasetName.BQ:
            if "roberta" in configs.model_type:
                match TEXTTYPE:
                    case TextType.ORI:
                        if "baseline" in configs.model_name:
                            MatchModel = RobertaModel_binary_classify
                    case TextType.SORTED_DATA:
                        MatchModel = RobertaModel_binary_classify
                    case TextType.JUST_DATA_AUG6:
                        MatchModel = RobertaModel_binary_classify
                    case TextType.DATA_AUG_REP2:
                        MatchModel = RobertaModel_rephrase
                    case TextType.DATA_AUG_REP4:
                        if "single_model" in configs.model_name:
                            if DATASETNAME == DatasetName.LCQMC or DATASETNAME == DatasetName.BQ:
                                MatchModel = RobertaModel_rephrase_zh
                            else:
                                if "contrast_only" in configs.model_name:
                                    MatchModel = RobertaModel_rephrase_contrast_only
                                elif "auxloss" in configs.model_name:
                                    MatchModel = RobertaModel_rephrase_auxloss
                                else:
                                    MatchModel = RobertaModel_rephrase
                        elif "multi_model" in configs.model_name:
                            if "shareclassifier" in configs.model_name:
                                MatchModel = RobertaMultiModel_rephrase_share_classifier
                            elif "withfused" in configs.model_name:
                                MatchModel = RobertaMultiModel_rephrase_withfused
                            else:
                                MatchModel = RobertaMultiModel_rephrase
                        elif "contrast_only" in configs.model_name:
                            MatchModel = RobertaModel_rephrase_contrast_only
                        elif "IWR" in configs.model_name:
                            MatchModel = RobertaModel_rephrase_IWR
                    case TextType.DATA_AUG_REP4_FUSED:
                        MatchModel = RobertaMultiModel_rephrase_fused
                    case TextType.DATA_AUG_REP4_CLOSED:
                        MatchModel = RobertaModel_rephrase_close
                    case TextType.JUST_DATA_AUG_REP4:
                        MatchModel = RobertaModel_rephrase_just_data_aug
                    case TextType.JUST_DATA_AUG_ORI:
                        MatchModel = RobertaModel_binary_classify
            elif "bert" in configs.model_type:
                if TEXTTYPE == TextType.ORI:
                    if "baseline" in configs.model_name:
                        MatchModel = BertModel_binary_classify
                if TEXTTYPE == TextType.DATA_AUG_REP4:
                    if "single_model" in configs.model_name:
                        if "contrast_only" in configs.model_name:
                            MatchModel = BertModel_rephrase_contrast_only
                        elif "auxloss_sep" in configs.model_name:
                            MatchModel = BertModel_rephrase_auxloss_sep
                        elif "auxloss" in configs.model_name:
                            MatchModel = BertModel_rephrase_auxloss
                        else:
                            MatchModel = BertModel_rephrase
                    elif "multi_model" in configs.model_name:
                        if "shareclassifier" in configs.model_name:
                            if "contrast_only" in configs.model_name:
                                MatchModel = BertMultiModel_rephrase_share_classifier_contrast_only
                            elif "auxloss" in configs.model_name:
                                MatchModel = BertMultiModel_rephrase_share_classifier_auxloss
                            else:
                                MatchModel = BertMultiModel_rephrase_share_classifier
                        elif "withfused" in configs.model_name:
                            MatchModel = BertMultiModel_rephrase_withfused
                        else:
                            MatchModel = BertMultiModel_rephrase
                    elif "IWR" in configs.model_name:
                        MatchModel = BertModel_rephrase_IWR
                if TEXTTYPE == TextType.JUST_DATA_AUG_REP4:
                    MatchModel = BertModel_rephrase_just_data_aug
                if TEXTTYPE == TextType.JUST_DATA_AUG_ORI:
                    MatchModel = BertModel_rephrase_just_data_aug
    logger.info(f"local_rank {local_rank}: {str(MatchModel)}")

    # * Determine the model path
    if ckpt_manager.latest_id == -1:
        pretrainedModelDir = configs.model_dir if configs.model_dir is not None else configs.model_type
    else:
        pretrainedModelDir = ckpt_manager.latest_dir
    logger.info(f"local_rank {local_rank}: load model from {pretrainedModelDir}")

    # * Load model, tokenizer to CPU memory
    # TODO deepspeed 加载模型
    # model.load_checkpoint(load_dir=load_checkpoint_dir)
    logger.debug(f"local_rank {local_rank}: Loading model and tokenizer to CPU memory...")
    start = time.time()
    # 加载自定义配置
    my_config = None
    try:
        my_config = AutoConfig.from_pretrained(f"config/my_{configs.model_type}_config")
        logger.debug(str(my_config))
    except:
        pass
    # match MatchModel:
    #     case BertModel_multi_classify:
    #         model = MatchModel.from_pretrained(
    #             pretrainedModelDir, config=my_config, local_files_only=False, num_classification=configs.num_classification
    #         )
    #     case BertModel_multi_classify_noise:
    #         model = MatchModel.from_pretrained(
    #             pretrainedModelDir,
    #             config=my_config,
    #             local_files_only=False,
    #             num_classification=configs.num_classification,
    #             min_threshold=configs.min_threshold,
    #         )
    #     case BertModel_binary_classify_noise | BertModel_binary_classify_noise_only:
    #         model = MatchModel.from_pretrained(pretrainedModelDir, config=my_config, local_files_only=False, min_threshold=configs.min_threshold)

    # multi
    if MatchModel in (BertModel_multi_classify, RobertaModel_multi_classify):
        model = MatchModel.from_pretrained(
            pretrainedModelDir, config=my_config, local_files_only=False, num_classification=configs.num_classification
        )
    elif MatchModel is BertModel_rephrase_just_data_aug:
        model = MatchModel.from_pretrained(
            pretrainedModelDir, config=my_config, local_files_only=False, is_iwr=(TEXTTYPE == TextType.JUST_DATA_AUG_REP4)
        )
    else:
        if "multi_model" in configs.model_name and "warmboost" in configs.model_name and "s2m" not in configs.model_name:
            model = MatchModel.from_pretrained(pretrainedModelDir, False, config=my_config, local_files_only=False)
        elif "multi_model" in configs.model_name and "after_contrast" in configs.model_name:
            model = MatchModel.from_pretrained(pretrainedModelDir, False, config=my_config, local_files_only=False)
        elif "auxloss_sep" in configs.model_name:
            model = MatchModel.from_pretrained(pretrainedModelDir, config=my_config, local_files_only=False, auxloss=configs.auxloss)
        elif "auxloss" in configs.model_name:
            model = MatchModel.from_pretrained(
                pretrainedModelDir, config=my_config, local_files_only=False, auxloss_warmup_steps=configs.auxloss_warmup_steps
            )
            # getattr(configs, "auxloss_warmup_steps", None)
        else:
            model = MatchModel.from_pretrained(pretrainedModelDir, config=my_config, local_files_only=False)
    # tokenizer = AutoTokenizer.from_pretrained(pretrainedModelDir, do_lower_case=configs.do_lower_case)
    tokenizer = AutoTokenizer.from_pretrained(pretrainedModelDir)
    end = time.time()

    logger.debug(f"local_rank {local_rank}: Loading model and tokenizer from disk to CPU memory takes {end - start:.2f} sec.")
    return model, tokenizer
    # # * Load model to GPU memory
    # logger.debug(f"local_rank {local_rank}: Loading model to GPU memory...")
    # start = time.time()


@record
def main() -> None:
    # * Request GPU memory
    # allocate_gpu_memory(0.8)

    # * Loading model
    model, tokenizer = load_model()
    if "reset_emb" in configs.model_name:
        old_emb = model.get_input_embeddings()
        emb = torch.nn.Embedding(*old_emb.weight.shape)
        emb.weight.data = generate_spherical_vector(old_emb.weight.shape, device=old_emb.weight.device, dtype=old_emb.weight.dtype, r=torch.tensor(3))
        print(emb.weight.requires_grad)
        model.set_input_embeddings(emb)
    if configs.dashboard == "wandb":
        wandb.watch(model.module if hasattr(model, "module") else model, log_freq=256)

    # * load dataset
    train_dataset, val_dataset, test_dataset = load_dataset(tokenizer)

    # * Train
    trainer = Trainer(
        task_type="classify",
        evaluate_only=False,
        config=configs,
        model=model,
        dataset_train=train_dataset,
        dataset_val=val_dataset,
        dataset_test=test_dataset,
        extral_evaluators=[_Evaluator1],
        optimizer="AdamW",
        scheduler="linearWarmupDecay",
        tokenizer=tokenizer,
        dashboard_writer=run,
        extral_args_training={"is_train": True},
        extral_args_evaluation={"is_train": False},
    )
    trainer.train()
    # time.sleep(3)


if __name__ == "__main__":
    # * Get args
    configs: NLPTrainingConfig = Fire(NLPTrainingConfig)

    if configs.dataset_name == "LCQMC":
        if "warmboost" in configs.model_name:
            if "single" in configs.model_name or "baseline" in configs.model_name or "s2m" in configs.model_name:
                if "nodrop" in configs.model_name:
                    if "after_contrast" in configs.model_name:  # stage2
                        margin = re.search(r"margin=([\d\.]*)", configs.model_name).group(1)
                        stage1_model_dir = Path(
                            f"outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/nodrop_single_model_after_contrast_margin={margin}/3/16/3e-05"
                        )
                    elif "auxloss=kl" in configs.model_name:
                        stage1_model_dir = Path(
                            "outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/nodrop_single_model_auxloss=kl_warmupepoch=1/3/16/3e-05"
                        )
                    else:
                        stage1_model_dir = Path("outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/TIWR_nodrop_single_model/3/16/3e-05")
                else:
                    stage1_model_dir = Path("outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/single_model/3/16/3e-05")
            elif "multi" in configs.model_name:
                if "shareclassifier" in configs.model_name:
                    stage1_model_dir = Path("outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/multi_model_shareclassifier/3/16/3e-05")
                else:
                    stage1_model_dir = Path("outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/multi_model/3/16/3e-05")
            configs.model_dir = stage1_model_dir / str(configs.seed_of_stage1) / "optimal_checkpoint"
            if "mismatch" in configs.model_name:
                seeds_of_stage1:list = list(map(int, configs.seeds_of_stage1.split()))
                seed = seeds_of_stage1[(seeds_of_stage1.index(configs.seed_of_stage1)+1)%len(seeds_of_stage1)]
                configs.model_dir = stage1_model_dir / str(seed) / "optimal_checkpoint"
            if "nodrop" in configs.model_name:
                baseline_model_dir = Path("outputs/LCQMC/bert-base-chinese/ORI/all/Baseline_nodrop_baseline/3/16/3e-05")
            else:
                baseline_model_dir = Path("outputs/LCQMC/bert-base-chinese/ORI/all/baseline/3/16/3e-05")
        elif "after_contrast" in configs.model_name:  # stage1
            margin = re.search(r"margin=([\d\.]*)", configs.model_name).group(1)
            if "multi" in configs.model_name:
                configs.model_dir = (
                    Path(f"outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/multi_model_shareclassifier_contrast_only_margin={margin}/1/16/1e-05")
                    / "29"
                    / "optimal_checkpoint"
                )
            else:
                # contrast_only_margin
                configs.model_dir = (
                    Path(f"outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/single_model_contrast_only2_margin={margin}/1/16/3e-05")
                    / "29"
                    / "optimal_checkpoint"
                )
        # elif "auxloss_sep" in configs.model_name and configs.auxloss == False:
        #     configs.model_dir = (
        #         Path(f"outputs/LCQMC/bert-base-chinese/DATA_AUG_REP4/all/single_model_auxloss_sep=True/1/16/1e-05") / "29" / "optimal_checkpoint"
        #     )
    elif configs.dataset_name == "BQ":
        if "warmboost" in configs.model_name:
            if "single" in configs.model_name or "baseline" in configs.model_name or "s2m" in configs.model_name:
                if "nodrop" in configs.model_name:
                    if "auxloss" in configs.model_name:
                        stage1_model_dir = Path("outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/nodrop_single_model_auxloss=logits/3/16/3e-05")
                    elif "after_contrast_margin=1" in configs.model_name:
                        stage1_model_dir = Path(
                            "outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/nodrop_single_model_after_contrast_margin=1/3/16/3e-05"
                        )
                    else:
                        stage1_model_dir = Path("outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/TIWR_nodrop_single_model/3/16/3e-05")
                else:
                    stage1_model_dir = Path("outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/single_model/3/16/3e-05")
            elif "multi" in configs.model_name:
                if "shareclassifier" in configs.model_name:
                    stage1_model_dir = Path("outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/multi_model_shareclassifier/3/16/3e-05")
                else:
                    stage1_model_dir = Path("outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/multi_model/3/16/3e-05")
            configs.model_dir = stage1_model_dir / str(configs.seed_of_stage1) / "optimal_checkpoint"
            if "mismatch" in configs.model_name:
                seeds_of_stage1:list = list(map(int, configs.seeds_of_stage1.split()))
                seed = seeds_of_stage1[(seeds_of_stage1.index(configs.seed_of_stage1)+1)%len(seeds_of_stage1)]
                configs.model_dir = stage1_model_dir / str(seed) / "optimal_checkpoint"
            if "nodrop" in configs.model_name:
                baseline_model_dir = Path("outputs/BQ/bert-base-chinese/ORI/all/Baseline_nodrop_baseline/3/16/3e-05")
            else:
                baseline_model_dir = Path("outputs/BQ/bert-base-chinese/ORI/all/baseline/3/16/3e-05")
        elif "after_contrast" in configs.model_name:
            margin = re.search(r"margin=([\d\.]*)", configs.model_name).group(1)
            configs.model_dir = (
                Path(f"outputs/BQ/bert-base-chinese/DATA_AUG_REP4/all/single_model_contrast_only_margin={margin}/1/16/3e-05")
                / "149"
                / "optimal_checkpoint"
            )
    elif configs.dataset_name == "QQP":
        if "warmboost" in configs.model_name:
            if "single" in configs.model_name or "baseline" in configs.model_name or "s2m" in configs.model_name:
                if "nodrop" in configs.model_name:
                    if "after_contrast_margin=1" in configs.model_name:
                        stage1_model_dir = Path("outputs/QQP/roberta-base/DATA_AUG_REP4/all/nodrop_single_model_after_contrast_margin=1/3/16/3e-05")
                    else:
                        stage1_model_dir = Path("outputs/QQP/roberta-base/DATA_AUG_REP4/all/TIWR_nodrop_single_model/3/16/3e-05")
                else:
                    stage1_model_dir = Path("outputs/QQP/roberta-base/DATA_AUG_REP4/all/single_model/3/16/3e-05")
            elif "multi" in configs.model_name:
                if "shareclassifier" in configs.model_name:
                    stage1_model_dir = Path("outputs/QQP/roberta-base/DATA_AUG_REP4/all/multi_model_shareclassifier/3/16/3e-05")
                else:
                    stage1_model_dir = Path("outputs/QQP/roberta-base/DATA_AUG_REP4/all/multi_model/3/16/3e-05")
            configs.model_dir = stage1_model_dir / str(configs.seed_of_stage1) / "optimal_checkpoint"
            if "mismatch" in configs.model_name:
                # TODO
                seeds_of_stage1:list = list(map(int, configs.seeds_of_stage1.split()))
                seed = seeds_of_stage1[(seeds_of_stage1.index(configs.seed_of_stage1)+1)%len(seeds_of_stage1)]
                # print(f"########{configs.seed_of_stage1}, {seed}#########")
                # exit(1)
                configs.model_dir = stage1_model_dir / str(seed) / "optimal_checkpoint"
            if "nodrop" in configs.model_name:
                baseline_model_dir = Path("outputs/QQP/roberta-base/ORI/all/Baseline_nodrop_baseline/3/16/3e-05")
            else:
                baseline_model_dir = Path("outputs/QQP/roberta-base/ORI/all/baseline/3/16/3e-05")
        elif "after_contrast" in configs.model_name:
            margin = re.search(r"margin=([\d\.]*)", configs.model_name).group(1)
            configs.model_dir = (
                Path(f"outputs/QQP/roberta-base/DATA_AUG_REP4/all/single_model_contrast_only_margin={margin}/1/16/3e-05") / "2" / "optimal_checkpoint"
            )
    elif configs.dataset_name == "MRPC":
        if "warmboost" in configs.model_name:
            if "single" in configs.model_name or "baseline" in configs.model_name or "s2m" in configs.model_name:
                if "nodrop" in configs.model_name:
                    stage1_model_dir = Path("outputs/MRPC/roberta-base/DATA_AUG_REP4/all/TIWR_nodrop_single_model/3/16/2e-05")
                else:
                    stage1_model_dir = Path("outputs/MRPC/roberta-base/DATA_AUG_REP4/all/single_model/3/16/2e-05")
            elif "multi" in configs.model_name:
                if "shareclassifier" in configs.model_name:
                    stage1_model_dir = Path("outputs/MRPC/roberta-base/DATA_AUG_REP4/all/multi_model_shareclassifier/3/16/2e-05")
                else:
                    stage1_model_dir = Path("outputs/MRPC/roberta-base/DATA_AUG_REP4/all/multi_model/3/16/2e-05")
            configs.model_dir = stage1_model_dir / str(configs.seed_of_stage1) / "optimal_checkpoint"
            if "mismatch" in configs.model_name:
                seeds_of_stage1:list = list(map(int, configs.seeds_of_stage1.split()))
                seed = seeds_of_stage1[(seeds_of_stage1.index(configs.seed_of_stage1)+1)%len(seeds_of_stage1)]
                configs.model_dir = stage1_model_dir / str(seed) / "optimal_checkpoint"
            if "nodrop" in configs.model_name:
                baseline_model_dir = Path("outputs/MRPC/roberta-base/ORI/all/Baseline_nodrop_baseline/3/16/2e-05")
            else:
                baseline_model_dir = Path("outputs/MRPC/roberta-base/ORI/all/baseline/3/16/2e-05")
    print(configs.shuffle)
    # * Create checkpoints,tensorboard outputs directory
    _dir = Path(
        configs.dataset_name,
        configs.model_type,
        configs.text_type,
        configs.part,
        configs.model_name,
        str(configs.epochs),
        str(configs.train_batch_size),
        str(configs.opt_lr),
        str(configs.seed),
    )
    configs.save_dir = Path("outputs", _dir)
    configs.save(configs.save_dir, silence=False)

    # * Create checkpoint manager
    ckpt_manager = CheckpointManager(configs.save_dir)

    # * Create logger
    output_path_logger = configs.save_dir / "report.log"
    logger = getLogger(__name__, output_path_logger)
    toolkit.set_file_logger(output_path_logger)

    # * Initalize parallel and seed
    local_rank, world_size = initialize(configs)
    print("local_rank: ", local_rank, "world_size: ", world_size)
    # if configs.parallel:
    #     local_rank, world_size = setup_parallel()
    # else:
    #     local_rank, world_size = 0, 1
    #     torch.cuda.set_device(0)
    # setup_seed(configs.seed)

    # * Global variable
    DATASETNAME = DatasetName[configs.dataset_name]
    TEXTTYPE = TextType[configs.text_type]

    # * Create tensorboard writer
    if configs.dashboard is None:
        run = None
        main()
    else:
        if local_rank == 0:
            if configs.dashboard == "wandb":
                with wandb.init(
                    # mode="disabled",
                    project="second",
                    config=configs.to_dict(),
                    group=f"{configs.dataset_name},train_data={configs.part}",
                    tags=[configs.dataset_name, configs.model_type, configs.model_name, configs.text_type],
                ) as run:
                    assert run is wandb.run
                    main()
            elif configs.dashboard == "tensorboard":
                run_dir = Path("runs", _dir, "logs")
                run_dir.mkdir(parents=True, exist_ok=True)
                with SummaryWriter(comment="training", log_dir=run_dir) as run:
                    main()
        else:
            run = None
            main()

    # if configs.seed == 5:
    #     time.sleep(99999)
