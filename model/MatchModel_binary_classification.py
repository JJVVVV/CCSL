import os
from os import PathLike
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.training.loss_functions import PairInBatchNegCoSentLoss, cos_loss, kl_loss
from torch import Tensor
from transformers import BertConfig, BertModel, PreTrainedModel, RobertaConfig, RobertaModel
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


##################################################Roberta###############################################
class RobertaModel_binary_classify(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train=True,
    ) -> dict[str, Tensor]:
        ret = dict()
        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


# -----------------------------------------------rephrase-----------------------------------------------
class RobertaModel_rephrase_IWR(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids[:, 0]
            attention_mask = attention_mask[:, 0]
            position_ids = position_ids[:, 0] if position_ids else None

            outputs = super().forward(input_ids, attention_mask, None, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float())
            ret["loss"] = loss
        else:
            for i in range(times):
                output = super().forward(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i],
                    position_ids=position_ids[:, i] if position_ids else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                logitss.append(logits)
                if labels is not None:
                    loss += self.loss_func(logits, labels.float())

            ret["logits"] = torch.cat(logitss, dim=1)
            if labels is not None:
                ret["loss"] = loss / batch_size
        return ret


class RobertaModel_rephrase(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=False, output_hidden_states=False
            )
            cls = outputs.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float())
            # ret["loss"] = loss
            ret["loss"] = loss / batch_size
        else:
            for i in range(times):
                output = super().forward(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i],
                    position_ids=position_ids[:, i] if position_ids else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                logitss.append(logits)
                if labels is not None:
                    loss += self.loss_func(logits, labels.float())

            ret["logits"] = torch.cat(logitss, dim=1)
            if labels is not None:
                ret["loss"] = loss / batch_size
        return ret


class RobertaModel_rephrase_zh(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float())
            ret["loss"] = loss
        else:
            for i in range(times):
                output = super().forward(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i],
                    token_type_ids=token_type_ids[:, i],
                    position_ids=position_ids[:, i] if position_ids else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                logitss.append(logits)
                if labels is not None:
                    loss += self.loss_func(logits, labels.float())

            ret["logits"] = torch.cat(logitss, dim=1)
            if labels is not None:
                ret["loss"] = loss
        return ret


class RobertaModel_rephrase_contrast_only(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=1)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)

        input_ids = input_ids.reshape(-1, input_ids.size(-1))
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
        labels = labels.repeat(1, times).reshape(-1, 1)

        outputs = super().forward(
            input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, output_attentions=False, output_hidden_states=False
        )
        cls = outputs.last_hidden_state[:, 0]  # (batch_size*times, embedding_size)
        cls = cls.view(batch_size, times, -1)  # (batch_size, times, embedding_size)
        loss_contrast = self.loss_func_contrast(cls[:, 0], cls[:, 1:])

        ret["loss"] = loss_contrast
        if not is_train:
            ret["logits"] = cls
        return ret


class RobertaMultiModel_rephrase(PreTrainedModel):
    times = 4

    def __init__(self, config: RobertaConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([RobertaModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(RobertaModel(config) for _ in range(self.times))
        self.pooler_list = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.times)])
        self.tanh = nn.Tanh()
        self.classifier_list = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(self.times)])
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            logits = self.classifier_list[i](self.tanh(self.pooler_list[i](cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        ret["logits"] = torch.cat(logitss, dim=1)
        if labels is not None:
            ret["loss"] = loss
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class RobertaMultiModel_rephrase_share_classifier(PreTrainedModel):
    times = 4

    def __init__(self, config: RobertaConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([RobertaModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(RobertaModel(config) for _ in range(self.times))
        if pretrained_model_name_or_path and "single_model" in str(pretrained_model_name_or_path):
            print("loading pooler and classifier")
            t = RobertaModel_rephrase.from_pretrained(pretrained_model_name_or_path)
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.pooler.load_state_dict(t.pooler.state_dict())
            self.classifier.load_state_dict(t.classifier.state_dict())
            del t
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        ret["logits"] = torch.cat(logitss, dim=1)
        if labels is not None:
            ret["loss"] = loss
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class RobertaMultiModel_rephrase_fused(PreTrainedModel):
    times = 4

    def __init__(self, config: RobertaConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([RobertaModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(RobertaModel(config) for _ in range(self.times))
        self.pooler = nn.Linear(config.hidden_size * self.times, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        clss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            clss.append(cls)
        cls = torch.cat(clss, dim=1)
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits
        if labels is not None:
            ret["loss"] = self.loss_func(logits, labels.float())
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class RobertaMultiModel_rephrase_withfused(PreTrainedModel):
    times = 4

    def __init__(self, config: RobertaConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([RobertaModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(RobertaModel(config) for _ in range(self.times))
        self.pooler = nn.Linear(config.hidden_size * self.times, config.hidden_size)
        self.pooler_aux = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        clss = []
        logitss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            clss.append(cls)
            logits = self.classifier(self.tanh(self.pooler_aux(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        cls = torch.cat(clss, dim=1)
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = torch.cat([logits] + logitss, dim=1)
        if labels is not None:
            loss = (loss + self.loss_func(logits, labels.float())) / 2
            ret["loss"] = loss
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = RobertaConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class RobertaModel_rephrase_fused(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: Tensor | None = None, attention_mask: Tensor | None = None, position_ids: Tensor | None = None, labels: Tensor = None
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        cls = 0
        for i in range(times):
            output = super().forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls += output.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits
        if labels is not None:
            loss = self.loss_func(logits, labels.float())
            ret["loss"] = loss

        return ret


class RobertaModel_rephrase_close(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(
        self, input_ids: Tensor | None = None, attention_mask: Tensor | None = None, position_ids: Tensor | None = None, labels: Tensor = None
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        clss = []
        for i in range(times):
            output = super().forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                position_ids=position_ids,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            clss.append(cls)
            logits = self.classifier(self.tanh(self.pooler(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_fn(logits, labels.float())
        for i in range(len(clss) - 1):
            loss += kl_loss(clss[i], clss[i + 1], 1)
        ret["logits"] = torch.cat(logitss, dim=1)
        ret["loss"] = loss

        return ret


# class RobertaModel_6times_bi(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss()

#     def forward(self, input_ids, attention_mask, labels=None, is_training=True):
#         ret = dict()
#         times = input_ids.shape[1]
#         # input_ids: (batch_size, 4, seqence_len)
#         loss = 0
#         logitss = []
#         for i in range(times):
#             output = self.roberta(input_ids=input_ids[:, i], attention_mask=attention_mask[:, i], output_attentions=False)
#             cls = output.last_hidden_state[:, 0]
#             logits = self.classifier(self.tanh(self.pooler(cls)))
#             logitss.append(logits)
#             if labels is not None:
#                 loss += self.loss_func(logitss[i], labels.float()) if i < 4 else self.loss_func(logitss[i], torch.ones_like(labels).float())

#         ret["logits"] = torch.stack(logitss[:4], dim=1).squeeze()
#         ret["loss"] = loss

#         return ret


# class RobertaModel_4times_4classifier_bi(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         # self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.poolers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(4)])
#         self.tanh = nn.Tanh()
#         self.classifiers = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(4)])
#         self.loss_func = nn.BCEWithLogitsLoss()

#     def forward(self, input_ids, attention_mask, labels=None, **kwargs):
#         ret = dict()
#         # input_ids: (batch_size, 4, seqence_len)
#         outputs = []
#         for i in range(4):
#             outputs.append(self.roberta(input_ids=input_ids[:, i], attention_mask=attention_mask[:, i], output_attentions=False))
#         clss = [output.last_hidden_state[:, 0] for output in outputs]

#         logitss = [self.classifiers[i](self.tanh(self.poolers[i](cls))) for cls in clss]
#         ret["logits"] = logitss

#         if labels is None:
#             return logitss

#         losses = [self.loss_func(logits, labels.float()) for logits in logitss]
#         ret["loss"] = sum(losses)
#         return ret

###################################################BERT#####################################################################################################


class BertModel_binary_classify(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs.last_hidden_state[:, 0]
        # ret["cls_hidden_state"] = cls
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


# -------------------------------------------------rephrase-------------------------------------------------
class BertModel_rephrase_IWR(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids[:, 0]
            attention_mask = attention_mask[:, 0]
            token_type_ids = token_type_ids[:, 0]
            position_ids = position_ids[:, 0] if position_ids else None

            outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float())
            ret["loss"] = loss
        else:
            for i in range(times):
                output = super().forward(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i],
                    token_type_ids=token_type_ids[:, i],
                    position_ids=position_ids[:, i] if position_ids else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                logitss.append(logits)
                if labels is not None:
                    loss += self.loss_func(logits, labels.float())

            ret["logits"] = torch.cat(logitss, dim=1)
            if labels is not None:
                ret["loss"] = loss / batch_size
        return ret


class BertModel_rephrase(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float())
            # ret["loss"] = loss
            ret["loss"] = loss / batch_size
        else:
            for i in range(times):
                output = super().forward(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i],
                    token_type_ids=token_type_ids[:, i],
                    position_ids=position_ids[:, i] if position_ids else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                logitss.append(logits)
                if labels is not None:
                    loss += self.loss_func(logits, labels.float())

            ret["logits"] = torch.cat(logitss, dim=1)
            if labels is not None:
                ret["loss"] = loss / batch_size
        return ret


class BertModel_rephrase_auxloss(BertModel):
    def __init__(self, config, add_pooling_layer=True, auxloss_warmup_steps=None):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.aux_pooler = nn.Linear(config.hidden_size, 2)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.sigmoid = nn.Sigmoid()
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=0)
        self.training_runtime = None
        self.auxloss_warmup_steps = auxloss_warmup_steps

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            # s = self.tanh(self.pooler(cls))
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float()) / batch_size
            # loss = self.loss_func(logits.view((batch_size, times, 1))[:, 0], labels.float())
            ret["loss"] = loss
            # if self.auxloss_warmup_steps is None or self.training_runtime["cur_epoch"] > self.auxloss_warmup_steps - 1:
            if self.training_runtime["cur_epoch"] <= self.auxloss_warmup_steps - 1:
                # ----------------aux logits loss1----------------
                # logits = logits.view((batch_size, times, 1))
                # aux_loss = 0
                # for i in range(times):
                #     for j in range(i + 1, times):
                #         aux_loss += (
                #             self.loss_func(logits[:, i], self.sigmoid(logits[:, j])) + self.loss_func(logits[:, j], self.sigmoid(logits[:, i]))
                #         ) / 2
                #         # aux_loss += cos_loss(s[:, i], s[:, j], "sum")
                # aux_loss = aux_loss / 6 / batch_size
                # ----------------aux logits loss2---------------
                # aux_loss = 0
                # logits = logits.view((batch_size, times, 1))
                # for i in range(1, times):
                #     aux_loss += (self.loss_func(logits[:, 0], self.sigmoid(logits[:, i])) + self.loss_func(logits[:, i], self.sigmoid(logits[:, 0]))) / 2
                # aux_loss = aux_loss / 3 / batch_size
                # ----------------aux ranking loss---------------
                # cls_contrast = self.tanh(self.aux_pooler(cls))
                # cls_contrast = cls_contrast.view((batch_size, times, -1))
                # aux_loss = self.loss_func_contrast(cls_contrast[:, 0], cls_contrast[:, 1:])
                # ----------------aux ranking2 loss---------------
                cls_contrast = self.tanh(self.aux_pooler(cls))
                cls_contrast = cls_contrast.view((batch_size, times, -1))
                aux_loss = 0
                for i in range(times):
                    for j in range(i + 1, times):
                        aux_loss += self.loss_func_contrast(cls_contrast[:, i], cls_contrast[:, j])
                aux_loss /= 6
                # ----------------aux kl loss---------------
                # cls = cls.view((batch_size, times, -1))
                # aux_loss = 0
                # for i in range(times):
                #     for j in range(i + 1, times):
                #         aux_loss += kl_loss(cls[:, i], cls[:, j])
                # aux_loss = aux_loss / 6
                # ----------------aux kl2 loss---------------
                # logits = logits.view((batch_size, times))
                # aux_loss = 0
                # for i in range(times):
                #     for j in range(i + 1, times):
                #         aux_loss += kl_loss(logits[:, i], logits[:, j], 0.1)
                # aux_loss = aux_loss / 6

                # ret["loss"] = aux_loss
                ret["loss"] += aux_loss
                ret["loss_display"] = torch.tensor([loss.item(), aux_loss.item()])
            # else:
            #     loss = self.loss_func(logits, labels.float()) / batch_size
            #     # loss = self.loss_func(logits.view((batch_size, times, 1))[:, 0], labels.float())
            #     ret["loss"] = loss
        else:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            # s = self.tanh(self.pooler(cls))
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits.view((batch_size, times))
            if labels is not None:
                loss = self.loss_func(logits, labels.float()) / batch_size
                ret["loss"] = loss
        return ret


class RobertaModel_rephrase_auxloss(RobertaModel):
    def __init__(self, config, add_pooling_layer=True, auxloss_warmup_steps=None):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.sigmoid = nn.Sigmoid()
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=0.2)
        self.training_runtime = None
        self.auxloss_warmup_steps = auxloss_warmup_steps

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        if is_train:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(input_ids, attention_mask, None, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            # s = self.tanh(self.pooler(cls))
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits
            loss = self.loss_func(logits, labels.float()) / batch_size
            # loss = self.loss_func(logits.view((batch_size, times, 1))[:, 0], labels.float())
            ret["loss"] = loss
            if self.auxloss_warmup_steps is None or self.training_runtime["cur_epoch"] > self.auxloss_warmup_steps - 1:
                # ----------------aux logits loss1----------------
                # logits = logits.view((batch_size, times, 1))
                # aux_loss = 0
                # for i in range(times):
                #     for j in range(i + 1, times):
                #         aux_loss += (
                #             self.loss_func(logits[:, i], self.sigmoid(logits[:, j])) + self.loss_func(logits[:, j], self.sigmoid(logits[:, i]))
                #         ) / 2
                #         # aux_loss += cos_loss(s[:, i], s[:, j], "sum")
                # aux_loss = aux_loss / 6 / batch_size
                # ----------------aux logits loss2---------------
                # aux_loss = 0
                # logits = logits.view((batch_size, times, 1))
                # for i in range(1, times):
                #     aux_loss += (self.loss_func(logits[:, 0], self.sigmoid(logits[:, i])) + self.loss_func(logits[:, i], self.sigmoid(logits[:, 0]))) / 2
                # aux_loss = aux_loss / 3 / batch_size
                # ----------------aux ranking loss---------------
                # cls = cls.view((batch_size, times, -1))
                # aux_loss = self.loss_func_contrast(cls[:, 0], cls[:, 1:])
                # ----------------aux kl loss---------------
                cls = cls.view((batch_size, times, -1))
                aux_loss = 0
                for i in range(times):
                    for j in range(i + 1, times):
                        aux_loss += kl_loss(cls[:, i], cls[:, j], 0.1)
                aux_loss = aux_loss / 6

                ret["loss"] += aux_loss
                ret["loss_display"] = torch.tensor([loss.item(), aux_loss.item()])
        else:
            input_ids = input_ids.reshape(-1, input_ids.size(-1))
            attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
            position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
            labels = labels.repeat(1, times).reshape(-1, 1)

            outputs = super().forward(input_ids, attention_mask, None, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            # s = self.tanh(self.pooler(cls))
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits.view((batch_size, times))
            if labels is not None:
                loss = self.loss_func(logits, labels.float()) / batch_size
                ret["loss"] = loss
        return ret


class BertModel_rephrase_auxloss_sep(BertModel):
    def __init__(self, config, add_pooling_layer=True, auxloss=False):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.sigmoid = nn.Sigmoid()
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=0.2)
        self.auxloss = auxloss

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []

        input_ids = input_ids.reshape(-1, input_ids.size(-1))
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
        position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
        labels = labels.repeat(1, times).reshape(-1, 1)

        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs.last_hidden_state[:, 0]
        # s = self.tanh(self.pooler(cls))
        logits = self.classifier(self.tanh(self.pooler(cls)))

        if not self.auxloss:
            loss = self.loss_func(logits, labels.float()) / batch_size
            # loss = self.loss_func(logits.view((batch_size, times, 1))[:, 0], labels.float())
        else:
            # ----------------aux logits loss1----------------
            logits = logits.view((batch_size, times, 1))
            loss = 0
            for i in range(times):
                for j in range(i + 1, times):
                    loss += (self.loss_func(logits[:, i], self.sigmoid(logits[:, j])) + self.loss_func(logits[:, j], self.sigmoid(logits[:, i]))) / 2
                    # aux_loss += cos_loss(s[:, i], s[:, j], "sum")
            loss = loss / 6 / batch_size
            # ----------------aux logits loss2---------------
            # aux_loss = 0
            # logits = logits.view((batch_size, times, 1))
            # for i in range(1, times):
            #     aux_loss += (self.loss_func(logits[:, 0], self.sigmoid(logits[:, i])) + self.loss_func(logits[:, i], self.sigmoid(logits[:, 0]))) / 2
            # aux_loss = aux_loss / 3 / batch_size
            # ----------------aux ranking loss---------------
            # cls = cls.view((batch_size, times, -1))
            # aux_loss = self.loss_func_contrast(cls[:, 0], cls[:, 1:])

            ret["loss"] = loss
            # ret["loss_display"] = torch.tensor([loss.item(), aux_loss.item()])
            ret["logits"] = logits.view((batch_size, times))
        return ret


class BertModel_rephrase_just_data_aug(BertModel):
    def __init__(self, config, add_pooling_layer=True, is_iwr=False):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

        self.is_iwr = is_iwr

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        if is_train:
            outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            # ret["cls_hidden_state"] = cls
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits

            if labels is None:
                return ret

            loss = self.loss_func(logits, labels.float())
            # ret["loss"] = loss / 16
            ret["loss"] = loss
            return ret
        else:
            if self.is_iwr:
                batch_size, times, _ = input_ids.shape
                # input_ids: (batch_size, 4, seqence_len)
                loss = 0
                logitss = []
                for i in range(times):
                    output = super().forward(
                        input_ids=input_ids[:, i],
                        attention_mask=attention_mask[:, i],
                        token_type_ids=token_type_ids[:, i],
                        position_ids=position_ids[:, i] if position_ids else None,
                        output_attentions=False,
                        output_hidden_states=False,
                    )
                    cls = output.last_hidden_state[:, 0]
                    logits = self.classifier(self.tanh(self.pooler(cls)))
                    logitss.append(logits)
                    if labels is not None:
                        loss += self.loss_func(logits, labels.float())

                ret["logits"] = torch.cat(logitss, dim=1)
                if labels is not None:
                    # ret["loss"] = loss / 16
                    ret["loss"] = loss
            else:
                batch_size, _ = input_ids.shape
                output = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                ret["logits"] = logits

                if labels is None:
                    return ret

                loss = self.loss_func(logits, labels.float())
                # ret["loss"] = loss / 16
                ret["loss"] = loss
        return ret


# class RobertaModel_just_data_aug(RobertaModel):
#     def __init__(self, config, add_pooling_layer=True):
#         super().__init__(config, add_pooling_layer)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")

#     def forward(
#         self,
#         input_ids: Tensor | None = None,
#         attention_mask: Tensor | None = None,
#         position_ids: Tensor | None = None,
#         labels: Tensor = None,
#         is_train: bool = True,
#     ) -> dict[str, Tensor]:
#         ret = dict()
#         if is_train:
#             outputs = super().forward(input_ids, attention_mask, None, position_ids, output_attentions=False, output_hidden_states=False)
#             cls = outputs.last_hidden_state[:, 0]
#             # ret["cls_hidden_state"] = cls
#             logits = self.classifier(self.tanh(self.pooler(cls)))
#             ret["logits"] = logits

#             if labels is None:
#                 return ret

#             loss = self.loss_func(logits, labels.float())
#             ret["loss"] = loss / 16
#             return ret

#         return ret


class RobertaModel_rephrase_just_data_aug(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        if is_train:
            outputs = super().forward(input_ids, attention_mask, None, position_ids, output_attentions=False, output_hidden_states=False)
            cls = outputs.last_hidden_state[:, 0]
            # ret["cls_hidden_state"] = cls
            logits = self.classifier(self.tanh(self.pooler(cls)))
            ret["logits"] = logits

            if labels is None:
                return ret

            loss = self.loss_func(logits, labels.float())
            ret["loss"] = loss / 16
            return ret
        else:
            batch_size, times, _ = input_ids.shape
            # input_ids: (batch_size, 4, seqence_len)
            loss = 0
            logitss = []
            for i in range(times):
                output = super().forward(
                    input_ids=input_ids[:, i],
                    attention_mask=attention_mask[:, i],
                    position_ids=position_ids[:, i] if position_ids else None,
                    output_attentions=False,
                    output_hidden_states=False,
                )
                cls = output.last_hidden_state[:, 0]
                logits = self.classifier(self.tanh(self.pooler(cls)))
                logitss.append(logits)
                if labels is not None:
                    loss += self.loss_func(logits, labels.float())

            ret["logits"] = torch.cat(logitss, dim=1)
            if labels is not None:
                ret["loss"] = loss / 16
        return ret


# class BertModel_rephrase_contrast(BertModel):
#     def __init__(self, config, add_pooling_layer=True):
#         super().__init__(config, add_pooling_layer)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func_contrast = PairInBatchNegCoSentLoss()
#         # self.loss_func = nn.BCEWithLogitsLoss()
#         self.loss_func = nn.BCEWithLogitsLoss(reduction="sum")

#     def forward(
#         self,
#         input_ids: Tensor | None = None,
#         attention_mask: Tensor | None = None,
#         token_type_ids: Tensor | None = None,
#         position_ids: Tensor | None = None,
#         labels: Tensor = None,
#         is_train: bool = True,
#     ) -> dict[str, Tensor]:
#         ret = dict()
#         batch_size, times, _ = input_ids.shape
#         # input_ids: (batch_size, 4, seqence_len)
#         loss = 0
#         logitss = []
#         if is_train:
#             input_ids = input_ids.reshape(-1, input_ids.size(-1))
#             attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
#             token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
#             position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
#             labels = labels.repeat(1, times).reshape(-1, 1)

#             outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
#             cls = outputs.last_hidden_state[:, 0]  # (batch_size*times, embedding_size)
#             logits = self.classifier(self.tanh(self.pooler(cls)))
#             ret["logits"] = logits
#             loss = self.loss_func(logits, labels.float())

#             cls = cls.view(batch_size, times, -1)
#             loss_contrast = self.loss_func_contrast(cls[:, 0], cls[:, 1:])

#             # ret["loss"] = loss
#             ret["loss"] = loss / batch_size + loss_contrast

#         else:
#             for i in range(times):
#                 output = super().forward(
#                     input_ids=input_ids[:, i],
#                     attention_mask=attention_mask[:, i],
#                     token_type_ids=token_type_ids[:, i],
#                     position_ids=position_ids[:, i] if position_ids else None,
#                     output_attentions=False,
#                     output_hidden_states=False,
#                 )
#                 cls = output.last_hidden_state[:, 0]
#                 logits = self.classifier(self.tanh(self.pooler(cls)))
#                 logitss.append(logits)
#                 if labels is not None:
#                     loss += self.loss_func(logits, labels.float())

#             ret["logits"] = torch.cat(logitss, dim=1)
#             if labels is not None:
#                 ret["loss"] = loss / batch_size
#         return ret


class BertModel_rephrase_contrast_only(BertModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=0)

        # self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        # self.tanh = nn.Tanh()
        # self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)

        input_ids = input_ids.reshape(-1, input_ids.size(-1))
        attention_mask = attention_mask.reshape(-1, attention_mask.size(-1))
        token_type_ids = token_type_ids.reshape(-1, token_type_ids.size(-1))
        position_ids = position_ids.reshape(-1, position_ids.size(-1)) if position_ids else None
        labels = labels.repeat(1, times).reshape(-1, 1)

        outputs = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs.last_hidden_state[:, 0]  # (batch_size*times, embedding_size)

        # ranking
        cls = cls.view(batch_size, times, -1)  # (batch_size, times, embedding_size)
        # aux_loss = self.loss_func_contrast(cls[:, 0], cls[:, 1:])
        aux_loss = 0
        for i in range(times):
            for j in range(i + 1, times):
                aux_loss += self.loss_func_contrast(cls[:, i], cls[:, j])
        aux_loss /= 6
        # logits
        # logits = self.classifier(self.tanh(self.pooler(cls)))
        # logits = logits.view((batch_size, times, 1))
        # aux_loss = 0
        # for i in range(times):
        #     for j in range(i + 1, times):
        #         aux_loss += (self.loss_func(logits[:, i], self.sigmoid(logits[:, j])) + self.loss_func(logits[:, j], self.sigmoid(logits[:, i]))) / 2
        #         # aux_loss += cos_loss(s[:, i], s[:, j], "sum")
        # aux_loss = aux_loss / 6 / batch_size

        ret["loss"] = aux_loss
        if not is_train:
            ret["logits"] = cls
        return ret


class BertMultiModel_rephrase(PreTrainedModel):
    times = 4

    def __init__(self, config: BertConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([BertModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(BertModel(config) for _ in range(self.times))
        self.pooler_list = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(self.times)])
        self.tanh = nn.Tanh()
        self.classifier_list = nn.ModuleList([nn.Linear(config.hidden_size, 1) for _ in range(self.times)])
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                token_type_ids=token_type_ids[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            logits = self.classifier_list[i](self.tanh(self.pooler_list[i](cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        ret["logits"] = torch.cat(logitss, dim=1)
        if labels is not None:
            ret["loss"] = loss
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class BertMultiModel_rephrase_share_classifier_contrast_only(PreTrainedModel):
    times = 4

    def __init__(self, config: BertConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([BertModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(BertModel(config) for _ in range(self.times))
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=1)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        clss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                token_type_ids=token_type_ids[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            clss.append(cls)
        cls = torch.stack(clss, dim=1)
        loss_contrast = self.loss_func_contrast(cls[:, 0], cls[:, 1:])

        ret["loss"] = loss_contrast
        if not is_train:
            ret["logits"] = cls
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class BertMultiModel_rephrase_share_classifier(PreTrainedModel):
    times = 4

    def __init__(self, config: BertConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([BertModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(BertModel(config) for _ in range(self.times))
        if pretrained_model_name_or_path and "single_model" in str(pretrained_model_name_or_path):
            print("loading pooler and classifier")
            t = BertModel_rephrase.from_pretrained(pretrained_model_name_or_path)
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.pooler.load_state_dict(t.pooler.state_dict())
            self.classifier.load_state_dict(t.classifier.state_dict())
            del t
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        times = input_ids.shape[1]
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        logitss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                token_type_ids=token_type_ids[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            logits = self.classifier(self.tanh(self.pooler(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        ret["logits"] = torch.cat(logitss, dim=1)
        if labels is not None:
            ret["loss"] = loss
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret


class BertMultiModel_rephrase_share_classifier_auxloss(PreTrainedModel):
    times = 4

    def __init__(self, config: BertConfig, pretrained_model_name_or_path=None, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config
        if pretrained_model_name_or_path:
            self.model_list = nn.ModuleList([BertModel.from_pretrained(pretrained_model_name_or_path) for _ in range(self.times)])
        else:
            self.model_list = nn.ModuleList(BertModel(config) for _ in range(self.times))
        if pretrained_model_name_or_path and "single_model" in str(pretrained_model_name_or_path):
            print("loading pooler and classifier")
            t = BertModel_rephrase.from_pretrained(pretrained_model_name_or_path)
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.pooler.load_state_dict(t.pooler.state_dict())
            self.classifier.load_state_dict(t.classifier.state_dict())
            del t
        else:
            self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
            self.classifier = nn.Linear(config.hidden_size, 1)
        self.tanh = nn.Tanh()
        self.loss_func = nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()
        self.loss_func_contrast = PairInBatchNegCoSentLoss(0.05, margin=1)

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
        is_train: bool = True,
    ) -> dict[str, Tensor]:
        ret = dict()
        batch_size, times, _ = input_ids.shape
        # input_ids: (batch_size, 4, seqence_len)
        loss = 0
        clss = []
        logitss = []
        for i in range(times):
            output = self.model_list[i].forward(
                input_ids=input_ids[:, i],
                attention_mask=attention_mask[:, i],
                token_type_ids=token_type_ids[:, i],
                position_ids=position_ids[:, i] if position_ids else None,
                output_attentions=False,
                output_hidden_states=False,
            )
            cls = output.last_hidden_state[:, 0]
            clss.append(cls)
            logits = self.classifier(self.tanh(self.pooler(cls)))
            logitss.append(logits)
            if labels is not None:
                loss += self.loss_func(logits, labels.float())
        ret["logits"] = torch.cat(logitss, dim=1)
        if labels is not None:
            ret["loss"] = loss

        if is_train:
            # ----------------aux logits loss1----------------
            logits = torch.stack(logitss, dim=1)
            aux_loss = 0
            for i in range(times):
                for j in range(i + 1, times):
                    aux_loss += (
                        self.loss_func(logits[:, i], self.sigmoid(logits[:, j])) + self.loss_func(logits[:, j], self.sigmoid(logits[:, i]))
                    ) / 2
                    # aux_loss += cos_loss(s[:, i], s[:, j], "sum")
            aux_loss = aux_loss / 6 / batch_size
            # ----------------aux ranking loss---------------
            # cls = torch.stack(clss, dim=1)
            # aux_loss = self.loss_func_contrast(cls[:, 0], cls[:, 1:])

            ret["loss"] += aux_loss
            ret["loss_display"] = torch.tensor([loss.item(), aux_loss.item()])
        return ret

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike | None,
        load_from_pretrained: bool = True,
        *model_args,
        config: PretrainedConfig | str | PathLike | None = None,
        cache_dir: str | PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool = None,
        **kwargs
    ):
        if load_from_pretrained:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            ret = cls(config, pretrained_model_name_or_path)
        else:
            config = BertConfig.from_pretrained(pretrained_model_name_or_path)
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                config=config,
                cache_dir=cache_dir,
                ignore_mismatched_sizes=ignore_mismatched_sizes,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                use_safetensors=use_safetensors,
                **kwargs
            )
        return ret
