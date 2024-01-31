import os
from os import PathLike
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from toolkit.training.loss_functions import PairInBatchNegCoSentLoss, cos_loss, kl_loss
from torch import Tensor
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertPreTrainedModel,
    PreTrainedModel,
    RobertaConfig,
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from .mod import WAIO
from .tricks import generate_distribution, generate_distribution2, generate_distribution3, rotate_embeddings, shift_embeddings

# from .mod2 import AttnNoProjVal


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


class RobertaModel_binary_classify_noise(RobertaModel):
    def __init__(self, config, min_threshold, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, min_threshold=None, *model_args, **kwargs):
        kwargs["min_threshold"] = min_threshold
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

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
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            dis = generate_distribution2(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=self.min_threshold)
            # emb = self.roberta.get_input_embeddings().weight.clone()
            emb = self.get_input_embeddings().weight
            input_emb = torch.matmul(dis, emb)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_emb, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class RobertaModel_binary_classify_shift(RobertaModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha
        # self.get_input_embeddings().weight.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

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
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = shift_embeddings(input_embs, self.alpha)
            # print(f"#######################\n{input_embs.dtype}")
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class RobertaModel_binary_classify_shift_only(RobertaModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

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
        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = shift_embeddings(input_embs, self.alpha)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
        else:
            outputs2 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        loss2 = self.loss_func(logits2, labels.float())
        ret["logits"] = logits2
        ret["loss"] = loss2
        return ret


class RobertaModel_binary_classify_rotate(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size

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
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = rotate_embeddings(input_embs)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class RobertaModel_binary_classify_rotate_only(RobertaModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size

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
        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = rotate_embeddings(input_embs)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
        else:
            outputs2 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        loss2 = self.loss_func(logits2, labels.float())
        ret["logits"] = logits2
        ret["loss"] = loss2
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
                ret["loss"] = loss / 16
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


# --------------------------------------------------shift---------------------------------------------
class BertModel_binary_classify_shift(BertModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha
        # self.get_input_embeddings().weight.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

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
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            input_embs = self.get_input_embeddings()(input_ids)
            input_embs = shift_embeddings(input_embs, self.alpha)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class BertModel_binary_classify_noise(BertModel):
    def __init__(self, config, min_threshold, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, min_threshold=None, *model_args, **kwargs):
        kwargs["min_threshold"] = min_threshold
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

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
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss

        if is_train:
            dis = generate_distribution2(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=self.min_threshold)
            # emb = self.roberta.get_input_embeddings().weight.clone()
            emb = self.get_input_embeddings().weight
            input_emb = torch.matmul(dis, emb)
            outputs2 = super().forward(
                None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_emb, output_attentions=False, output_hidden_states=False
            )
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class BertModel_binary_classify_shift_only(BertModel):
    def __init__(self, config, alpha, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.alpha = alpha
        # self.get_input_embeddings().weight.requires_grad = False

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, alpha, *model_args, **kwargs):
        kwargs["alpha"] = alpha
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

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
        # if is_train:
        #     input_embs = self.get_input_embeddings()(input_ids)
        #     input_embs = shift_embeddings(input_embs, self.alpha)
        #     outputs2 = super().forward(
        #         None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
        #     )
        # else:
        #     outputs2 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        input_embs = self.get_input_embeddings()(input_ids)
        input_embs = shift_embeddings(input_embs, self.alpha)
        outputs2 = super().forward(
            None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_embs, output_attentions=False, output_hidden_states=False
        )
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        loss2 = self.loss_func(logits2, labels.float())
        ret["logits"] = logits2
        ret["loss"] = loss2
        return ret


# ! 有bug: infer的时候, 不应该加 noise的
class BertModel_binary_classify_noise_only(BertModel):
    def __init__(self, config, min_threshold, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size
        self.min_threshold = min_threshold

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, min_threshold=None, *model_args, **kwargs):
        kwargs["min_threshold"] = min_threshold
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        dis = generate_distribution(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=self.min_threshold)
        # emb = self.roberta.get_input_embeddings().weight.clone()
        emb = self.get_input_embeddings().weight
        input_emb = torch.matmul(dis, emb)
        outputs2 = super().forward(
            None, attention_mask, token_type_ids, position_ids, inputs_embeds=input_emb, output_attentions=False, output_hidden_states=False
        )
        cls2 = outputs2.last_hidden_state[:, 0]
        logits2 = self.classifier(self.tanh(self.pooler(cls2)))
        ret["logits"] = logits2
        if labels is not None:
            loss2 = self.loss_func(logits2, labels.float())
            ret["loss"] = loss2
        return ret


# 不太行
###########################################################################################################################################


class RobertaModel_gaussion_label_bi(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 100)
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        # ret['weight_bert'] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        ret["cls_hidden_state"] = cls
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels)
        ret["loss"] = loss
        return ret


# 蒸馏相关
######################################################################################################################################################################


class Fusion(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.fusion_que_ans = nn.Linear(dim * 2, dim)
        self.fusion_qa = nn.Linear(dim * 2, dim)
        self.fusion_cls_qa = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, cls, que1, ans1, que2, ans2):
        qa1 = self.gelu(self.fusion_que_ans(torch.cat((que1, ans1), dim=1)))
        qa2 = self.gelu(self.fusion_que_ans(torch.cat((que2, ans2), dim=1)))
        qas = self.tanh(self.fusion_qa(torch.cat((qa1, qa2), dim=1)))
        ret = self.tanh(self.fusion_cls_qa(torch.cat((cls, qas), dim=1)))
        return ret


# Roberta_student
class RobertaModel_student_bi(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

        self.fusion = Fusion(config.hidden_size)

    def forward(self, input_ids, attention_mask, cls_sep_idx, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        cls = outputs.last_hidden_state[:, 0]
        # pdb.set_trace()
        # for i in range(input_ids.shape[0]):
        #     print(outputs.last_hidden_state[i][cls_sep_idx[i][0] + 1 : cls_sep_idx[i][1] - 64].shape)
        que1 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][0] + 1 : cls_sep_idx[i][1] - 64].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )
        ans1 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][1] - 64 : cls_sep_idx[i][1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )
        que2 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][2] + 1 : cls_sep_idx[i][-1] - 64].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )
        ans2 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][-1] - 64 : cls_sep_idx[i][-1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )

        finalHiddenState = self.fusion(cls, que1, ans1, que2, ans2)
        logits = self.classifier(self.tanh(self.pooler(finalHiddenState)))
        ret["logits"] = logits
        ret["ans1"] = ans1
        ret["ans2"] = ans2

        if labels is None:
            return logits

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


# Roberta_teacher
class RobertaModel_teacher(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        # self.tanh = nn.Tanh()
        # self.classifier = nn.Linear(config.hidden_size, 1)
        # self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, cls_sep_idx, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)

        ans1 = torch.stack([outputs.last_hidden_state[i][1 : cls_sep_idx[i][1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0)
        ans2 = torch.stack(
            [outputs.last_hidden_state[i][cls_sep_idx[i][2] + 1 : cls_sep_idx[i][-1]].mean(dim=0) for i in range(input_ids.shape[0])], dim=0
        )

        ret["ans1"] = ans1
        ret["ans2"] = ans2

        return ret


class Classifier(nn.Module):
    def __init__(self, config, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, cls1, cls2):
        cls_fin = torch.concat([cls1, cls2], dim=-1)
        return self.classifier(self.tanh(self.pooler(cls_fin)))


# Roberta_ensemble_baseline
class RobertaModel_ensemble_bi(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.classifier = Classifier(config)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        # ret['weight_bert'] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        ret["cls_hidden_state"] = cls
        return ret


# 第一个点
######################################################################################################################################################################


class Gate_and_fusion(nn.Module):
    def __init__(self, dim) -> None:
        super().__init__()
        self.fusion_same = nn.Linear(dim * 3, dim)
        self.fusion_dif = nn.Linear(dim * 2, dim)
        self.gelu = nn.GELU()
        self.tanh = nn.Tanh()

    def forward(self, cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o):
        same = self.gelu(self.fusion_same(torch.cat((cls, sep_s1, sep_s2), dim=1)))
        cls_fin1 = self.tanh(self.fusion_dif(torch.cat((same, sep_s1_o), dim=1)))
        cls_fin2 = self.tanh(self.fusion_dif(torch.cat((same, sep_s2_o), dim=1)))
        return cls_fin1, cls_fin2


def log(x, n):
    n = torch.tensor(n)
    return torch.log(x) / torch.log(n)


def get_weights_loss(logits, labels, base):
    logits = logits.detach()
    labels = labels.detach()
    y = torch.sigmoid(logits)
    gama = 1 - 1 / base
    weight_loss = torch.ones((logits.shape[0], 1), device=logits.device)
    already_classable = (logits > 0).view(-1) == labels.bool()
    x = (abs(labels - y) * 2)[already_classable] * (gama)  # x: [game, 0]
    x = 1 - x
    weight_loss[already_classable] = -log(x, base)
    return weight_loss


class FixedPositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, max_len=512, num_hiddens=768, dropout=0.1):
        super().__init__()
        # self.dropout = nn.Dropout(dropout)
        # 创建一个足够长的P
        self.P = torch.zeros((1, max_len, num_hiddens))
        X = torch.arange(max_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, num_hiddens, 2, dtype=torch.float32) / num_hiddens
        )
        self.P[:, :, 0::2] = torch.sin(X)
        self.P[:, :, 1::2] = torch.cos(X)

    def forward(self, X):
        X = X + self.P[:, : X.shape[1], :].to(X.device)
        return X
        # return self.dropout(X)


# class TrainablePositionalEncoding(nn.Module):
#     """位置编码"""
#     def __init__(self, seq_len=512, hidden_dim=768):
#         super().__init__()
#         self.weight = torch.zeros((seq_len, hidden_dim))
#         # self.P = torch.nn.Embedding(seq_len, hidden_dim)
#         X = torch.arange(seq_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(10000, torch.arange(
#             0, hidden_dim, 2, dtype=torch.float32) / hidden_dim)
#         self.weight[:, 0::2] = torch.sin(X)
#         self.weight[:, 1::2] = torch.cos(X)
#         self.P = torch.nn.Embedding(seq_len, hidden_dim).from_pretrained(self.weight)

#     def forward(self, X, position_ids):
#         X = X + self.P(position_ids)
#         return X


# # Roberta
# class RobertaMatchModel(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         self.pooler = nn.Linear(config.hidden_size*2, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
#         # --------------------------------------------------------------------------------------------------------------
#         self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # --------------------------------------------------------------------------------------------------------------
#         self.att_layer = WAIO(config, need_para=True)
#         self.fusion = Gate_and_fusion(config.hidden_size)


#     def forward(self, input_ids, attention_mask, cls_sep_idxs, labels=None):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         att_layer_input = self.fixed_position_encoder(att_layer_input)
#         # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
#         # --------------------------------------------------------------------------------------------------------------
#         last, last_overlook, weight, weight_o = self.att_layer(
#             att_layer_input, att_layer_input, att_layer_input,
#             key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         ret['weight'] = weight
#         ret['weight_o'] = weight_o
#         ret['weight_bert'] = outputs.attentions

#         cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)

#         # sep
#         sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         # --------------------------------------------------------------------------------------------------------------
#         cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
#         cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))
#         # --------------------------------------------------------------------------------------------------------------

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # WAIO30
#         base = 2
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret
# --------------------------------------------------------------------------------------------------------------

# # Roberta_without_mod1
# class RobertaMatchModel_without_mod1(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')

#     def forward(self, input_ids, attention_mask, labels=None, **kwargs):
#         ret = dict()
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         # pooler_output = outputs[1]
#         # logits = self.linear(self.dropout(pooler_output))

#         cls = outputs.last_hidden_state[:, 0]
#         logits = self.classifier(self.tanh(self.pooler(cls)))
#         ret['logits'] = logits
#         if labels is None:
#             return logits
#         loss = self.loss_func(logits, labels.float())

#         # WAIO30
#         base = 1000
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret

# # Roberta_withou_out_mod2
# class RobertaMatchModel_without_mod2(RobertaPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.roberta = RobertaModel(config)
#         self.pooler = nn.Linear(config.hidden_size*2, config.hidden_size)
#         self.tanh = nn.Tanh()
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
#         # --------------------------------------------------------------------------------------------------------------
#         self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # --------------------------------------------------------------------------------------------------------------
#         self.att_layer = WAIO(config, need_para=True)
#         self.fusion = Gate_and_fusion(config.hidden_size)


#     def forward(self, input_ids, attention_mask, cls_sep_idxs, labels=None):
#         outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         att_layer_input = self.fixed_position_encoder(att_layer_input)
#         # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
#         # --------------------------------------------------------------------------------------------------------------
#         last, last_overlook, weight, weight_o = self.att_layer(
#             att_layer_input, att_layer_input, att_layer_input,
#             key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         ret['weight'] = weight
#         ret['weight_o'] = weight_o
#         ret['weight_bert'] = outputs.attentions

#         cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)

#         # sep
#         sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         # --------------------------------------------------------------------------------------------------------------
#         cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
#         cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))
#         # --------------------------------------------------------------------------------------------------------------

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # # WAIO30
#         # base = 1000
#         # weight_loss = get_weights_loss(logits, labels, base)
#         # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret


class BertModelFirst(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert: BertModel = BertModel(config)
        # config.hidden_size=768
        # self.pooler = nn.Linear(config.hidden_size*3, config.hidden_size)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.tanh = nn.Tanh()
        # --------------------------------------------------------------------------------------------------------------
        # self.my_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # --------------------------------------------------------------------------------------------------------------
        self.att_layer = WAIO(config, need_para=True)

        self.fusion = Gate_and_fusion(config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # self.loss_aux = nn.CosineEmbeddingLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids, cls_sep_idx, labels=None):
        # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
        # outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
        #                     position_ids=position_ids, output_attentions=True)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True)
        ret = dict()
        # ret['representation'] = outputs.last_hidden_state
        att_layer_input = outputs.last_hidden_state
        # --------------------------------------------------------------------------------------------------------------
        # att_layer_input = outputs.last_hidden_state+self.my_position_embeddings(position_ids)
        att_layer_input = self.fixed_position_encoder(att_layer_input)
        # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
        # --------------------------------------------------------------------------------------------------------------
        last, last_overlook, weight, weight_o = self.att_layer(
            att_layer_input, att_layer_input, att_layer_input, key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idx
        )

        # last_overlook += self.my_position_embeddings(position_ids)
        # last, last_overlook, weight, weight_o = self.att_layer(
        #     outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
        #     key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

        ret["weight"] = weight
        ret["weight_o"] = weight_o

        ret["weight_bert"] = outputs.attentions

        cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)
        # cls = last[:, 0]

        # sep
        sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idx[:, 1]]
        sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idx[:, 1]]
        sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idx[:, 2]]
        sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idx[:, 2]]
        # --------------------------------------------------------------------------------------------------------------
        cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
        # same, cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        # cls_fin = torch.cat((same, cls_fin1, cls_fin2), dim=1)
        # logits = self.classifier(self.dropout(self.tanh(self.pooler(cls_fin))))
        logits = self.classifier(self.tanh(self.pooler(cls_fin)))
        # --------------------------------------------------------------------------------------------------------------

        ret["logits"] = logits

        if labels is None:
            return ret

        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())

        # 方法1 WAIO9
        # weight_loss = torch.softmax(torch.clone(loss).detach()*1, 0)
        # loss = torch.mm(loss.T, weight_loss)
        # # 方法2 WAIO10
        # alpha = 0  # 0
        # beta = 0.5  # 2
        # y = torch.sigmoid(logits).detach()
        # weight_loss = torch.ones((logits.shape[0], 1), device=logits.device)
        # already_classable = ((logits>0).view(-1)==labels)
        # weight_loss[already_classable] = ((abs(labels-y).detach()+alpha)[already_classable])*beta
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        # WAIO30
        base = 2
        weight_loss = get_weights_loss(logits, labels, base)
        loss = torch.mm(loss.T, weight_loss) / loss.shape[0]

        # # 方法3 WAIO12
        # alpha = 0 # 0
        # beta = 1  # 2
        # y = torch.sigmoid(logits).detach()
        # weight_loss = (abs(labels-y).detach()+alpha)*beta
        # weight_loss = torch.softmax(weight_loss, 0)
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        ret["loss"] = loss
        return ret


# # without_mod2
# class BertMatchModel_without_mod2(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         # config.hidden_size=768
#         self.pooler = nn.Linear(config.hidden_size*2, config.hidden_size)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')
#         self.tanh = nn.Tanh()
#         # --------------------------------------------------------------------------------------------------------------
#         # self.my_position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
#         self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
#         # --------------------------------------------------------------------------------------------------------------
#         self.att_layer = WAIO(config, need_para=True)

#         self.fusion = Gate_and_fusion(config.hidden_size)
#         self.dropout = nn.Dropout(config.hidden_dropout_prob)

#         # self.loss_aux = nn.CosineEmbeddingLoss()
#     def forward(self, input_ids, token_type_ids, attention_mask, position_ids, cls_sep_idxs, labels=None):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         # outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#         #                     position_ids=position_ids, output_attentions=True)
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#                             output_attentions=True)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         # att_layer_input = outputs.last_hidden_state+self.my_position_embeddings(position_ids)
#         att_layer_input = self.fixed_position_encoder(att_layer_input)
#         # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
#         # --------------------------------------------------------------------------------------------------------------
#         last, last_overlook, weight, weight_o = self.att_layer(
#             att_layer_input, att_layer_input, att_layer_input,
#             key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         # last_overlook += self.my_position_embeddings(position_ids)
#         # last, last_overlook, weight, weight_o = self.att_layer(
#         #     outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
#         #     key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs)

#         ret['weight'] = weight
#         ret['weight_o'] = weight_o

#         ret['weight_bert'] = outputs.attentions

#         cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)
#         # cls = last[:, 0]

#         # sep
#         sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
#         sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]

#         cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
#         cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)

#         # logits = self.classifier(self.dropout(self.tanh(self.pooler(cls_fin))))
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         ret['loss'] = loss
#         return ret


# # without_mod1
# class BertMatchModel_without_mod1(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         # config.hidden_size=768
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.tanh = nn.Tanh()

#     def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kargs):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#                             output_attentions=True)

#         cls = outputs.last_hidden_state[:, 0]
#         # cls = outputs.pooler_output

#         cls = self.pooler(cls)
#         cls = self.tanh(cls)
#         logits = self.classifier(cls)

#         ret = {'logits': logits}

#         # ret['representation'] = outputs.last_hidden_state
#         ret['weight_bert'] = outputs.attentions
#         if labels is None:
#             return ret
#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # without mod1
#         base = 1000
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

#         ret['loss'] = loss
#         return ret


######################################################################################################################################################################
