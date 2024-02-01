import os
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from transformers import BertModel, RobertaModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions


class BertModel_multi_classify(BertModel):
    def __init__(self, config, num_classification, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_classification)
        self.loss_func = nn.CrossEntropyLoss()
        self.vocab_size = self.config.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, num_classification, *model_args, **kwargs):
        kwargs["num_classification"] = num_classification
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
        # model.min_threshold = min_threshold

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.view(-1))
        ret["loss"] = loss

        return ret

class RobertaModel_multi_classify(RobertaModel):
    def __init__(self, config, num_classification, add_pooling_layer=True):
        super().__init__(config, add_pooling_layer)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, num_classification)
        self.loss_func = nn.CrossEntropyLoss()
        self.vocab_size = self.config.vocab_size

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | os.PathLike | None, num_classification, *model_args, **kwargs):
        kwargs["num_classification"] = num_classification
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        return model
        # model.min_threshold = min_threshold

    def forward(
        self,
        input_ids: Tensor | None = None,
        attention_mask: Tensor | None = None,
        token_type_ids: Tensor | None = None,
        position_ids: Tensor | None = None,
        labels: Tensor = None,
    ) -> Tuple[Tensor] | BaseModelOutputWithPoolingAndCrossAttentions:
        ret = dict()
        outputs1 = super().forward(input_ids, attention_mask, token_type_ids, position_ids, output_attentions=False, output_hidden_states=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.view(-1))
        ret["loss"] = loss

        return ret

