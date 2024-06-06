# Roberta
import torch.nn as nn
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    AutoModel,
    AutoModelForPreTraining,
    AutoModelForQuestionAnswering,
    BertModel,
    BertPreTrainedModel,
    PreTrainedModel,
    RobertaModel,
)


class RobertaMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        ret["logits"] = logits

        if labels is None:
            return ret

        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret
        # return loss, logits
