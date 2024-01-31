import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertPreTrainedModel, RobertaModel, RobertaPreTrainedModel


# Roberta_baseline
class RobertaModel_baseline_SA(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, label=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        # ret['weight_bert'] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        ret["cls_hidden_state"] = cls
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if label is None:
            return ret

        loss = self.loss_func(logits.squeeze(), label.float())
        ret["loss"] = loss
        return ret


def generate_distribution(input_ids, vocab_size, min_main_score=0.7):
    one_hot = F.one_hot(input_ids, num_classes=vocab_size).to(input_ids.device)
    main_score = np.random.rand(1).item() * (1 - min_main_score) + min_main_score
    # 生成符合正态分布的随机数
    dis = torch.abs(torch.randn((input_ids.shape[0], input_ids.shape[1], vocab_size), device=input_ids.device))
    # 对随机数进行归一化，使它们的和为x
    dis *= abs(one_hot - 1)
    dis = F.normalize(dis, dim=-1, p=1) * (1 - main_score)
    dis += one_hot * main_score
    return dis


# Roberta_baseline
class RobertaModel_enhanced_SA(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size

    def forward(self, input_ids, attention_mask, label=None, is_train=True, **kwargs):
        ret = dict()
        outputs1 = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if label is None:
            return ret

        loss = self.loss_func(logits.squeeze(), label.float())
        ret["loss"] = loss

        if is_train:
            dis = generate_distribution(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=0.8)
            # emb = self.roberta.get_input_embeddings().weight.clone()
            emb = self.roberta.get_input_embeddings().weight
            input_emb = torch.matmul(dis, emb)
            outputs2 = self.roberta(inputs_embeds=input_emb, attention_mask=attention_mask, output_attentions=False)
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2.squeeze(), label.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret


class BertModel_baseline_SA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, label=None, **kwargs):
        ret = dict()
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        # ret['weight_bert'] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        ret["cls_hidden_state"] = cls
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if label is None:
            return ret

        loss = self.loss_func(logits.squeeze(), label.float())
        ret["loss"] = loss
        return ret


def generate_distribution(input_ids, vocab_size, min_main_score=0.7):
    one_hot = F.one_hot(input_ids, num_classes=vocab_size)
    main_score = np.random.rand(1).item() * (1 - min_main_score) + min_main_score
    # 生成符合正态分布的随机数
    dis = torch.abs(torch.randn((input_ids.shape[0], input_ids.shape[1], vocab_size), device=input_ids.device))
    # 对随机数进行归一化，使它们的和为x
    dis *= abs(one_hot - 1)
    dis = F.normalize(dis, dim=-1, p=1) * (1 - main_score)
    dis += one_hot * main_score
    return dis


class BertModel_enhanced_SA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()
        self.vocab_size = self.config.vocab_size

    def forward(self, input_ids, attention_mask, label=None, is_train=True, **kwargs):
        ret = dict()
        outputs1 = self.bert(input_ids=input_ids, attention_mask=attention_mask, output_attentions=False)
        cls = outputs1.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits

        if label is None:
            return ret

        loss = self.loss_func(logits.squeeze(), label.float())
        ret["loss"] = loss

        if is_train:
            dis = generate_distribution(input_ids=input_ids, vocab_size=self.vocab_size, min_main_score=0.8)
            # emb = self.roberta.get_input_embeddings().weight.clone()
            emb = self.bert.get_input_embeddings().weight
            input_emb = torch.matmul(dis, emb)
            outputs2 = self.bert(inputs_embeds=input_emb, attention_mask=attention_mask, output_attentions=False)
            cls2 = outputs2.last_hidden_state[:, 0]
            logits2 = self.classifier(self.tanh(self.pooler(cls2)))
            loss2 = self.loss_func(logits2.squeeze(), label.float())
            ret["loss"] += loss2
            ret["loss"] /= 2

        return ret
