import torch
import torch.nn as nn
from transformers import AlbertModel, AlbertPreTrainedModel, BertModel, BertPreTrainedModel, RobertaModel, XLMRobertaPreTrainedModel

from .mod import WAIO

######################################################################################################################################################################


# WAIO9 10
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


# class Gate_and_fusion(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#         self.fusion_same = nn.Linear(dim*3, dim)
#         self.gelu = nn.GELU()
#         self.tanh = nn.Tanh()
#     def forward(self, cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o):
#         same = self.gelu(self.fusion_same(torch.cat((cls, sep_s1, sep_s2), dim=1)))
#         return same, sep_s1_o, sep_s2_o


def log(x, n):
    n = torch.tensor(n)
    return torch.log(x) / torch.log(n)


def get_weights_loss(logits, labels, base):
    logits = logits.detach()
    labels = labels.detach()
    y = torch.sigmoid(logits)
    gama = 1 - 1 / base
    weight_loss = torch.ones((logits.shape[0], 1), device=logits.device)
    already_classable = ((logits > 0) == labels.bool()).squeeze()
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


class TrainablePositionalEncoding(nn.Module):
    """位置编码"""

    def __init__(self, seq_len=512, hidden_dim=768):
        super().__init__()
        self.weight = torch.zeros((seq_len, hidden_dim))
        # self.P = torch.nn.Embedding(seq_len, hidden_dim)
        X = torch.arange(seq_len, dtype=torch.float32).reshape(-1, 1) / torch.pow(
            10000, torch.arange(0, hidden_dim, 2, dtype=torch.float32) / hidden_dim
        )
        self.weight[:, 0::2] = torch.sin(X)
        self.weight[:, 1::2] = torch.cos(X)
        self.P = torch.nn.Embedding(seq_len, hidden_dim).from_pretrained(self.weight)

    def forward(self, X, position_ids):
        X = X + self.P(position_ids)
        return X


# WAIO9 10
class BertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert: BertModel = BertModel(config)
        # config.hidden_size=768
        # WAIO 60
        # self.pooler = nn.Linear(config.hidden_size*3, config.hidden_size)
        # WAIO 40
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

    def forward(self, input_ids, token_type_ids, attention_mask, cls_sep_idxs=None, position_ids=None, labels=None):
        if cls_sep_idxs is None:
            cls_sep_idxs = (
                ((input_ids == self.tokenizer.cls_token_id) | (input_ids == self.tokenizer.sep_token_id))
                .nonzero()[:, 1]
                .reshape(input_ids.shape[0], -1)
            )
            # print(input_ids[0])
            # print(cls_sep_idxs[0])
        # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
        # outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
        #                     position_ids=position_ids, output_attentions=True)
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True)
        ret = dict()
        # ret['representation'] = outputs.last_hidden_state
        att_layer_input = outputs.last_hidden_state
        # --------------------------------------------------------------------------------------------------------------
        # att_layer_input = outputs.last_hidden_state+self.my_position_embeddings(position_ids)
        att_layer_input_pos = self.fixed_position_encoder(att_layer_input)
        # att_layer_input_pos = att_layer_input
        # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
        # --------------------------------------------------------------------------------------------------------------
        last, last_overlook, weight, weight_o = self.att_layer(
            att_layer_input_pos,
            att_layer_input_pos,
            att_layer_input_pos,
            key_padding_mask=~attention_mask.bool(),
            need_weights=True,
            cls_sep_idxs=cls_sep_idxs,
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
        sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        # --------------------------------------------------------------------------------------------------------------
        cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        # # WAIO 60
        # cls_fin = torch.cat((cls_fin1, cls_fin2, torch.abs(cls_fin1-cls_fin2)), dim=1)
        # WAIO 40
        cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)

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
        # already_classable = ((logits>0)==labels).squeeze()
        # weight_loss[already_classable] = ((abs(labels-y).detach()+alpha)[already_classable])*beta
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        # WAIO30
        base = 1000
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


# without_mod2
class BertMatchModel_without_mod2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert: BertModel = BertModel(config)
        # config.hidden_size=768
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")
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

    def forward(self, input_ids, token_type_ids, attention_mask, position_ids, cls_sep_idxs, labels=None):
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
            att_layer_input, att_layer_input, att_layer_input, key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs
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
        sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]

        cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)

        # logits = self.classifier(self.dropout(self.tanh(self.pooler(cls_fin))))
        logits = self.classifier(self.tanh(self.pooler(cls_fin)))

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
        # already_classable = ((logits>0)==labels).squeeze()
        # weight_loss[already_classable] = ((abs(labels-y).detach()+alpha)[already_classable])*beta
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        # # WAIO30
        # base = 1000
        # weight_loss = get_weights_loss(logits, labels, base)
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        # # 方法3 WAIO12
        # alpha = 0 # 0
        # beta = 1  # 2
        # y = torch.sigmoid(logits).detach()
        # weight_loss = (abs(labels-y).detach()+alpha)*beta
        # weight_loss = torch.softmax(weight_loss, 0)
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        ret["loss"] = loss
        return ret


# without_mod1
class BertMatchModel_without_mod1(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert: BertModel = BertModel(config)
        # config.hidden_size=768
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kargs):
        # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True)

        cls = outputs.last_hidden_state[:, 0]
        # cls = outputs.pooler_output

        cls = self.pooler(cls)
        cls = self.tanh(cls)
        logits = self.classifier(cls)

        ret = {"logits": logits}

        # ret['representation'] = outputs.last_hidden_state
        ret["weight_bert"] = outputs.attentions
        if labels is None:
            return ret
        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())

        # without mod1
        base = 1000
        weight_loss = get_weights_loss(logits, labels, base)
        loss = torch.mm(loss.T, weight_loss) / loss.shape[0]

        ret["loss"] = loss
        return ret


#############################################################################################################


# # 消融实验1 不用sep_s1_o, sep_s2_o
# class Gate_and_fusion(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#         self.fusion_same = nn.Linear(dim*3, dim)
#         self.gelu = nn.GELU()
#         self.tanh = nn.Tanh()
#     def forward(self, cls, sep_s1, sep_s2):
#         same = self.gelu(self.fusion_same(torch.cat((cls, sep_s1, sep_s2), dim=1)))
#         return same

# class BertMatchModel(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         # config.hidden_size=768
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='none')
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
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
#                             output_attentions=True)
#         ret = dict()
#         att_layer_input = outputs.last_hidden_state
#         # --------------------------------------------------------------------------------------------------------------
#         # att_layer_input = outputs.last_hidden_state+self.my_position_embeddings(position_ids)
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
#         sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]

#         cls_fin = self.fusion(cls, sep_s1, sep_s2)
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))
#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         loss = self.loss_func(logits, labels.float())


#         base = 1000
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret


# BERT_baseline
class BertMatchModel_baseline(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert: BertModel = BertModel(config)
        # config.hidden_size=768
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None, **kargs):
        # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, output_attentions=True)

        cls = outputs.last_hidden_state[:, 0]
        # cls = outputs.pooler_output

        cls = self.pooler(cls)
        cls = self.tanh(cls)
        logits = self.classifier(cls)

        ret = {"logits": logits}

        # ret['representation'] = outputs.last_hidden_state
        ret["weight_bert"] = outputs.attentions
        if labels is None:
            return ret
        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())

        # # without mod1
        # base = 1000
        # weight_loss = get_weights_loss(logits, labels, base)
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        ret["loss"] = loss
        return ret


######################################################################################################################################################################


# Roberta
class RobertaMatchModel(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")
        # --------------------------------------------------------------------------------------------------------------
        self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # --------------------------------------------------------------------------------------------------------------
        self.att_layer = WAIO(config, need_para=True)
        self.fusion = Gate_and_fusion(config.hidden_size)

    def forward(self, input_ids, attention_mask, cls_sep_idxs, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        ret = dict()
        # ret['representation'] = outputs.last_hidden_state
        att_layer_input = outputs.last_hidden_state
        # --------------------------------------------------------------------------------------------------------------
        att_layer_input = self.fixed_position_encoder(att_layer_input)
        # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
        # --------------------------------------------------------------------------------------------------------------
        last, last_overlook, weight, weight_o = self.att_layer(
            att_layer_input, att_layer_input, att_layer_input, key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs
        )

        ret["weight"] = weight
        ret["weight_o"] = weight_o
        ret["weight_bert"] = outputs.attentions

        cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)

        # sep
        sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        # --------------------------------------------------------------------------------------------------------------
        cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
        logits = self.classifier(self.tanh(self.pooler(cls_fin)))
        # --------------------------------------------------------------------------------------------------------------

        ret["logits"] = logits

        if labels is None:
            return ret

        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())

        # WAIO30
        base = 1000
        weight_loss = get_weights_loss(logits, labels, base)
        loss = torch.mm(loss.T, weight_loss) / loss.shape[0]

        ret["loss"] = loss
        return ret


# --------------------------------------------------------------------------------------------------------------
# class Gate_and_fusion(nn.Module):
#     def __init__(self, dim) -> None:
#         super().__init__()
#         self.fusion_same = nn.Linear(dim*2, dim)
#         self.fusion_dif = nn.Linear(dim*2, dim)
#         self.gelu = nn.GELU()
#         self.tanh = nn.Tanh()
#     def forward(self, cls, my_cls, my_cls_o):
#         same = self.gelu(self.fusion_same(torch.cat((cls, my_cls), dim=1)))
#         cls_fin =  self.tanh(self.fusion_dif(torch.cat((same, my_cls_o), dim=1)))
#         return cls_fin

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
#         # self.fusion = Gate_and_fusion(config.hidden_size)


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
#         my_cls = last[:, 0]
#         my_cls_o = last_overlook[:, 0]
#         # --------------------------------------------------------------------------------------------------------------
#         # cls_fin = self.fusion(cls, my_cls, my_cls_o)

#         cls_fin = torch.cat([my_cls, my_cls_o], dim=1)
#         logits = self.classifier(self.tanh(self.pooler(cls_fin)))
#         # --------------------------------------------------------------------------------------------------------------

#         ret['logits'] = logits

#         if labels is None:
#             return ret

#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())

#         # WAIO30
#         base = 1000
#         weight_loss = get_weights_loss(logits, labels, base)
#         loss = torch.mm(loss.T, weight_loss)/loss.shape[0]


#         ret['loss'] = loss
#         return ret


# Roberta_without_mod1
class RobertaMatchModel_without_mod1(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # pooler_output = outputs[1]
        # logits = self.linear(self.dropout(pooler_output))

        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())

        # WAIO30
        base = 1000
        weight_loss = get_weights_loss(logits, labels, base)
        loss = torch.mm(loss.T, weight_loss) / loss.shape[0]

        ret["loss"] = loss
        return ret


# Roberta_withou_out_mod2
class RobertaMatchModel_without_mod2(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.pooler = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss(reduction="mean")
        # --------------------------------------------------------------------------------------------------------------
        self.fixed_position_encoder = FixedPositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # self.trainable_position_encoder = TrainablePositionalEncoding(config.max_position_embeddings, config.hidden_size)
        # --------------------------------------------------------------------------------------------------------------
        self.att_layer = WAIO(config, need_para=True)
        self.fusion = Gate_and_fusion(config.hidden_size)

    def forward(self, input_ids, attention_mask, cls_sep_idxs, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        ret = dict()
        # ret['representation'] = outputs.last_hidden_state
        att_layer_input = outputs.last_hidden_state
        # --------------------------------------------------------------------------------------------------------------
        att_layer_input = self.fixed_position_encoder(att_layer_input)
        # att_layer_input = self.trainable_position_encoder(att_layer_input, position_ids)
        # --------------------------------------------------------------------------------------------------------------
        last, last_overlook, weight, weight_o = self.att_layer(
            att_layer_input, att_layer_input, att_layer_input, key_padding_mask=~attention_mask.bool(), need_weights=True, cls_sep_idxs=cls_sep_idxs
        )

        ret["weight"] = weight
        ret["weight_o"] = weight_o
        ret["weight_bert"] = outputs.attentions

        cls = outputs.last_hidden_state[:, 0]  # shape: (batch, hidden_size)

        # sep
        sep_s1 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s1_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 1]]
        sep_s2 = last[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        sep_s2_o = last_overlook[list(range(input_ids.shape[0])), cls_sep_idxs[:, 2]]
        # --------------------------------------------------------------------------------------------------------------
        cls_fin1, cls_fin2 = self.fusion(cls, sep_s1, sep_s1_o, sep_s2, sep_s2_o)
        cls_fin = torch.cat((cls_fin1, cls_fin2), dim=1)
        logits = self.classifier(self.tanh(self.pooler(cls_fin)))
        # --------------------------------------------------------------------------------------------------------------

        ret["logits"] = logits

        if labels is None:
            return ret

        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())

        # # WAIO30
        # base = 1000
        # weight_loss = get_weights_loss(logits, labels, base)
        # loss = torch.mm(loss.T, weight_loss)/loss.shape[0]

        ret["loss"] = loss
        return ret


# Roberta_baseline
class RobertaMatchModel_baseline(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.tanh = nn.Tanh()
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None, **kwargs):
        ret = dict()
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, output_attentions=True)
        # pooler_output = outputs[1]
        # logits = self.linear(self.dropout(pooler_output))

        ret["weight_bert"] = outputs.attentions
        cls = outputs.last_hidden_state[:, 0]
        logits = self.classifier(self.tanh(self.pooler(cls)))
        ret["logits"] = logits
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        ret["loss"] = loss
        return ret


######################################################################################################################################################################


# # BERT
# class BertMatchModel_stage1(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         self.dropout_classifier = nn.Dropout(config.hidden_dropout_prob)
#         # config.hidden_size=768
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

#         self.att_layer = nn.MultiheadAttention(config.hidden_size, 1, batch_first=True, bias=False)
#         self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
#         self.dropout_pooler = nn.Dropout(config.hidden_dropout_prob)
#         self.tanh = nn.Tanh()

#     def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

#         # # 取出[CLS]表示, ==> [batchSize, 768]
#         # pooler_output = outputs.pooler_output
#         # # 把[CLS]表示送入linear分类得到打分 ==> [batchSize, 1]
#         # logits = self.linear(self.dropout(pooler_output))
#         # # logits = self.linear(pooler_output)
#         ret = dict()
#         last = self.att_layer(outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
#                                     key_padding_mask =~attention_mask.bool(), need_weights=False)[0]
#         cls = last[:, 0, :]  # shape: (batch, hidden_size)
#         ret['cls'] = cls
#         cls = self.pooler(self.dropout_pooler(cls))
#         logits = self.classifier(self.dropout_classifier(cls))

#         ret['logits'] = logits
#         ret['representation'] = outputs.last_hidden_state

#         if labels is None:
#             return ret
#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())
#         ret['loss'] = loss
#         return ret


# class BertMatchModel_stage2(BertMatchModel_stage1):
#     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         self.gate_and_fusion = Gate_and_fusion(config.hidden_size)

#     def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
#         ret = dict()

#         # ret['representation'] = outputs.last_hidden_state
#         last = self.att_layer(outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
#                                     key_padding_mask =~attention_mask.bool(), need_weights=False)[0]

#         cls = last[:, 0, :]  # shape: (batch, hidden_size)
#         last_ignore = self.att_layer(-outputs.last_hidden_state, outputs.last_hidden_state, outputs.last_hidden_state,
#                                     key_padding_mask =~attention_mask.bool(), need_weights=False)[0]
#         cls_ignore = last_ignore[:, 0, :]

#         ret['logits1'] = self.classifier(self.dropout_classifier(self.pooler(self.dropout_pooler(cls))))
#         ret['logits2'] = self.classifier(self.dropout_classifier(self.pooler(self.dropout_pooler(cls_ignore))))

#         cls = self.gate_and_fusion(cls, cls_ignore)

#         cls = self.pooler(self.dropout_pooler(cls))
#         logits = self.classifier(self.dropout_classifier(cls))

#         ret['logits'] = logits

#         ret['representation'] = outputs.last_hidden_state

#         if labels is None:
#             return ret
#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())
#         ret['loss'] = loss
#         return ret


# BERT
# class BertMatchModel(BertPreTrainedModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.bert: BertModel = BertModel(config)
#         self.dropout_classifier = nn.Dropout(config.hidden_dropout_prob)
#         # config.hidden_size=768
#         self.classifier = nn.Linear(config.hidden_size, 1)
#         self.loss_func = nn.BCEWithLogitsLoss(reduction='mean')

#         self.encoder_layer = myTransformerEncoderLayer(d_model=config.hidden_size, nhead=1, batch_first=True, reverse=False)
#         self.pooler = nn.Linear(768, 768)
#         self.dropout_pooler = nn.Dropout(config.hidden_dropout_prob)
#         self.tanh = nn.Tanh()

#         self.gate_and_fusion = nn.GRU(768, 768, 1, batch_first=True)
#     def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

#         # # 取出[CLS]表示, ==> [batchSize, 768]
#         # pooler_output = outputs.pooler_output
#         # # 把[CLS]表示送入linear分类得到打分 ==> [batchSize, 1]
#         # logits = self.linear(self.dropout(pooler_output))
#         # # logits = self.linear(pooler_output)
#         ret = dict()
#         # ret['representation'] = outputs.last_hidden_state
#         # self.encoder_layer.reverse = False
#         last = self.encoder_layer(outputs.last_hidden_state, attention_mask, reverse=False)
#         cls = last[:, 0, :]  # shape: (batch, hidden_size)

#         # self.encoder_layer.reverse = True
#         last_ignore = self.encoder_layer(outputs.last_hidden_state, attention_mask, reverse=True)
#         cls_ignore = last_ignore[:, 0, :]

#         cls = cls.unsqueeze(0).contiguous()
#         cls_ignore = cls_ignore.unsqueeze(1).contiguous()
#         cls = self.gate_and_fusion(cls_ignore, cls)
#         cls = cls[0].squeeze()

#         cls = self.pooler(self.dropout_pooler(cls))
#         logits = self.classifier(self.dropout_classifier(cls))

#         ret['logits'] = logits

#         # ret['representation'] = outputs.last_hidden_state

#         if labels is None:
#             return ret
#         # 交叉熵损失函数
#         loss = self.loss_func(logits, labels.float())
#         ret['loss'] = loss
#         return ret


# ERNIE
class ErnieMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        # logits = self.linear(pooler_output)
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits


class AlbertMatchModel(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.albert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits
