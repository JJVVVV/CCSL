import torch.nn as nn
from transformers import (AlbertModel, AlbertPreTrainedModel, AutoModel,
                          AutoModelForPreTraining,
                          AutoModelForQuestionAnswering, BertModel,
                          BertPreTrainedModel, PretrainedConfig, PreTrainedModel, RobertaModel)

from .MatchModel_binary_classification import BertMatchModel_stage1
from .mod import myTransformerEncoderLayer


class BertMatchModel_stage2(BertMatchModel_stage1):
    def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.afterFineTune: BertMatchModel_stage1 = BertMatchModel_stage1(config)
        # self.gate_and_fusion = nn.GRU(768, 768, 1, batch_first=True)
        # self.

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
        ret = dict()
        
        outputs = self.afterFineTune(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

        # self.afterFineTune.encoder_layer.reverse = False
        # last = self.afterFineTune.encoder_layer(outputs['representation'])
        # cls = last[:, 0, :]  # shape: (batch, hidden_size)
        cls = outputs['cls']

        last_ignore = self.afterFineTune.att_layer(outputs['representation'], reverse=True)
        cls_ignore = last_ignore[:, 0, :]

        cls = cls.unsqueeze(0).contiguous()
        cls_ignore = cls_ignore.unsqueeze(1).contiguous()
        cls = self.gate_and_fusion(cls_ignore, cls)
        cls = cls[0].squeeze()

        cls = self.afterFineTune.pooler(self.afterFineTune.dropout_pooler(cls))
        logits = self.afterFineTune.classifier(self.afterFineTune.dropout_classifier(cls))

        ret['logits'] = logits

        # ret['representation'] = outputs.last_hidden_state

        if labels is None:
            return ret
        # 交叉熵损失函数
        loss = self.afterFineTune.loss_func(logits, labels.float())
        ret['loss'] = loss
        return ret

# class BertMatchModel_stage2(BertMatchModel_stage1):
#     def __init__(self, config: PretrainedConfig, *inputs, **kwargs):
#         super().__init__(config, *inputs, **kwargs)
#         self.afterFineTune: BertMatchModel_stage1 = BertMatchModel_stage1(config)
#         self.gate_and_fusion = nn.GRU(768, 768, 1, batch_first=True)

#     def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
#         # 得到BERT输出的结果 ==> outputs: last_hidden_state, pooler_output, ....
#         ret = dict()
        
#         outputs = self.afterFineTune(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)

#         # self.afterFineTune.encoder_layer.reverse = False
#         # last = self.afterFineTune.encoder_layer(outputs['representation'])
#         # cls = last[:, 0, :]  # shape: (batch, hidden_size)
#         cls = outputs['cls']

#         last_ignore = self.afterFineTune.encoder_layer(outputs['representation'], reverse=True)
#         cls_ignore = last_ignore[:, 0, :]

#         cls = cls.unsqueeze(0).contiguous()
#         cls_ignore = cls_ignore.unsqueeze(1).contiguous()
#         cls = self.gate_and_fusion(cls_ignore, cls)
#         cls = cls[0].squeeze()

#         cls = self.afterFineTune.pooler(self.afterFineTune.dropout_pooler(cls))
#         logits = self.afterFineTune.classifier(self.afterFineTune.dropout_classifier(cls))

#         ret['logits'] = logits

#         # ret['representation'] = outputs.last_hidden_state

#         if labels is None:
#             return ret
#         # 交叉熵损失函数
#         loss = self.afterFineTune.loss_func(logits, labels.float())
#         ret['loss'] = loss
#         return ret