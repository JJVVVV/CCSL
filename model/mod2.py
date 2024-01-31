import torch
import torch.nn as nn

class ScaledDotProductAttention(nn.Module):
    """缩放点积注意力"""
    def __init__(self) -> None:
        super().__init__()
    def forward(self, queries, keys, values, key_padding_mask, need_weights=False):
        d = queries.shape[-1]

        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1,2)) / torch.sqrt(torch.tensor(d))
        key_padding_mask = key_padding_mask.repeat(1, scores.shape[1]).reshape((scores.shape[0], scores.shape[1], scores.shape[2]))
        scores[key_padding_mask] -= torch.inf
        attention_weights = torch.nn.functional.softmax(scores, dim=2)
        if need_weights: 
            return torch.bmm(attention_weights, values), attention_weights
        else:
            return torch.bmm(attention_weights, values), None
        

class AttnNoProjVal(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=True)
        self.dotAtt = ScaledDotProductAttention()
    def forward(self, hidden_states, key_padding_mask, need_weights=False):
        queries = self.Wq(hidden_states)
        keys = self.Wk(hidden_states)
        values = hidden_states

        state, weight = self.dotAtt(queries, keys, values, key_padding_mask=key_padding_mask, 
                                need_weights=need_weights)
        return state, weight
