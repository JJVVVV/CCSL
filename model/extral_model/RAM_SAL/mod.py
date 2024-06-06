import torch
import torch.nn as nn

# from transformers import RoFormerModel


class myDotProductAttention(nn.Module):
    """缩放点积注意力"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, queries, keys, values, key_padding_mask, cls_sep_idxs, need_weights=False):
        d = queries.shape[-1]

        # 设置transpose_b=True为了交换keys的最后两个维度
        scores = torch.bmm(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(d))
        key_padding_mask = key_padding_mask.repeat(1, scores.shape[1]).reshape((scores.shape[0], scores.shape[1], scores.shape[2]))

        if cls_sep_idxs.shape[1] == 3:  # bert
            for batch_idx in range(queries.shape[0]):
                # sep
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], cls_sep_idxs[batch_idx, 1] + 1 : cls_sep_idxs[batch_idx, 2] + 1] = True
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], 0] = True
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 2], cls_sep_idxs[batch_idx, 0] + 1 : cls_sep_idxs[batch_idx, 1] + 1] = True
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 2], 0] = True
                # # unused1
                # key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 0]+1, cls_sep_idxs[batch_idx, 1]+1:cls_sep_idxs[batch_idx, 2]+1] = True
                # key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 0]+1, 0] = True
                # key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1]+1, cls_sep_idxs[batch_idx, 0]+1:cls_sep_idxs[batch_idx, 1]+1] = True
                # key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1]+1, 0] = True
        else:  # roberta
            for batch_idx in range(queries.shape[0]):
                # sep
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], cls_sep_idxs[batch_idx, 1] + 1 : cls_sep_idxs[batch_idx, 3] + 1] = True
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], 0] = True
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 3], cls_sep_idxs[batch_idx, 0] + 1 : cls_sep_idxs[batch_idx, 2] + 1] = True
                key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 3], 0] = True

        scores[key_padding_mask] -= torch.inf
        self.attention_weights = torch.nn.functional.softmax(scores, dim=2)
        if need_weights:
            return torch.bmm(self.attention_weights, values), self.attention_weights
        else:
            return torch.bmm(self.attention_weights, values), None


# # WAIO
# class WAIO(nn.Module):
#     def __init__(self, config, need_para, bias=True) -> None:
#         super().__init__()
#         self.need_para = need_para
#         if need_para:
#             self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#             self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#         self.att = myDotProductAttention()
#     def forward(self, queries, keys, values, key_padding_mask, cls_sep_idxs, need_weights=False):
#         if self.need_para:
#             queries = self.Wq(queries)
#             keys = self.Wk(keys)
#         state, weight = self.att(queries, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
#         # 验证取负的作用
#         state_o, weight_o = self.att(-queries, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
#         if need_weights:
#             return state, state_o, weight, weight_o
#         else:
#             return state, state_o, None, None


# # ================================================================================================================================================================
# WAIO13
class WAIO(nn.Module):
    def __init__(self, config, need_para, bias=True) -> None:
        super().__init__()
        self.need_para = need_para
        if need_para:
            self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
            self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
        self.att = myDotProductAttention()

    def forward(self, queries, keys, values, key_padding_mask, cls_sep_idxs, need_weights=False):
        if self.need_para:
            queries_p = self.Wq(queries)
            queries_n = self.Wq(-queries)
            keys = self.Wk(keys)
        else:
            queries_p = queries
            queries_n = -queries
        state, weight = self.att(queries_p, keys, values, key_padding_mask=key_padding_mask, need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
        # 验证取负的作用
        state_o, weight_o = self.att(queries_n, keys, values, key_padding_mask=key_padding_mask, need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
        if need_weights:
            return state, state_o, weight, weight_o
        else:
            return state, state_o, None, None


# ================================================================================================================================================================

# class R:
#     """相对位置编码"""
#     def __init__(self, max_len=513, num_hiddens=768):
#         # 创建一个足够长的P
#         self.P = torch.zeros((1, max_len, num_hiddens))
#         X = torch.arange(max_len, dtype=torch.float32).reshape(
#             -1, 1) / torch.pow(10000, torch.arange(
#             0, num_hiddens, 2, dtype=torch.float32) / num_hiddens)
#         self.P[:, :, 0::2] = torch.sin(X)
#         self.P[:, :, 1::2] = torch.cos(X)
#         self.mid = max_len//2  # 256 0-255, 256, 257-512

#     def __call__(self, i, j) -> torch.Tensor:
#         # j-i: 0-256
#         return self.P[0, j-i+self.mid]


# class myDotProductAttention(nn.Module):
#     """缩放点积注意力"""
#     def __init__(self) -> None:
#         super().__init__()
#     def forward(self, queries, keys, values, key_padding_mask, cls_sep_idxs, need_weights=False):
#         d = queries.shape[-1]

#         # 设置transpose_b=True为了交换keys的最后两个维度
#         scores = torch.bmm(queries, keys.transpose(1,2)) / torch.sqrt(torch.tensor(d))


#         # 1表示被mask, 0表示没有被mask
#         key_padding_mask = key_padding_mask.repeat(1, scores.shape[1]).reshape((scores.shape[0], scores.shape[1], scores.shape[2]))

#         if cls_sep_idxs.shape[1]==3: # bert
#             for batch_idx in range(queries.shape[0]):
#                 # sep
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], cls_sep_idxs[batch_idx, 1]+1:cls_sep_idxs[batch_idx, 2]+1] = True
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], 0] = True
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 2], cls_sep_idxs[batch_idx, 0]+1:cls_sep_idxs[batch_idx, 1]+1] = True
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 2], 0] = True
#         else: # roberta
#             for batch_idx in range(queries.shape[0]):
#                 # sep
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], cls_sep_idxs[batch_idx, 1]+1:cls_sep_idxs[batch_idx, 3]+1] = True
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 1], 0] = True
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 3], cls_sep_idxs[batch_idx, 0]+1:cls_sep_idxs[batch_idx, 2]+1] = True
#                 key_padding_mask[batch_idx, cls_sep_idxs[batch_idx, 3], 0] = True

#         scores[key_padding_mask] -= torch.inf
#         self.attention_weights = torch.nn.functional.softmax(scores, dim=2)

#         if need_weights:
#             return torch.bmm(self.attention_weights, values), self.attention_weights
#         else:
#             return torch.bmm(self.attention_weights, values), None

# ================================================================================================================================================================


# class RoFormerSinusoidalPositionalEmbedding(nn.Embedding):
#     """This module produces sinusoidal positional embeddings of any length."""

#     def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None) -> None:
#         super().__init__(num_positions, embedding_dim)
#         self.weight = self._init_weight(self.weight)

#     @staticmethod
#     def _init_weight(out: nn.Parameter) -> nn.Parameter:
#         """
#         Identical to the XLM create_sinusoidal_embeddings except features are not interleaved. The cos features are in
#         the 2nd half of the vector. [dim // 2:]
#         """
#         n_pos, dim = out.shape
#         position_enc = np.array(
#             [[pos / np.power(10000, 2 * (j // 2) / dim) for j in range(dim)] for pos in range(n_pos)]
#         )
#         out.requires_grad = False  # set early to avoid an error in pytorch-1.8+
#         sentinel = dim // 2 if dim % 2 == 0 else (dim // 2) + 1
#         out[:, 0:sentinel] = torch.FloatTensor(np.sin(position_enc[:, 0::2]))
#         out[:, sentinel:] = torch.FloatTensor(np.cos(position_enc[:, 1::2]))
#         out.detach_()
#         return out

#     @torch.no_grad()
#     def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0) -> torch.Tensor:
#         """`input_ids_shape` is expected to be [bsz x seqlen]."""
#         bsz, seq_len = input_ids_shape[:2]
#         positions = torch.arange(
#             past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
#         )
#         return super().forward(positions)


# class RoFormerSelfAttention(nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
#             raise ValueError(
#                 f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
#                 f"heads ({config.num_attention_heads})"
#             )

#         self.num_attention_heads = config.num_attention_heads
#         self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
#         self.all_head_size = self.num_attention_heads * self.attention_head_size

#         self.query = nn.Linear(config.hidden_size, self.all_head_size)
#         self.key = nn.Linear(config.hidden_size, self.all_head_size)
#         self.value = nn.Linear(config.hidden_size, self.all_head_size)

#         self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

#         self.rotary_value = config.rotary_value
#         self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
#             config.max_position_embeddings, config.hidden_size // config.num_attention_heads
#         )

#     def transpose_for_scores(self, x):
#         new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
#         x = x.view(*new_x_shape)
#         return x.permute(0, 2, 1, 3)

#     def forward(
#         self,
#         hidden_states,
#         attention_mask=None,
#         head_mask=None,
#         output_attentions=False,
#     ):
#         mixed_query_layer = self.query(hidden_states)
#         query_layer = self.transpose_for_scores(mixed_query_layer)

#         key_layer = self.transpose_for_scores(self.key(hidden_states))
#         value_layer = self.transpose_for_scores(self.value(hidden_states))

#         sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1])[None, None, :, :]

#         if self.rotary_value:
#             query_layer, key_layer, value_layer = self.apply_rotary_position_embeddings(
#                 sinusoidal_pos, query_layer, key_layer, value_layer
#             )
#         else:
#             query_layer, key_layer = self.apply_rotary_position_embeddings(
#                 sinusoidal_pos, query_layer, key_layer
#             )


#         # Take the dot product between "query" and "key" to get the raw attention scores.
#         attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

#         attention_scores = attention_scores / math.sqrt(self.attention_head_size)
#         if attention_mask is not None:
#             # Apply the attention mask is (precomputed for all layers in RoFormerModel forward() function)
#             attention_scores = attention_scores + attention_mask

#         # Normalize the attention scores to probabilities.
#         attention_probs = nn.functional.softmax(attention_scores, dim=-1)

#         # This is actually dropping out entire tokens to attend to, which might
#         # seem a bit unusual, but is taken from the original Transformer paper.
#         attention_probs = self.dropout(attention_probs)

#         # Mask heads if we want to
#         if head_mask is not None:
#             attention_probs = attention_probs * head_mask

#         context_layer = torch.matmul(attention_probs, value_layer)

#         context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
#         new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
#         context_layer = context_layer.view(*new_context_layer_shape)

#         outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)


#         return outputs

#     @staticmethod
#     def apply_rotary_position_embeddings(sinusoidal_pos, query_layer, key_layer, value_layer=None):
#         # https://kexue.fm/archives/8265
#         # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         sin, cos = sinusoidal_pos.chunk(2, dim=-1)
#         # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
#         # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
#         # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
#         rotate_half_query_layer = torch.stack([-query_layer[..., 1::2], query_layer[..., ::2]], dim=-1).reshape_as(
#             query_layer
#         )
#         query_layer = query_layer * cos_pos + rotate_half_query_layer * sin_pos
#         # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
#         rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
#         key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
#         if value_layer is not None:
#             # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
#             rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
#                 value_layer
#             )
#             value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
#             return query_layer, key_layer, value_layer
#         return query_layer, key_layer


# # WAIO50
# class WAIO(nn.Module):
#     def __init__(self, config, need_para, bias=True) -> None:
#         super().__init__()
#         self.need_para = need_para
#         if need_para:
#             self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#             self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#         self.att = myDotProductAttention()
#         self.embed_positions = RoFormerSinusoidalPositionalEmbedding(
#             config.max_position_embeddings, config.hidden_size
#         )
#         self.rotary_value = True
#     def forward(self, hidden_states, key_padding_mask, cls_sep_idxs, need_weights=False):
#         if self.need_para:
#             queries_p = self.Wq(hidden_states)
#             queries_n = self.Wq(-hidden_states)
#             # queries = self.Wq(hidden_states)
#             keys = self.Wk(hidden_states)
#             values = hidden_states
#         else:
#             queries_p = hidden_states
#             queries_n = -hidden_states
#             # queries = hidden_states
#             keys = hidden_states
#             values = hidden_states

#         sinusoidal_pos = self.embed_positions(hidden_states.shape[:-1])[None, :, :]

#         if self.rotary_value:
#             queries_p, queries_n, keys, values = self.apply_rotary_position_embeddings(
#                 sinusoidal_pos, queries_p, queries_n, keys, values
#             )
#         else:
#              queries_p, queries_n, keys = self.apply_rotary_position_embeddings(
#                 sinusoidal_pos, queries_p, queries_n, keys
#             )

#         state, weight = self.att(queries_p, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
#         # 验证取负的作用
#         state_o, weight_o = self.att(queries_n, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
#         if need_weights:
#             return state, state_o, weight, weight_o
#         else:
#             return state, state_o, None, None

#     @staticmethod
#     def apply_rotary_position_embeddings(sinusoidal_pos, query_layer_p, query_layer_n, key_layer, value_layer=None):
#         # https://kexue.fm/archives/8265
#         # sin [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         # cos [batch_size, num_heads, sequence_length, embed_size_per_head//2]
#         sin, cos = sinusoidal_pos.chunk(2, dim=-1)
#         # sin [θ0,θ1,θ2......θd/2-1] -> sin_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         sin_pos = torch.stack([sin, sin], dim=-1).reshape_as(sinusoidal_pos)
#         # cos [θ0,θ1,θ2......θd/2-1] -> cos_pos [θ0,θ0,θ1,θ1,θ2,θ2......θd/2-1,θd/2-1]
#         cos_pos = torch.stack([cos, cos], dim=-1).reshape_as(sinusoidal_pos)
#         # rotate_half_query_layer [-q1,q0,-q3,q2......,-qd-1,qd-2]
#         rotate_half_query_layer_p = torch.stack([-query_layer_p[..., 1::2], query_layer_p[..., ::2]], dim=-1).reshape_as(
#             query_layer_p
#         )
#         query_layer_p = query_layer_p * cos_pos + rotate_half_query_layer_p * sin_pos
#         rotate_half_query_layer_n = torch.stack([-query_layer_n[..., 1::2], query_layer_n[..., ::2]], dim=-1).reshape_as(
#             query_layer_n
#         )
#         query_layer_n = query_layer_n * cos_pos + rotate_half_query_layer_n * sin_pos
#         # rotate_half_key_layer [-k1,k0,-k3,k2......,-kd-1,kd-2]
#         rotate_half_key_layer = torch.stack([-key_layer[..., 1::2], key_layer[..., ::2]], dim=-1).reshape_as(key_layer)
#         key_layer = key_layer * cos_pos + rotate_half_key_layer * sin_pos
#         if value_layer is not None:
#             # rotate_half_value_layer [-v1,v0,-v3,v2......,-vd-1,vd-2]
#             rotate_half_value_layer = torch.stack([-value_layer[..., 1::2], value_layer[..., ::2]], dim=-1).reshape_as(
#                 value_layer
#             )
#             value_layer = value_layer * cos_pos + rotate_half_value_layer * sin_pos
#             return query_layer_p, query_layer_n, key_layer, value_layer
#         return query_layer_p, query_layer_n, key_layer


# # WAIO14
# class PositionWiseFFN(nn.Module):
#     """基于位置的前馈网络"""
#     def __init__(self, ffn_num_input, ffn_num_hiddens, ffn_num_outputs,
#                  **kwargs):
#         super(PositionWiseFFN, self).__init__(**kwargs)
#         self.dense1 = nn.Linear(ffn_num_input, ffn_num_hiddens)
#         self.relu = nn.ReLU()
#         self.dense2 = nn.Linear(ffn_num_hiddens, ffn_num_outputs)

#     def forward(self, X):
#         return self.dense2(self.relu(self.dense1(X)))
# class AddNorm(nn.Module):
#     """残差连接后进行层规范化"""
#     def __init__(self, normalized_shape, dropout, **kwargs):
#         super(AddNorm, self).__init__(**kwargs)
#         self.dropout = nn.Dropout(dropout)
#         self.ln = nn.LayerNorm(normalized_shape)
#     def forward(self, X, Y):
#         # return self.ln(self.dropout(Y) + X)
#         return self.ln(Y)

# class WAIO(nn.Module):
#     def __init__(self, config, need_para, bias=True) -> None:
#         super().__init__()
#         self.need_para = need_para
#         if need_para:
#             self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#             self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#         self.att = myDotProductAttention()
#         self.addnorm1_1 = AddNorm(config.hidden_size, config.hidden_dropout_prob)
#         self.addnorm1_2 = AddNorm(config.hidden_size, config.hidden_dropout_prob)
#         self.addnorm2_1 = AddNorm(config.hidden_size, config.hidden_dropout_prob)
#         self.addnorm2_2 = AddNorm(config.hidden_size, config.hidden_dropout_prob)
#         self.ffn = PositionWiseFFN(config.hidden_size, config.hidden_size, config.hidden_size)
#     def forward(self, queries, keys, values, key_padding_mask, cls_sep_idxs, need_weights=False):
#         if self.need_para:
#             queries_p = self.Wq(queries)
#             queries_n = self.Wq(-queries)
#             keys = self.Wk(keys)
#         else:
#             queries_p = queries
#             queries_n = -queries
#         state, weight = self.att(queries_p, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
#         # 验证取负的作用
#         state_o, weight_o = self.att(queries_n, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights, cls_sep_idxs=cls_sep_idxs)
#         state = self.addnorm1_1(values, state); state_o = self.addnorm2_1(values, state_o)
#         state = self.addnorm1_2(state, self.ffn(state)); state_o = self.addnorm2_2(state_o, self.ffn(state_o))
#         if need_weights:
#             return state, state_o, weight, weight_o
#         else:
#             return state, state_o, None, None


# class myGRU(nn.Module):
#     def __init__(self) -> None:
#         super().__init__()


# class myDotProductAttention(nn.Module):
#     """缩放点积注意力"""
#     def __init__(self) -> None:
#         super().__init__()
#     def forward(self, queries, keys, values, key_padding_mask, need_weights=False):
#         d = queries.shape[-1]
#         # 设置transpose_b=True为了交换keys的最后两个维度
#         scores = torch.bmm(queries, keys.transpose(1,2)) / torch.sqrt(torch.tensor(d))
#         key_padding_mask = key_padding_mask.repeat(1, scores.shape[1]).reshape((scores.shape[0], scores.shape[1], scores.shape[2]))
#         scores[key_padding_mask] -= torch.inf
#         self.attention_weights = torch.nn.functional.softmax(scores, dim=2)
#         if need_weights:
#             return torch.bmm(self.attention_weights, values), self.attention_weights
#         else:
#             return torch.bmm(self.attention_weights, values), None

# # WAIO
# class WAIO(nn.Module):
#     def __init__(self, config, need_para, bias=True) -> None:
#         super().__init__()
#         self.need_para = need_para
#         if need_para:
#             self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#             self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#         self.att = myDotProductAttention()
#     def forward(self, queries, keys, values, key_padding_mask, need_weights=False):
#         if self.need_para:
#             queries = self.Wq(queries)
#             keys = self.Wk(keys)

#         state, weight = self.att(queries, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights)
#         state_o, weight_o = self.att(-queries, keys, values, key_padding_mask=key_padding_mask,
#                                 need_weights=need_weights)
#         if need_weights:
#             return state, state_o, weight, weight_o
#         else:
#             return state, state_o, None, None

# class myBertModel(BertModel):
#     def __init__(self, config, add_pooling_layer=True):
#         super().__init__(config, add_pooling_layer)
#     def forward(
#         self,
#         input_ids: Optional[torch.Tensor] = None,
#         attention_mask: Optional[torch.Tensor] = None,
#         token_type_ids: Optional[torch.Tensor] = None,
#         position_ids: Optional[torch.Tensor] = None,
#         head_mask: Optional[torch.Tensor] = None,
#         inputs_embeds: Optional[torch.Tensor] = None,
#         encoder_hidden_states: Optional[torch.Tensor] = None,
#         encoder_attention_mask: Optional[torch.Tensor] = None,
#         past_key_values: Optional[List[torch.FloatTensor]] = None,
#         use_cache: Optional[bool] = None,
#         output_attentions: Optional[bool] = None,
#         output_hidden_states: Optional[bool] = None,
#         return_dict: Optional[bool] = None,
#     ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
#         r"""
#         encoder_hidden_states  (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
#             Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
#             the model is configured as a decoder.
#         encoder_attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
#             Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
#             the cross-attention if the model is configured as a decoder. Mask values selected in `[0, 1]`:

#             - 1 for tokens that are **not masked**,
#             - 0 for tokens that are **masked**.
#         past_key_values (`tuple(tuple(torch.FloatTensor))` of length `config.n_layers` with each tuple having 4 tensors of shape `(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
#             Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

#             If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those that
#             don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of all
#             `decoder_input_ids` of shape `(batch_size, sequence_length)`.
#         use_cache (`bool`, *optional*):
#             If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
#             `past_key_values`).
#         """
#         output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
#         output_hidden_states = (
#             output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
#         )
#         return_dict = return_dict if return_dict is not None else self.config.use_return_dict

#         if self.config.is_decoder:
#             use_cache = use_cache if use_cache is not None else self.config.use_cache
#         else:
#             use_cache = False

#         if input_ids is not None and inputs_embeds is not None:
#             raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
#         elif input_ids is not None:
#             input_shape = input_ids.size()
#         elif inputs_embeds is not None:
#             input_shape = inputs_embeds.size()[:-1]
#         else:
#             raise ValueError("You have to specify either input_ids or inputs_embeds")

#         batch_size, seq_length = input_shape
#         device = input_ids.device if input_ids is not None else inputs_embeds.device

#         # past_key_values_length
#         past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

#         if attention_mask is None:
#             attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

#         if token_type_ids is None:
#             if hasattr(self.embeddings, "token_type_ids"):
#                 buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
#                 buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
#                 token_type_ids = buffered_token_type_ids_expanded
#             else:
#                 token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

#         # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
#         # ourselves in which case we just need to make it broadcastable to all heads.
#         extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape)

#         # If a 2D or 3D attention mask is provided for the cross-attention
#         # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
#         if self.config.is_decoder and encoder_hidden_states is not None:
#             encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
#             encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
#             if encoder_attention_mask is None:
#                 encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
#             encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
#         else:
#             encoder_extended_attention_mask = None

#         # Prepare head mask if needed
#         # 1.0 in head_mask indicate we keep the head
#         # attention_probs has shape bsz x n_heads x N x N
#         # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
#         # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
#         head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

#         embedding_output = self.embeddings(
#             input_ids=input_ids,
#             position_ids=position_ids,
#             token_type_ids=token_type_ids,
#             inputs_embeds=inputs_embeds,
#             past_key_values_length=past_key_values_length,
#         )
#         encoder_outputs = self.encoder(
#             embedding_output,
#             attention_mask=extended_attention_mask,
#             head_mask=head_mask,
#             encoder_hidden_states=encoder_hidden_states,
#             encoder_attention_mask=encoder_extended_attention_mask,
#             past_key_values=past_key_values,
#             use_cache=use_cache,
#             output_attentions=output_attentions,
#             output_hidden_states=output_hidden_states,
#             return_dict=return_dict,
#         )
#         sequence_output = encoder_outputs[0]
#         pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

#         if not return_dict:
#             return (sequence_output, pooled_output) + encoder_outputs[1:]

#         return BaseModelOutputWithPoolingAndCrossAttentions(
#             last_hidden_state=sequence_output,
#             pooler_output=pooled_output,
#             past_key_values=encoder_outputs.past_key_values,
#             hidden_states=encoder_outputs.hidden_states,
#             attentions=encoder_outputs.attentions,
#             cross_attentions=encoder_outputs.cross_attentions,
#         )


# class WAIO(nn.Module):
#     def __init__(self, config, need_para, bias=True) -> None:
#         super().__init__()
#         self.need_para = need_para
#         # self.dropout = nn.Dropout(dropout_prob)
#         if need_para:
#             self.Wq = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#             self.Wk = nn.Linear(config.hidden_size, config.hidden_size, bias=bias)
#         # self.Wv_overlook = nn.Linear(input_dim, output_dim, bias=bias)
#         # self.out_proj = nn.Linear(input_dim, output_dim, bias=bias)

#     def forward(self, queries, keys, values, key_padding_mask=None, need_weights=False):
#         if self.need_para:
#             queries = self.Wq(queries)
#             keys = self.Wk(keys)

#         d = queries.shape[-1]

#         scores = torch.bmm(queries, keys.transpose(1,2)) / torch.sqrt(torch.tensor(d))
#         scores_o = torch.bmm(-queries, keys.transpose(1,2)) / torch.sqrt(torch.tensor(d))
#         key_padding_mask = key_padding_mask.repeat(1, scores.shape[1]).reshape((scores.shape[0], scores.shape[1], scores.shape[2]))
#         scores[key_padding_mask] -= torch.inf
#         scores_o[key_padding_mask] -= torch.inf

#         attention_weights = torch.nn.functional.softmax(scores, dim=2)
#         attention_weights_o = torch.nn.functional.softmax(scores_o, dim=2)

#         states = torch.bmm(attention_weights, values)
#         states_o = torch.bmm(attention_weights_o, values)

#         if need_weights:
#             return states, states_o, attention_weights, attention_weights_o
#         else:
#             return states, states_o


# class WAIO(nn.Module):
#     def __init__(self, config, num_layer=0) -> None:
#         super().__init__()
#         self.reverse_att = nn.MultiheadAttention(config.hidden_size, 1, batch_first=True, bias=False)
#         self.multi_layer_trans = nn.ModuleList()
#         self.num_layer = num_layer
#         for i in range(num_layer):
#             if i!=num_layer-1:
#                 self.multi_layer_trans.append(
#                     nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, 4*config.hidden_size, 0.1, 'gelu', batch_first=True))
#             else:
#                 self.multi_layer_trans.append(
#                     nn.TransformerEncoderLayer(config.hidden_size, config.num_attention_heads, 4*config.hidden_size, 0.1, nn.Tanh(), batch_first=True))
#     def forward(self, X, attention_mask):
#         X_r = self.reverse_att(X, X, X, key_padding_mask=attention_mask, need_weights=False)[0]
#         for i in range(self.num_layer):
#             X_r = self.multi_layer_trans[i](X_r, src_key_padding_mask=attention_mask)
#         X_o = self.reverse_att(-X, X, X, key_padding_mask=attention_mask, need_weights=False)[0]
#         for i in range(self.num_layer):
#             X_o = self.multi_layer_trans[i](X_o, src_key_padding_mask=attention_mask)
#         return X_r, X_o


# class myTransformerEncoderLayer(nn.TransformerEncoderLayer):
#     def __init__(self, d_model: int, nhead: int, reverse: bool = False, dim_feedforward: int = 2048, dropout: float = 0.1, activation: Union[str, Callable[[Tensor], Tensor]] = nn.functional.relu, layer_norm_eps: float = 0.00001, batch_first: bool = False, norm_first: bool = False, device=None, dtype=None) -> None:
#         super().__init__(d_model, nhead, dim_feedforward, dropout, activation, layer_norm_eps, batch_first, norm_first, device, dtype)
#         self.reverse = reverse

#     def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
#         if self.reverse:
#             q = -x
#         else:
#             q = x
#         # print(self.reverse)
#         x = self.self_attn(q, x, x,
#                            attn_mask=attn_mask,
#                            key_padding_mask=key_padding_mask,
#                            need_weights=False)[0]
#         return self.dropout1(x)
#         # return super()._sa_block(x, attn_mask, key_padding_mask)

#     def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, reverse = False) -> Tensor:
#         self.reverse = reverse
#         return super().forward(src, src_mask, src_key_padding_mask)
