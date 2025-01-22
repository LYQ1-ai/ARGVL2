import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from timm.models.vision_transformer import Block



class SelfAttentionFeatureExtract(torch.nn.Module):
    def __init__(self, multi_head_num, input_size):
        super(SelfAttentionFeatureExtract, self).__init__()
        self.attention = MultiHeadedAttention(multi_head_num, input_size)
    def forward(self, query,key,value, mask=None):
        mask = mask.view(mask.size(0), 1, 1, mask.size(-1))

        feature, attn = self.attention(query=query,
                                 value=value,
                                 key=key,
                                 mask=mask
                                 )
        return feature, attn


class Attention(torch.nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / math.sqrt(query.size(-1))

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        p_attn = F.softmax(scores, dim=-1)

        if dropout is not None:
            p_attn = dropout(p_attn)

        return torch.matmul(p_attn, value), p_attn

class MultiHeadedAttention(torch.nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = torch.nn.ModuleList([torch.nn.Linear(d_model, d_model) for _ in range(3)])
        self.output_linear = torch.nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        if mask is not None:
            mask = mask.repeat(1, self.h, 1, 1)
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        # print('x shape after self attention: {}'.format(x.shape))

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x), attn

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims,dropout):
        super(Classifier, self).__init__()
        layers = []
        for i in hidden_dims:
            layers.append(nn.Linear(input_dim, i))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = i

        layers.append(nn.Linear(input_dim, 1))
        layers.append(nn.Sigmoid())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class AttentionPooling(nn.Module):

    def __init__(self, emb_dim):
        super(AttentionPooling, self).__init__()
        self.attention = nn.Linear(emb_dim, 1)

    def forward(self, input_feature, attention_mask=None):
        """
        :param input_feature: shape (batch_size, seq_len, emb_dim)
        :param attention_mask: shape (batch_size, seq_len)
        :return: pooling_feature: shape (batch_size, emb_dim)
        """

        attention_scores = self.attention(input_feature).squeeze(-1)  # shape (batch_size, seq_len)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # shape (batch_size, seq_len)
        attention_pooling_feature = torch.bmm(attention_weights.unsqueeze(1), input_feature).squeeze(1)  # shape (batch_size, emb_dim)
        return attention_pooling_feature




class AvgPooling(nn.Module):

    def forward(self, input_feature, attention_mask=None):
        """
        :param input_feature: shape (batch_size, seq_len, emb_dim)
        :param attention_mask: shape (batch_size, seq_len)
        :return: pooling_feature: shape (batch_size, emb_dim)
        """
        return torch.mean(input_feature, dim=1)





class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout=0.0):
        super(MultiHeadCrossAttention, self).__init__()
        self.attention = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, query, key,value, mask=None):
        """
        :param query: shape (batch_size, seq_len1, emb_dim)
        :param key:  shape (batch_size, seq_len2, emb_dim)
        :param value: shape (batch_size, seq_len2, emb_dim)
        :param mask: shape (batch_size, seq_len2)
        :return: shape (batch_size, seq_len1, emb_dim)
        """
        return self.attention(query, key, value, key_padding_mask=~mask.bool() if mask is not None else None)


