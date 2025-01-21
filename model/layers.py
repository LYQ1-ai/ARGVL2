import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
from timm.models.vision_transformer import Block


class DualCrossAttentionLayer(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
        super(DualCrossAttentionLayer, self).__init__()
        self.attention1 = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)
        self.attention2 = nn.MultiheadAttention(emb_dim, num_heads, dropout=dropout, batch_first=True)

    def forward(self, input1, input2, mask1, mask2):
        """
        :param input1: shape (batch_size, seq_len, emb_dim)
        :param input2: shape (batch_size,seq_len, emb_dim)
        :return: (batch_size, seq_len, emb_dim) ,(batch_size, seq_len, emb_dim)
        """
        return (self.attention1(query=input1, key=input2, value=input2, key_padding_mask=~mask2.bool())[0],
                self.attention2(query=input2, key=input1, value=input1, key_padding_mask=~mask1.bool())[0])


class DualCrossAttentionFusion(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout, num_layers):
        super(DualCrossAttentionFusion, self).__init__()
        self.dualCrossAttention = nn.ModuleList([
            DualCrossAttentionLayer(emb_dim, num_heads, dropout) for _ in range(num_layers)
        ])

    def forward(self, input1, input2, mask1, mask2):
        for blk in self.dualCrossAttention:
            input1, input2 = blk(input1, input2, mask1, mask2)

        return torch.cat([input1, input2], dim=1), torch.cat([mask1, mask2], dim=1)

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
        #self.attention = nn.Sequential(
        #    nn.Linear(emb_dim, int(emb_dim / 2)),
        #    nn.LayerNorm(int(emb_dim / 2)),
        #    nn.ReLU(),
        #    nn.Linear(int(emb_dim / 2), 1)
        #)
        self.attention = nn.Linear(emb_dim, 1)

    def forward(self, input_feature, attention_mask=None):
        """
        :param input_feature: shape (batch_size, seq_len, emb_dim)
        :param attention_mask: shape (batch_size, seq_len)
        :return: pooling_feature: shape (batch_size, emb_dim)
        """

        attention_scores = self.attention(input_feature).squeeze(-1)  # shape (batch_size, seq_len)
        if attention_mask is not None:
            attention_scores = attention_scores.masked_fill(~attention_mask.bool(), float('-inf'))
        attention_weights = nn.functional.softmax(attention_scores, dim=-1)  # shape (batch_size, seq_len)
        attention_pooling_feature = torch.bmm(attention_weights.unsqueeze(1), input_feature).squeeze(
            1)  # shape (batch_size, emb_dim)
        return attention_pooling_feature


class AvgPooling(nn.Module):

    #def forward(self, input_feature, attention_mask):
    #    """
    #    :param input_feature: shape (batch_size, seq_len, emb_dim)
    #    :param attention_mask: shape (batch_size, seq_len)
    #    :return: pooling_feature: shape (batch_size, emb_dim)
    #    """
    #    # 扩展 attention_mask 的维度以匹配 input_feature 的维度
    #    expanded_attention_mask = attention_mask.unsqueeze(-1)  # (batch_size, seq_len, 1)

    #    # 将 attention_mask 应用于 input_feature
    #    masked_input_features = input_feature * expanded_attention_mask

    #    # 对每个样本的特征求和，并除以有效的（未被mask遮挡的）特征数量
    #    sum_masked_features = masked_input_features.sum(dim=1)  # (batch_size, emb_dim)
    #    num_valid_features = expanded_attention_mask.sum(dim=1)  # (batch_size, 1)

    #    # 防止除以0的情况发生
    #    num_valid_features = torch.where(num_valid_features == 0,
    #                                     torch.ones_like(num_valid_features, device=num_valid_features.device),
    #                                     num_valid_features)

    #    # 计算平均值
    #    pooling_feature = sum_masked_features / num_valid_features

    #    return pooling_feature

    def forward(self, input_feature, attention_mask):
        return torch.mean(input_feature, dim=1)



class SigmoidWithLearnableBeta(nn.Module):
    def __init__(self, init_beta=1.0):
        super(SigmoidWithLearnableBeta, self).__init__()
        # 使用nn.Parameter定义可学习的参数beta，并初始化
        self.beta = nn.Parameter(torch.tensor(init_beta))

    def forward(self, x):
        # 应用带有可学习beta的Sigmoid函数
        return torch.sigmoid(self.beta * x)


class ImageCaptionGate(nn.Module):
    def __init__(self):
        super(ImageCaptionGate, self).__init__()

    def forward(self, content_pooling_feature, caption_pooling_feature):
        """
        :param content_pooling_feature: shape (batch_size, emb_dim)
        :param caption_pooling_feature: shape (batch_size, emb_dim)
        :return: similarity: shape (batch_size)
        """
        similarity = nn.functional.cosine_similarity(content_pooling_feature, caption_pooling_feature, dim=1)
        return similarity


class FeatureAggregation(nn.Module):

    def __init__(self, emb_dim):
        super(FeatureAggregation, self).__init__()
        self.imageGate = ImageCaptionGate()
        self.maskAttention = AttentionPooling(emb_dim)

    def forward(self, content_pooling_feature, caption_pooling_feature, image_pooling_feature, FTR2_pooling_feature,
                FTR3_pooling_feature):
        image_gate_value = self.imageGate(content_pooling_feature, caption_pooling_feature).unsqueeze(
            -1)  # shape (batch_size,1)
        image_pooling_feature = image_gate_value * image_pooling_feature
        final_feature = torch.cat([content_pooling_feature.unsqueeze(1), image_pooling_feature.unsqueeze(1),
                                   FTR2_pooling_feature.unsqueeze(1), FTR3_pooling_feature.unsqueeze(1)], dim=1)
        return self.maskAttention(final_feature)[0]

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, emb_dim, num_heads, dropout):
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


