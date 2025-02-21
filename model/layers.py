import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F



def contrastive_loss(logits):
    n = logits.size(0)  # 获取批次大小

    # 生成标签 (0 到 n-1)
    labels = torch.arange(n, device=logits.device)

    # 计算 loss_i: 沿 axis=0 的交叉熵损失
    loss_i = F.cross_entropy(logits, labels)

    # 计算 loss_t: 沿 axis=1 的交叉熵损失
    # 转置 logits 矩阵以沿不同轴计算
    loss_t = F.cross_entropy(logits.T, labels)

    # 平均损失
    loss = (loss_i + loss_t) / 2

    return loss

class BaseRationaleFusion(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=config['dropout'])
        self.content2rationale_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=config['dropout'])

        self.rationale_useful_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                                           nn.ReLU(),
                                                           nn.Linear(config['mlp']['dims'][-1], 1),
                                                           nn.Sigmoid()
                                                           )
        self.LLM_judge_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                                nn.ReLU(),
                                                nn.Linear(config['mlp']['dims'][-1], 3))
        self.rationale_attention_pooling = AttentionPooling(config['emb_dim'])
        self.rationale_reweight_net = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                             nn.LayerNorm(config['mlp']['dims'][-1]),
                                             # nn.BatchNorm1d(config['mlp']['dims'][-1]),
                                             nn.ReLU(),
                                             nn.Dropout(config['dropout']),
                                             nn.Linear(config['mlp']['dims'][-1], 64),
                                             nn.LayerNorm(64),
                                             # nn.BatchNorm1d(64),
                                             nn.ReLU(),
                                             nn.Dropout(config['dropout']),
                                             nn.Linear(64, 1),
                                             nn.Sigmoid()
                                             )
        self.avg_pool = AvgPooling()

    def forward(self,
                content_feature,
                content_mask,
                rationale_feature,
                rationale_mask):
        rationale2content_feature = self.rationale2content_CA(query=rationale_feature,
                                                              key=content_feature,
                                                              value=content_feature,
                                                              mask=content_mask)[0]

        rationale2content_pooling_feature = self.avg_pool(rationale2content_feature, rationale_mask)
        content2rationale_feature = self.content2rationale_CA(query=content_feature,
                                                              key=rationale_feature,
                                                              value=rationale_feature,
                                                              mask=rationale_mask)[0]
        content2rationale_pooling_feature = self.avg_pool(content2rationale_feature, content_mask)
        rationale_useful_pred = self.rationale_useful_predictor(content2rationale_pooling_feature).squeeze(1)
        llm_judge_pred = self.LLM_judge_predictor(
            self.rationale_attention_pooling(rationale2content_feature, rationale_mask)
        )
        td_rationale_weight = self.rationale_reweight_net(content2rationale_pooling_feature)
        rationale2content_pooling_feature = td_rationale_weight * rationale2content_pooling_feature
        return rationale2content_pooling_feature,rationale_useful_pred, llm_judge_pred


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction='mean'):
        """
        Focal Loss的PyTorch实现，用于处理类别不平衡问题。

        参数:
            alpha (float): 正样本的权重系数，默认为0.25。负样本权重为1 - alpha。
            gamma (float): 调节难易样本的聚焦参数，默认为2。
            reduction (str): 损失汇总方式，可选'mean'（平均）或'sum'（求和）。
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # 确保targets的dtype为float，以便后续计算
        targets = targets.float()

        # 计算二元交叉熵损失（不进行汇总）
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # 计算概率pt：模型预测正确类别的概率
        pt = torch.exp(-ce_loss)

        # 根据目标类别选择对应的alpha系数
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)

        # 计算Focal Loss
        focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss

        # 根据reduction参数汇总损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class DualCrossAttentionLayer(nn.Module):
    def __init__(self, d_model, n_head,dropout=0.1):
        super(DualCrossAttentionLayer, self).__init__()
        self.attention1 = nn.MultiheadAttention(d_model, n_head,dropout=dropout,batch_first=True)
        self.attention2 = nn.MultiheadAttention(d_model, n_head,dropout=dropout,batch_first=True)

    def forward(self, input1,mask1,input2,mask2):
        out1 = self.attention1(query=input1,key=input2,value=input2,key_padding_mask=~mask2.bool())[0]
        out2 = self.attention2(query=input2,key=input1,value=input1,key_padding_mask=~mask1.bool())[0]
        return out1,out2


class DualCrossAttention(nn.Module):

    def __init__(self, d_model, n_head,layers=1,dropout=0.1):
        super(DualCrossAttention, self).__init__()
        self.dca_layers = nn.ModuleList([DualCrossAttentionLayer(d_model,n_head,dropout) for _ in range(layers)])

    def forward(self, input1,mask1,input2,mask2):
        for blk in self.dca_layers:
            input1,input2 = blk(input1,mask1,input2,mask2)
        return torch.cat([input1,input2],dim=1),torch.cat([mask1,mask2],dim=1)






class FeatureAggregation(nn.Module):

    def __init__(self, emb_dim):
        super(FeatureAggregation,self).__init__()
        self.attentionPooling = AttentionPooling(emb_dim)

    def forward(self, content_pooling_feature, image_pooling_feature, rationale1_pooling_feature,
                rationale2_pooling_feature):
        final_feature = torch.cat([content_pooling_feature.unsqueeze(1), image_pooling_feature.unsqueeze(1),
                                   rationale1_pooling_feature.unsqueeze(1), rationale2_pooling_feature.unsqueeze(1)], dim=1)
        return self.attentionPooling(final_feature)


class SelfAttentionFeatureAggregation(nn.Module):

    def __init__(self, emb_dim, nums_head, dropout=0.1, layers=1):
        super(SelfAttentionFeatureAggregation,self).__init__()
        self.attentionPooling = AttentionPooling(emb_dim)
        self.attention_layers = nn.ModuleList([nn.MultiheadAttention(emb_dim,nums_head,dropout=dropout,batch_first=True) for _ in range(layers)])

    def forward(self, content_pooling_feature, rationale1_pooling_feature,
                rationale2_pooling_feature):
        final_feature = torch.cat([content_pooling_feature.unsqueeze(1),
                                   rationale1_pooling_feature.unsqueeze(1), rationale2_pooling_feature.unsqueeze(1)],
                                  dim=1)

        for blk in self.attention_layers:
            final_feature = blk(query=final_feature, key=final_feature, value=final_feature)[0]
        return self.attentionPooling(final_feature)

class MultiViewAggregationLayer(nn.Module):
    def __init__(self, emb_dim, nums_view, dropout=0.1):
        super(MultiViewAggregationLayer,self).__init__()
        self.emb_dim = emb_dim
        self.aggregator = nn.Sequential(
            nn.Linear(emb_dim, nums_view, bias=False),
            nn.BatchNorm1d(nums_view),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, multi_features):
        attention_weight = nn.functional.softmax(self.aggregator(multi_features).transpose(1,2),dim=-1)
        return torch.bmm(attention_weight,multi_features)


class MultiViewAggregation(nn.Module):

    def __init__(self, emb_dim, nums_view, dropout=0.1,layers=1):
        super(MultiViewAggregation,self).__init__()
        self.MultiViewAggregationLayers = nn.ModuleList([MultiViewAggregationLayer(emb_dim,nums_view,dropout) for _ in range(layers)])
        self.attentionPooling = AttentionPooling(emb_dim)

    def forward(self, multi_features):
        for blk in self.MultiViewAggregationLayers:
            multi_features = blk(multi_features)

        return self.attentionPooling(multi_features)












class ImageCaptionGate(nn.Module):
    def __init__(self,config):
        super(ImageCaptionGate, self).__init__()
        self.attention = nn.MultiheadAttention(config['emb_dim'],config['num_heads'],batch_first=True)
        self.attention_pooling = AttentionPooling(config['emb_dim'])
        self.image_reweight_net = nn.Sequential(
            nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
            nn.BatchNorm1d(config['mlp']['dims'][-1]),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['mlp']['dims'][-1], 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, caption_feature,content_feature,caption_mask,content_mask):
        """
        :param caption_feature: shape (batch_size, seq_len, emb_dim)
        :param content_feature: shape (batch_size, seq_len, emb_dim)
        :param caption_mask: shape (batch_size, seq_len)
        :param content_mask: shape (batch_size, seq_len)
        :return: shape: (batch_size, 1)
        """
        content2caption_attn = self.attention(query = content_feature,
                                              key = caption_feature,
                                              value = caption_feature,
                                              key_padding_mask=~caption_mask.bool())[0]
        content2caption_attn_pooling = self.attention_pooling(content2caption_attn,attention_mask=content_mask)
        return self.image_reweight_net(content2caption_attn_pooling)


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


