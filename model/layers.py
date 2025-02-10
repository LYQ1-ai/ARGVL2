import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math



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


class WeightedBCELoss(nn.Module):
    def __init__(self, weight=0.0, reduce_class=0, reduction='mean', eps=1e-8):
        super(WeightedBCELoss, self).__init__()
        self.weight = weight
        self.reduce_class = reduce_class
        self.reduction = reduction
        self.eps = eps

    def forward(self, y_pred, y_true):
        # Clamp predictions to avoid numerical instability
        y_pred = torch.clamp(y_pred, min=self.eps, max=1 - self.eps)

        log_y_hat = torch.log(y_pred)
        log_1_reduce_y_hat = torch.log(1 - y_pred)

        # Apply weights based on reduce_class
        if self.reduce_class == 0:
            # Reduce loss for class 0 (negative samples: y_true=0)
            loss_negative = (1 - y_true) * (y_pred ** self.weight) * log_1_reduce_y_hat
            loss_positive = y_true * log_y_hat
        else:
            # Reduce loss for class 1 (positive samples: y_true=1)
            loss_positive = y_true * ((1 - y_pred) ** self.weight) * log_y_hat
            loss_negative = (1 - y_true) * log_1_reduce_y_hat

        loss = -(loss_positive + loss_negative)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss





class FeatureAggregation(nn.Module):

    def __init__(self, emb_dim):
        super(FeatureAggregation,self).__init__()
        self.attentionPooling = AttentionPooling(emb_dim)

    def forward(self, content_pooling_feature, image_pooling_feature, rationale1_pooling_feature,
                rationale2_pooling_feature):
        final_feature = torch.cat([content_pooling_feature.unsqueeze(1), image_pooling_feature.unsqueeze(1),
                                   rationale1_pooling_feature.unsqueeze(1), rationale2_pooling_feature.unsqueeze(1)], dim=1)
        return self.attentionPooling(final_feature)

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


