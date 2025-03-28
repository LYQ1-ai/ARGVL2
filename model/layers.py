import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sigmoid
from torch.distributions import Independent, Normal, kl_divergence
from torch.nn.functional import softplus



def contrastive_loss(logits,label):
    batch_size = logits.size(0)
    label_full = torch.zeros_like(logits, dtype=torch.long,device=logits.device)
    label_full[range(batch_size),range(batch_size)] = label
    return F.cross_entropy(logits,label_full.float())


class BaseRationaleFusionWORUP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=config['dropout'])


        self.LLM_judge_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                                nn.ReLU(),
                                                nn.Linear(config['mlp']['dims'][-1], 3))
        self.rationale_attention_pooling = AttentionPooling(config['emb_dim'])

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
        llm_judge_pred = self.LLM_judge_predictor(
            self.rationale_attention_pooling(rationale_feature, rationale_mask)
        )
        return rationale2content_pooling_feature,None, llm_judge_pred

class BaseRationaleFusionWOLJP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.content2rationale_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=config['dropout'])
        self.rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=config['dropout'])
        self.rationale_useful_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                                           nn.ReLU(),
                                                           nn.Linear(config['mlp']['dims'][-1], 1),
                                                           nn.Sigmoid()
                                                           )

        self.rationale_attention_pooling = AttentionPooling(config['emb_dim'])
        self.rationale_reweight_net = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                             nn.LayerNorm(config['mlp']['dims'][-1]),
                                             nn.ReLU(),
                                             nn.Dropout(config['dropout']),
                                             nn.Linear(config['mlp']['dims'][-1], 64),
                                             nn.LayerNorm(64),
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
        rationale_weight = self.rationale_reweight_net(content2rationale_pooling_feature)
        rationale2content_pooling_feature = rationale_weight * rationale2content_pooling_feature
        return rationale2content_pooling_feature,rationale_useful_pred,None

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
                                             nn.ReLU(),
                                             nn.Dropout(config['dropout']),
                                             nn.Linear(config['mlp']['dims'][-1], 64),
                                             nn.LayerNorm(64),
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
            self.rationale_attention_pooling(rationale_feature, rationale_mask)
        )
        rationale_weight = self.rationale_reweight_net(content2rationale_pooling_feature)
        rationale2content_pooling_feature = rationale_weight * rationale2content_pooling_feature
        return rationale2content_pooling_feature,rationale_useful_pred, llm_judge_pred

class RationaleFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config['ablation'] == 'LJP':
            self.rationale_fusion = BaseRationaleFusionWOLJP(config)
        elif config['ablation'] == 'RUP':
            self.rationale_fusion = BaseRationaleFusionWORUP(config)
        else:
            self.rationale_fusion = BaseRationaleFusion(config)
    def forward(self,**kwargs):
        return self.rationale_fusion(**kwargs)



class VAERationaleFusion(nn.Module):
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
        self.rationale_reweight_net = RationaleReweightNet(config['emb_dim'],64,dropout=config['dropout'])
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
        rationale_weight = self.rationale_reweight_net(r_pooling_feature=rationale2content_pooling_feature,c_pooling_feature=content2rationale_pooling_feature).unsqueeze(-1)
        rationale2content_pooling_feature = rationale_weight * rationale2content_pooling_feature
        return rationale2content_pooling_feature,rationale_useful_pred, llm_judge_pred



class VAEEncoder(nn.Module):
    def __init__(self, emb_dim, latent_dim,dropout=0.1):
        super().__init__()
        self.mu_net =  nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, latent_dim),
        )  # 输出均值
        self.logvar_net =  nn.Sequential(
            nn.Linear(emb_dim, emb_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim // 2, latent_dim),
        )  # 输出对数方差

    def forward(self, x):
        mu = self.mu_net(x)
        log_var = self.logvar_net(x)
        sigma = torch.exp(log_var * 0.5) + 1e-7
        return Independent(Normal(loc=mu, scale=sigma), 1)

class RationaleReweightNet(nn.Module):

    def __init__(self,emb_dim, latent_dim,dropout):
        super().__init__()
        self.rationale_vae = VAEEncoder(emb_dim, latent_dim,dropout)
        self.content_vae = VAEEncoder(emb_dim, latent_dim,dropout)

    def forward(self,r_pooling_feature,c_pooling_feature):
        """
        x shape (batch_size, emb_dim)
        """
        p_z1_given_r = self.rationale_vae(r_pooling_feature)
        p_z2_given_c = self.content_vae(c_pooling_feature)

        # 对称 KL 散度
        kl_1_2 = torch.distributions.kl_divergence(p_z1_given_r, p_z2_given_c)
        kl_2_1 = torch.distributions.kl_divergence(p_z2_given_c, p_z1_given_r)
        kl_symmetric = sigmoid(0.5 * (kl_1_2 + kl_2_1))

        return kl_symmetric


class FrequencyRationaleReweightNet(nn.Module):
    def __init__(self,emb_dim,dropout):
        super().__init__()




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
    def __init__(self, emb_dim, nums_view):
        super(MultiViewAggregationLayer,self).__init__()
        self.emb_dim = emb_dim
        self.aggregator = nn.Sequential(
            nn.Linear(emb_dim, nums_view, bias=False),
            nn.BatchNorm1d(nums_view),
            nn.ReLU(),
        )

    def forward(self, multi_features,mask=None):
        attention_scores = self.aggregator(multi_features).transpose(1,2)
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask.to(device=attention_scores.device),float('-inf'))
        attention_weight = nn.functional.softmax(attention_scores,dim=-1) # attention_weight shape = (batch_size,nums_view,nums_view)
        return torch.bmm(attention_weight,multi_features)





class MultiViewAggregation(nn.Module):

    def __init__(self, emb_dim, nums_view,layers=1):
        super(MultiViewAggregation,self).__init__()
        self.MultiViewAggregationLayers = nn.ModuleList([MultiViewAggregationLayer(emb_dim,nums_view) for _ in range(layers)])
        self.attentionPooling = AttentionPooling(emb_dim)

    def forward(self, multi_features,mask=None):
        for blk in self.MultiViewAggregationLayers:
            residual = multi_features
            # TODO 添加残差
            multi_features = blk(multi_features, mask)
            multi_features = residual + multi_features

        return self.attentionPooling(multi_features)


class MSALayer(nn.Module):

    def __init__(self, emb_dim):
        super(MSALayer,self).__init__()
        self.attention = nn.MultiheadAttention(emb_dim,num_heads=8,batch_first=True,bias=False,dropout=0.4)
        self.normal = nn.LayerNorm(emb_dim)

    def forward(self, x,mask=None):
        if mask is not None:
            mask = ~mask.bool()
        out = self.attention(query=x,key=x,value=x,key_padding_mask=mask)[0]
        return self.normal(out + x)



class MultiViewSAFeatureAggregation(nn.Module):
    def __init__(self, emb_dim,layers=4):
        super(MultiViewSAFeatureAggregation,self).__init__()
        self.MultiViewAggregationLayers = nn.ModuleList([MSALayer(emb_dim) for _ in range(layers)])
        self.attentionPooling = AttentionPooling(emb_dim)
    def forward(self, multi_features,mask=None):
        for blk in self.MultiViewAggregationLayers:
            residual = multi_features
            # TODO 添加残差
            multi_features = blk(multi_features, mask)
            multi_features = residual + multi_features

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


