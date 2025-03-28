import logging
import os
import time

import torch
from sympy.polys.polyconfig import query
from tensorboardX import SummaryWriter
from torch import nn
import torch.nn.functional as F
from torch.nn import BCELoss
from tqdm import tqdm
from transformers import AutoModel, BertModel, Swinv2Model

from Util import dataloader
from Util.Util import try_all_gpus, Recorder, Averager, data_to_device, MetricsRecorder, Decision
from model.layers import AttentionPooling, Classifier, AvgPooling, MultiHeadCrossAttention


def freeze_bert_params(model):
    for name, param in model.named_parameters():
        if name.startswith("encoder.layer.11"):
            param.requires_grad = True
        else:
            param.requires_grad = False

def freeze_pretrained_params(model):
    if isinstance(model,BertModel):
        freeze_bert_params(model)
    elif isinstance(model,Swinv2Model):
        # TODO freeze swin params
        pass




class ARGModel(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.content_encoder = BertModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        freeze_pretrained_params(self.content_encoder)
        self.rationale_encoder = BertModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        freeze_pretrained_params(self.rationale_encoder)

        self.td_rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=0.1)
        self.cs_rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=0.1)
        self.content2td_rationale_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=0.1)
        self.content2cs_rationale_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],dropout=0.1)

        # self.td_rationale2content_CA = SelfAttentionFeatureExtract(config['num_heads'] ,config['emb_dim'])
        # self.cs_rationale2content_CA = SelfAttentionFeatureExtract(config['num_heads'] ,config['emb_dim'])
        # self.content2td_rationale_CA = SelfAttentionFeatureExtract(config['num_heads'] ,config['emb_dim'])
        # self.content2cs_rationale_CA = SelfAttentionFeatureExtract(config['num_heads'] ,config['emb_dim'])



        self.td_rationale_useful_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                            nn.ReLU(),
                                            nn.Linear(config['mlp']['dims'][-1], 1),
                                            nn.Sigmoid()
                                            )
        self.cs_rationale_useful_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                            nn.ReLU(),
                                            nn.Linear(config['mlp']['dims'][-1], 1),
                                            nn.Sigmoid()
                                            )

        self.td_judge_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                              nn.ReLU(),
                                              nn.Linear(config['mlp']['dims'][-1], 3))
        self.cs_judge_predictor = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                              nn.ReLU(),
                                              nn.Linear(config['mlp']['dims'][-1], 3))

        self.td_attention_pooling = AttentionPooling(config['emb_dim'])
        self.cs_attention_pooling = AttentionPooling(config['emb_dim'])
        self.content_attention_pooling = AttentionPooling(config['emb_dim'])
        self.avg_pool = AvgPooling()

        self.td_reweight_net = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                                nn.LayerNorm(config['mlp']['dims'][-1]),
                                                #nn.BatchNorm1d(config['mlp']['dims'][-1]),
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

        self.cs_reweight_net = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
                                                nn.LayerNorm(config['mlp']['dims'][-1]),
                                                #nn.BatchNorm1d(config['mlp']['dims'][-1]),
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

        self.feature_aggregation = AttentionPooling(config['emb_dim'])

        self.classifier = Classifier(config['emb_dim'], config['mlp']['dims'], config['dropout'])







    def forward(self,**kwargs):
        """
        :param kwargs: {
            "content":"",
            "content_mask"
            "label":"0",
            "publish_date":1448118196000,
            "source_id":893,
            "td_rationale":"",
            "td_rationale_mask":
            "td_pred":"0",
            "td_acc":1,
            "cs_rationale":"无法确定。因为没有给出具体的消息内容，无法判断其真实性。",
            "cs_rationale_mask":
            "cs_pred":"0",
            "cs_acc":1,
            "caption":Optional[Tensor],
            "caption_mask":Optional[Tensor]
            "split":"train"
        }
        :return:
        """
        content_mask = kwargs['content_mask']
        content_features = self.content_encoder(kwargs['content'],attention_mask=content_mask).last_hidden_state

        td_rationale_mask = kwargs['td_rationale_mask']
        cs_rationale_mask = kwargs['cs_rationale_mask']

        td_rationale_features = self.rationale_encoder(kwargs['td_rationale'],attention_mask=td_rationale_mask).last_hidden_state
        cs_rationale_features = self.rationale_encoder(kwargs['cs_rationale'],attention_mask=cs_rationale_mask).last_hidden_state

        td2content_feature = self.td_rationale2content_CA(query=td_rationale_features,
                                                               key=content_features,
                                                               value=content_features,
                                                               mask=content_mask)[0]

        td2content_pooling_feature = self.avg_pool(td2content_feature,td_rationale_mask)

        cs2content_feature = self.cs_rationale2content_CA(query=cs_rationale_features,
                                                               key=content_features,
                                                               value=content_features,
                                                               mask=content_mask)[0]

        cs2content_pooling_feature = self.avg_pool(cs2content_feature,cs_rationale_mask)

        content2td_feature = self.content2td_rationale_CA(query=content_features,
                                                          key=td_rationale_features,
                                                          value=td_rationale_features,
                                                          mask=td_rationale_mask)[0]
        content2td_pooling_feature = self.avg_pool(content2td_feature,content_mask)

        content2cs_feature = self.content2cs_rationale_CA(query=content_features,
                                                          key=cs_rationale_features,
                                                          value=cs_rationale_features,
                                                          mask=cs_rationale_mask)[0]
        content2cs_pooling_feature = self.avg_pool(content2cs_feature,content_mask)

        td_rationale_useful_pred = self.td_rationale_useful_predictor(content2td_pooling_feature).squeeze(1)
        cs_rationale_useful_pred = self.cs_rationale_useful_predictor(content2cs_pooling_feature).squeeze(1)

        td_judge_pred = self.td_judge_predictor(
            self.td_attention_pooling(td_rationale_features,td_rationale_mask)
        )
        cs_judge_pred = self.cs_judge_predictor(
            self.cs_attention_pooling(cs_rationale_features,cs_rationale_mask)
        )

        td_rationale_weight = self.td_reweight_net(content2td_pooling_feature)
        cs_rationale_weight = self.cs_reweight_net(content2cs_pooling_feature)

        td2content_pooling_feature = td_rationale_weight * td2content_pooling_feature
        cs2content_pooling_feature = cs_rationale_weight * cs2content_pooling_feature

        content_features = self.content_attention_pooling(content_features,content_mask)

        all_features = torch.cat([
            content_features.unsqueeze(1),td2content_pooling_feature.unsqueeze(1),cs2content_pooling_feature.unsqueeze(1)
        ],dim=1)
        all_features = self.feature_aggregation(all_features)
        label_pred = self.classifier(all_features).squeeze(1)

        return {
            "classify_pred":label_pred,
            "td_rationale_weight":td_rationale_weight,
            "cs_rationale_weight":cs_rationale_weight,
            "td_rationale_useful_pred":td_rationale_useful_pred,
            "cs_rationale_useful_pred":cs_rationale_useful_pred,
            "td_judge_pred":td_judge_pred,
            "cs_judge_pred":cs_judge_pred
        }




def train_epoch(model, loss_fn, config, train_loader, optimizer, epoch, num_rationales,device):
    print('---------- epoch {} ----------'.format(epoch))
    model.train()
    train_data_iter = tqdm(train_loader)
    avg_loss_classify = Averager()



    for step_n, batch in enumerate(train_data_iter):
        batch_data = data_to_device(
            batch,
            device
        )
        label = batch_data['label']

        td_useful_label = batch_data['td_acc']
        cs_useful_label = batch_data['cs_acc']

        td_judge_label = batch_data['td_pred']
        cs_judge_label = batch_data['cs_pred']

        res = model(**batch_data)
        loss_classify = loss_fn(res['classify_pred'], label.float())

        loss_useful_fn = BCELoss()
        loss_hard_aux = loss_useful_fn(res['td_rationale_useful_pred'], td_useful_label.float()) + loss_useful_fn(
            res['cs_rationale_useful_pred'], cs_useful_label.float())

        loss_judge_fn = torch.nn.CrossEntropyLoss()
        loss_simple_aux = loss_judge_fn(res['td_judge_pred'],
                                             td_judge_label.long()) + loss_judge_fn(
            res['cs_judge_pred'], cs_judge_label.long())

        loss = loss_classify
        loss += config['train']['rationale_usefulness_evaluator_weight'] * loss_hard_aux / num_rationales
        loss += config['train']['llm_judgment_predictor_weight'] * loss_simple_aux / num_rationales

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss_classify.add(loss_classify.item(),len(label))

    return avg_loss_classify



class Trainer:

    def model_device_init(self):
        try:
            if self.config['train']['device']['use_cuda']:
                if self.config['train']['device']['use_multi']:
                    devices = try_all_gpus()
                    if len(devices) > 1:
                        self.logger.info(f'use multiple gpus: {devices}')
                        self.model = torch.nn.DataParallel(self.model)
                    device = devices[0] if devices else torch.device('cpu')
                    self.logger.info(f'Using device: {device}')
                else:
                    device = torch.device(
                        f'cuda:{self.config["train"]["device"]["gpu"]}' if torch.cuda.is_available() else 'cpu')
                    self.logger.info(f'Using device: {device}')
            else:
                device = torch.device('cpu')
                self.logger.info(f'Using CPU for training.')

            self.model = self.model.to(device)
            return device
        except Exception as e:
            self.logger.error(f"Error initializing device: {e}")
            raise



    def __init__(self, config):
        self.config = config
        self.model = ARGModel(config['model'])
        self.rationale_names = ['td','cs']
        self.writer = SummaryWriter(f'logs/tensorboard/arg_{config["dataset"]["name"]}')
        self.logger = logging.getLogger(__name__)
        self.running_tag = f'{config["model"]["name"]}/{config["model"]["version"]}/{config["dataset"]["name"]}'
        self.save_path = f'{config["train"]["save_param_dir"]}/{config["model"]["name"]}_{config["model"]["version"]}/{config["dataset"]["name"]}'
        self.save_path = os.path.join(config["train"]["save_param_dir"],
                                      f'{config["model"]["name"]}_{config["model"]["version"]}',
                                      config["dataset"]["name"])
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            os.makedirs(self.save_path)



    def train(self):
        st_tm = time.time()
        self.logger.info('start training......')
        self.logger.info('==================== start training ====================')
        device = self.model_device_init()
        loss_fn = BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['train']['lr'],
                                     weight_decay=self.config['train']['weight_decay'])
        # 获取早停监控的指标名称
        early_stopping_metric_name = self.config['train']['early_stopping_metric_name']

        # 根据指标名称动态设置 mode 参数
        if 'loss' in early_stopping_metric_name:
            mode = 'min'
        else:
            mode = 'max'
            self.logger.info(f"Assuming '{early_stopping_metric_name}' needs to be maximized. "
                             "If this is incorrect, please check the metric name.")

        # 初始化早停类
        recorder = Recorder(
            patience=self.config['train']['patience'],
            metric_name=early_stopping_metric_name,
            mode=mode
        )

        train_loader, val_loader, test_loader = dataloader.load_data(text_encoder_path=self.config['model']['text_encoder_path'],
                                                                             image_encoder_path=self.config['model']['image_encoder_path'],
                                                                             use_image=False,
                                                                            **self.config['dataset'])
        num_rationales = len(self.rationale_names)


        ed_tm = time.time()
        self.logger.info('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
        for epoch in range(self.config['train']['num_epochs']):
            avg_loss_classify = train_epoch(self.model, loss_fn, self.config, train_loader, optimizer, epoch, num_rationales,device)
            self.writer.add_scalar('train/loss_classify', avg_loss_classify.item(), epoch)
            self.logger.info('epoch: {}, train_loss_classify: {:.4f}'.format(epoch, avg_loss_classify.item()))
            self.logger.info('----- in val progress... -----')
            val_metrics = self.test(val_loader, device)
            mark = recorder.add(val_metrics['classifier'])

            self.writer.add_scalar('test/loss_classify', val_metrics['classifier']['loss_classify'], epoch)
            self.logger.info('epoch: {}, test_loss_classify: {:.4f}'.format(epoch, val_metrics["classifier"]['loss_classify']))

            if mark == Decision.SAVE:
                torch.save(self.model.state_dict(), os.path.join(self.save_path, 'model.pth'))
            if mark == Decision.ESCAPE:
                break

        self.logger.info('----- in test progress... -----')
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'model.pth')))
        test_metrics = self.test(test_loader, device)
        self.logger.info("test metrics: {}.".format(test_metrics['classifier']))
        self.writer.add_scalars(self.running_tag, test_metrics['classifier'])

        return test_metrics









    def test(self, dataloader,device):
        """
        :param dataloader: dataloader
        :return: {
            'loss_classify': avg_loss_classify.item(),
            'classifier': classifier_metrics,
            'llm_judgment': llm_judgment_metrics,
            'rationale_usefulness': rationale_usefulness_metrics
        }
        """
        loss_fn = torch.nn.BCELoss()
        self.model.eval()
        data_iter = tqdm(dataloader)
        avg_loss_classify = Averager()
        metrics_recorder = MetricsRecorder()

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data_to_device(
                    batch,
                    device
                )
                batch_label = batch_data['label']
                res = self.model(**batch_data)
                loss_classify = loss_fn(res['classify_pred'], batch_label.float())
                avg_loss_classify.add(loss_classify.item(),len(batch_label))
                metrics_recorder.record(batch_data, res,self.rationale_names)

        result_metrics = metrics_recorder.get_metrics()
        result_metrics['classifier']['loss_classify'] = avg_loss_classify.item()
        return result_metrics

































