import logging
import os
import time

import torch
from sympy.polys.polyconfig import query
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, BertModel, Swinv2Model

from Util import dataloader
from Util.Util import try_all_gpus, Recorder, Averager, data_to_device, MetricsRecorder
from model.layers import MultiHeadCrossAttention, AvgPooling, AttentionPooling, Classifier


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
        self.content_encoder = AutoModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        freeze_pretrained_params(self.content_encoder)
        self.rationale_encoder = AutoModel.from_pretrained(config['text_encoder_path']).requires_grad_(False)
        freeze_pretrained_params(self.rationale_encoder)

        self.td_rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],config['dropout'])
        self.cs_rationale2content_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],config['dropout'])
        self.content2td_rationale_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],config['dropout'])
        self.content2cs_rationale_CA = MultiHeadCrossAttention(config['emb_dim'],config['num_heads'],config['dropout'])

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

        self.td_reweight_net = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
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

        self.cs_reweight_net = nn.Sequential(nn.Linear(config['emb_dim'], config['mlp']['dims'][-1]),
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

        td2content_pooling_feature = torch.mean(td2content_feature,dim=1)

        cs2content_feature = self.cs_rationale2content_CA(query=cs_rationale_features,
                                                               key=content_features,
                                                               value=content_features,
                                                               mask=content_mask)[0]

        cs2content_pooling_feature = torch.mean(cs2content_feature,dim=1)

        content2td_feature = self.content2td_rationale_CA(query=content_features,
                                                          key=td_rationale_features,
                                                          value=td_rationale_features,
                                                          mask=td_rationale_mask)[0]
        content2td_pooling_feature = torch.mean(content2td_feature,dim=1)

        content2cs_feature = self.content2cs_rationale_CA(query=content_features,
                                                          key=cs_rationale_features,
                                                          value=cs_rationale_features,
                                                          mask=cs_rationale_mask)[0]
        content2cs_pooling_feature = torch.mean(content2cs_feature,dim=1)

        td_rationale_useful_pred = self.td_rationale_useful_predictor(content2td_pooling_feature).squeeze(1)
        cs_rationale_useful_pred = self.cs_rationale_useful_predictor(content2cs_pooling_feature).squeeze(1)

        td_judge_pred = self.td_judge_predictor(
            self.td_attention_pooling(td_rationale_features,td_rationale_mask)
        ).squeeze(1)
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
        all_features = self.feature_aggregation(all_features,None)
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


def train_epoch(model, loss_fn, config, train_loader, optimizer, epoch, num_rationales):
    print('---------- epoch {} ----------'.format(epoch))
    model.train()
    train_data_iter = tqdm(train_loader)
    avg_loss_classify = Averager()
    loss_useful_fn = torch.nn.BCELoss()
    loss_judge_fn = torch.nn.CrossEntropyLoss()

    for step_n, batch in enumerate(train_data_iter):
        batch_data = data_to_device(
            batch,
            config['train']['use_cuda']
        )
        label = batch_data['label']

        td_useful_label = batch_data['td_acc']
        cs_useful_label = batch_data['cs_acc']

        td_judge_label = batch_data['td_pred']
        cs_judge_label = batch_data['cs_pred']

        res = model(**batch_data)
        loss_classify = loss_fn(res['classify_pred'], label.float())


        loss_hard_aux = loss_useful_fn(res['td_rationale_useful_pred'], td_useful_label.float()) + loss_useful_fn(
            res['cs_rationale_useful_pred'], cs_useful_label.float())


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
        if self.config['train']['use_cuda']:
            devices = try_all_gpus()
            if len(devices) > 1:
                print('use multiple gpus')
                self.model = torch.nn.DataParallel(self.model)
            else:
                self.model = self.model.cuda()



    def __init__(self, config):
        self.config = config
        self.model = ARGModel(config['model'])
        self.num_rationales = 2
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

        if self.config['train']['use_cuda']:
            devices = try_all_gpus()
            if len(devices) > 1:
                print('use multiple gpus')
                self.model = torch.nn.DataParallel(self.model)

            self.model = self.model.cuda()

        loss_fn = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['train']['lr'],
                                     weight_decay=self.config['train']['weight_decay'])
        early_stopping_metric_name = self.config['train']['early_stopping_metric_name']
        recorder = Recorder(self.config['train']['patience'],early_stopping_metric_name,mode='min' if 'loss' in early_stopping_metric_name else 'max')

        train_loader, val_loader, test_loader = dataloader.load_data(text_encoder_path=self.config['model']['text_encoder_path'],
                                                                             image_encoder_path=self.config['model']['image_encoder_path'],
                                                                             use_image=False,
                                                                            **self.config['dataset'])

        ed_tm = time.time()
        self.logger.info('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
        for epoch in range(self.config['train']['num_epochs']):
            avg_loss_classify = train_epoch(self.model, loss_fn, self.config, train_loader, optimizer, epoch, self.num_rationales)
            self.writer.add_scalar('train/loss_classify', avg_loss_classify.item(), epoch)
            self.logger.info('epoch: {}, train_loss_classify: {:.4f}'.format(epoch, avg_loss_classify.item()))
            self.logger.info('----- in val progress... -----')
            val_metrics = self.test(val_loader)
            mark = recorder.add(val_metrics['classifier'])

            self.writer.add_scalar('test/loss_classify', val_metrics['classifier']['loss_classify'], epoch)
            self.logger.info('epoch: {}, test_loss_classify: {:.4f}'.format(epoch, val_metrics["classifier"]['loss_classify']))

            if mark == 'save':
                torch.save(self.model.state_dict(),os.path.join(self.save_path, 'parameter_bert.pkl'))
            if mark == 'esc':
                break
            else:
                continue

        self.logger.info('----- in test progress... -----')
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'parameter_bert.pkl')))
        test_metrics = self.test(test_loader)
        self.logger.info("test metrics: {}.".format(test_metrics['classifier']))
        #self.logger.info("lr: {}, avg test score: {}.\n\n".format(self.config['lr'], future_results['classifier']['metric']))
        self.writer.add_scalars(self.running_tag, test_metrics['classifier'])

        return test_metrics









    def test(self, dataloader):
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
        pred = []
        label = []
        self.model.eval()
        data_iter = tqdm(dataloader)
        avg_loss_classify = Averager()
        metrics_recorder = MetricsRecorder()

        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data_to_device(
                    batch,
                    self.config['train']['use_cuda']
                )
                batch_label = batch_data['label']
                res = self.model(**batch_data)

                loss_classify = loss_fn(res['classify_pred'], batch_label.float())

                label.extend(batch_label.detach().cpu().numpy().tolist())
                pred.extend(res['classify_pred'].detach().cpu().numpy().tolist())
                avg_loss_classify.add(loss_classify.item(),len(batch_label))
                metrics_recorder.record(batch_data, res)

        result_metrics = metrics_recorder.get_metrics()
        result_metrics['classifier']['loss_classify'] = avg_loss_classify.item()
        return result_metrics

































