

import logging
import os
import time

import numpy as np
import torch.nn.functional as F
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from transformers import AutoModel, BertModel, Swinv2Model, RobertaModel

import Util.Util
from Util import dataloader
from Util.Util import try_all_gpus, Recorder, Averager, data_to_device, MetricsRecorder, Decision
from model import layers
from model.layers import AttentionPooling, Classifier, AvgPooling, MultiHeadCrossAttention, DualCrossAttention, \
    FocalLoss, MultiViewAggregation, ImageCaptionGate


def freeze_bert_params(model):
    for name, param in model.named_parameters():
        if name.startswith("encoder.layer.11"):
            param.requires_grad = True
        else:
            param.requires_grad = False


def freeze_swinv2_params(model):
    for name, param in model.named_parameters():
        if name.startswith("encoder.layers.3"):
            param.requires_grad = True
        else:
            param.requires_grad = False

def freeze_pretrained_params(model):
    if isinstance(model,BertModel) or isinstance(model,RobertaModel):
        freeze_bert_params(model)
    elif isinstance(model,Swinv2Model):
        freeze_swinv2_params(model)





class ARGVL2Model(nn.Module):

    @staticmethod
    def get_text_encoder(config,use_contrastive_model=False):
        text_encoder = AutoModel.from_pretrained(config['text_encoder_path'])
        if use_contrastive_model:
            text_encoder_model_path = os.path.join(config['ContrastiveModel_path'],'text_model.pth')
            if os.path.exists(text_encoder_model_path):
                text_encoder.load_state_dict(torch.load(text_encoder_model_path,weights_only=True))

        freeze_pretrained_params(text_encoder)
        return text_encoder

    @staticmethod
    def get_image_encoder(config,use_contrastive_model=False):
        image_encoder = Swinv2Model.from_pretrained(config['image_encoder_path'])
        image_encoder_model_path = os.path.join(config['ContrastiveModel_path'], 'image_model.pth')
        if use_contrastive_model and os.path.exists(image_encoder_model_path):
            image_encoder.load_state_dict(torch.load(image_encoder_model_path,weights_only=True))
        freeze_pretrained_params(image_encoder)
        return image_encoder

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.content_encoder = ARGVL2Model.get_text_encoder(config)
        self.rationale_encoder = ARGVL2Model.get_text_encoder(config)
        self.img_rationale_encoder = ARGVL2Model.get_text_encoder(config,True)
        self.image_encoder = ARGVL2Model.get_image_encoder(config,True)
        self.caption_content_fusion = DualCrossAttention(config['emb_dim'], config['num_heads'], dropout=config['dropout'], layers=1)
        self.rationale_set = set(config['rationale_name'])
        self.td_rationale_fusion = layers.BaseRationaleFusion(config)
        self.itc_rationale_fusion = layers.BaseRationaleFusion(config)
        self.img_rationale_fusion = layers.BaseRationaleFusion(config)

        self.content_attention_pooling = AttentionPooling(config['emb_dim'])
        # self.image_content_attention_pooling = AttentionPooling(config['emb_dim'])

        self.featureAggregator = MultiViewAggregation(config['emb_dim'], len(self.rationale_set) + 1,layers=2,mask_enable=True)

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
            "itc_rationale":"无法确定。因为没有给出具体的消息内容，无法判断其真实性。",
            "itc_rationale_mask":
            "itc_pred":"0",
            "itc_acc":1,
            "caption":Optional[Tensor],
            "caption_mask":Optional[Tensor]
            "split":"train"
            "image": Tensor (batch_size,49,768)
        }
        :return:
        """
        content_mask = kwargs['content_mask']
        content_features = self.content_encoder(kwargs['content'], attention_mask=content_mask).last_hidden_state
        # image_content_features = self.image_encoder(kwargs['image']).last_hidden_state

        td_rationale_mask = kwargs['td_rationale_mask']
        itc_rationale_mask = kwargs['itc_rationale_mask']
        img_rationale_mask = kwargs['img_rationale_mask']

        all_features = []
        res = {}

        if 'td' in self.rationale_set:
            td_rationale_features = self.rationale_encoder(kwargs['td_rationale'],
                                                           attention_mask=td_rationale_mask).last_hidden_state
            td_rationale_pooling_feature,td_rationale_useful_pred,td_judge_pred = self.td_rationale_fusion(
                content_feature=content_features,
                content_mask=content_mask,
                rationale_feature=td_rationale_features,
                rationale_mask=td_rationale_mask,
            )
            all_features.append(td_rationale_pooling_feature.unsqueeze(1))
            res['td_judge_pred'] = td_judge_pred
            res['td_rationale_useful_pred'] = td_rationale_useful_pred


        if 'itc' in self.rationale_set:
            itc_rationale_features = self.rationale_encoder(kwargs['itc_rationale'],
                                                            attention_mask=itc_rationale_mask).last_hidden_state
            caption_mask = kwargs['caption_mask']
            caption_features = self.content_encoder(kwargs['caption'], attention_mask=caption_mask).last_hidden_state
            multi_feature, multi_mask = self.caption_content_fusion(caption_features, caption_mask, content_features,
                                                                    content_mask)
            itc_rationale_pooling_feature,itc_rationale_useful_pred,itc_judge_pred = self.itc_rationale_fusion(
                content_feature=multi_feature,
                content_mask=multi_mask,
                rationale_feature=itc_rationale_features,
                rationale_mask=itc_rationale_mask,
            )
            all_features.append(itc_rationale_pooling_feature.unsqueeze(1))
            res['itc_judge_pred'] = itc_judge_pred
            res['itc_rationale_useful_pred'] = itc_rationale_useful_pred


        if 'img' in self.rationale_set:
            image_content_features = self.image_encoder(kwargs['image']).last_hidden_state
            img_rationale_features = self.img_rationale_encoder(kwargs['img_rationale'],
                                                                attention_mask=img_rationale_mask).last_hidden_state
            img_rationale_pooling_feature,img_rationale_useful_pred,img_judge_pred = self.img_rationale_fusion(
                content_feature=image_content_features,
                content_mask=None,
                rationale_feature=img_rationale_features,
                rationale_mask=img_rationale_mask,
            )
            all_features.append(img_rationale_pooling_feature.unsqueeze(1))
            res['img_judge_pred'] = img_judge_pred
            res['img_rationale_useful_pred'] = img_rationale_useful_pred


        all_features.insert(
            0,self.content_attention_pooling(content_features,content_mask).unsqueeze(1)
        )


        final_feature = self.featureAggregator(torch.cat(all_features,dim=1))

        label_pred = self.classifier(final_feature).squeeze(1)
        res['classify_pred'] = label_pred

        return res


def train_epoch(model, loss_fn, config, train_loader, optimizer, epoch, rationale_names,device):
    print('---------- epoch {} ----------'.format(epoch))
    model.train()
    train_data_iter = tqdm(train_loader)
    avg_loss_classify = Averager()
    logging.info(f"UsefulPredict use FocalLoss : {config['train']['FocalLoss']['enable']}")
    for step_n, batch in enumerate(train_data_iter):
        batch_data = data_to_device(
            batch,
            device
        )
        label = batch_data['label']

        res = model(**batch_data)
        loss_classify = loss_fn(res['classify_pred'], label.float())

        loss_judge_fn = torch.nn.CrossEntropyLoss()
        loss_useful_fn = FocalLoss(alpha=config['train']['FocalLoss']['alpha'],
                                   gamma=config['train']['FocalLoss']['gamma']) if config['train']['FocalLoss'][
            'enable'] else nn.BCELoss()

        loss_useful_pred = 0.0
        loss_judgement_pred = 0.0
        for r_name in rationale_names:
            loss_useful_pred += loss_useful_fn(res[f'{r_name}_rationale_useful_pred'], batch_data[f'{r_name}_acc'])
            loss_judgement_pred += loss_judge_fn(res[f'{r_name}_judge_pred'], batch_data[f'{r_name}_pred'])


        loss = loss_classify
        if len(rationale_names) > 0:
            loss += config['train']['rationale_usefulness_evaluator_weight'] * loss_useful_pred / len(rationale_names)
            loss += config['train']['llm_judgment_predictor_weight'] * loss_judgement_pred / len(rationale_names)

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
        self.model = ARGVL2Model(config['model'])
        self.rationale_names = config['model']['rationale_name']
        self.running_tag = f'{config["model"]["name"]}/{config["model"]["version"]}/{config["dataset"]["name"]}'
        self.writer = SummaryWriter(f'logs/tensorboard/{self.running_tag}')
        self.logger = logging.getLogger(__name__)
        self.save_path = f'{config["train"]["save_param_dir"]}/{self.running_tag}'
        self.save_path = os.path.join(config["train"]["save_param_dir"],self.running_tag)
        if os.path.exists(self.save_path):
            self.save_param_dir = self.save_path
        else:
            os.makedirs(self.save_path)



    def train(self):
        st_tm = time.time()
        self.logger.info('start training......')
        self.logger.info('==================== start training ====================')
        device = self.model_device_init()
        #loss_fn = WeightedBCELoss(weight=self.config['train']['LossWeight']['classify'],reduce_class=0)
        loss_fn = nn.BCELoss()
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
                                                                             use_image=True,
                                                                            **self.config['dataset'])

        ed_tm = time.time()
        self.logger.info('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
        for epoch in range(self.config['train']['num_epochs']):
            avg_loss_classify = train_epoch(self.model, loss_fn, self.config, train_loader, optimizer, epoch, self.rationale_names,device)
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
        self.model.load_state_dict(torch.load(os.path.join(self.save_path, 'model.pth'),weights_only=True))
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

































