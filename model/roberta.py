import logging
import os
import time

import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from transformers import RobertaForSequenceClassification

from Util import dataloader
from Util.Util import try_all_gpus, Recorder, Averager, data_to_device, Decision, MetricsRecorder


class Roberta(nn.Module):

    def __init__(self, config):
        super(Roberta, self).__init__()
        self.encoder = RobertaForSequenceClassification.from_pretrained(config['text_encoder_path'])


    def forward(self, **kwargs):
        classify_pred =  self.encoder(kwargs['content'],attention_mask=kwargs['content_mask']).logits
        return {
            'classify_pred':classify_pred
        }




def train_epoch(model, loss_fn, train_loader, optimizer, epoch,device):
    print('---------- epoch {} ----------'.format(epoch))
    model.train()
    train_data_iter = tqdm(train_loader)
    avg_loss_classify = Averager()
    for step_n, batch in enumerate(train_data_iter):
        batch_data = data_to_device(
            batch,
            device
        )
        label = torch.nn.functional.one_hot(batch_data['label'],num_classes=2).to(dtype=torch.float32)


        batch_res = model(**batch_data)
        loss = loss_fn(batch_res['classify_pred'], label)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss_classify.add(loss.item(),label.size(0))

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
        self.model = Roberta(config['model'])
        self.running_tag = f'{config["model"]["name"]}/{config["model"]["version"]}/{config["dataset"]["name"]}'
        self.writer = SummaryWriter(f'logs/tensorboard/{self.running_tag}')
        self.logger = logging.getLogger(__name__)
        self.logger.info(f'config : {self.config}')
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
        loss_fn = nn.BCEWithLogitsLoss()
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
                                                                            **self.config['dataset'])

        ed_tm = time.time()
        self.logger.info('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
        for epoch in range(self.config['train']['num_epochs']):
            avg_loss_classify = train_epoch(self.model, loss_fn, train_loader, optimizer, epoch,device)
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
        loss_fn = torch.nn.BCEWithLogitsLoss()
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
                batch_label = torch.nn.functional.one_hot(batch_data['label'],num_classes=2).to(dtype=torch.float32)
                res = self.model(**batch_data)
                loss_classify = loss_fn(res['classify_pred'], batch_label)
                avg_loss_classify.add(loss_classify.item(),batch_label.shape[0])
                metrics_recorder.record(batch_data, res,None)

        result_metrics = metrics_recorder.get_metrics()
        result_metrics['classifier']['loss_classify'] = avg_loss_classify.item()
        return result_metrics
