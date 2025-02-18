import logging
import os
import time

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from tqdm import tqdm
from transformers import AutoModel
import torch.nn.functional as F

from Util import dataloader
from Util.Util import try_all_gpus, Decision, Averager, MetricsRecorder, data_to_device, Recorder
from model.layers import contrastive_loss


class ContrastiveModel(nn.Module):

    def __init__(self, text_encoder_path,image_encoder_path,sim_emb,temperature=0.02,**kwargs):
        super(ContrastiveModel, self).__init__()
        self.text_encoder = AutoModel.from_pretrained(text_encoder_path)
        self.text_projection = nn.Sequential(
            nn.Linear(768, sim_emb),
            nn.ReLU(),
        )
        self.image_encoder = AutoModel.from_pretrained(image_encoder_path)
        self.image_projection = nn.Sequential(
            nn.Linear(768, sim_emb),
            nn.ReLU(),
        )
        self.temperature = temperature


    def forward(self,**kwargs):
        text ,text_mask = kwargs['img_rationale'],kwargs['img_rationale_mask']
        image = kwargs['image']
        text_pooling_features = self.text_projection(self.text_encoder(text,attention_mask=text_mask).pooler_output)
        image_pooling_features = self.image_projection(self.image_encoder(image).pooler_output)

        text_pooling_features = F.normalize(text_pooling_features, p=2, dim=1)
        image_pooling_features = F.normalize(image_pooling_features, p=2, dim=1)

        cos_sim_matrix = torch.matmul(text_pooling_features, image_pooling_features.T) * np.exp(self.temperature)

        return cos_sim_matrix

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

        cos_sim_matrix = model(**batch_data)
        loss = loss_fn(cos_sim_matrix)


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss_classify.add(loss.item(),cos_sim_matrix.size(0))

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
        self.model = ContrastiveModel(**config['model'])
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
        loss_fn = contrastive_loss
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.config['train']['lr'],
                                     weight_decay=self.config['train']['weight_decay'])

        # 初始化早停类
        recorder = Recorder(
            patience=self.config['train']['patience'],
            metric_name='val_loss',
            mode='min'
        )

        train_loader, val_loader, test_loader = dataloader.load_data(text_encoder_path=self.config['model']['text_encoder_path'],
                                                                             image_encoder_path=self.config['model']['image_encoder_path'],
                                                                             use_image=True,
                                                                            **self.config['dataset'])

        ed_tm = time.time()
        self.logger.info('time cost in model and data loading: {}s'.format(ed_tm - st_tm))
        for epoch in range(self.config['train']['num_epochs']):
            # model, loss_fn, train_loader, optimizer, epoch,device
            avg_loss_classify = train_epoch(self.model, loss_fn, train_loader, optimizer, epoch,device)
            self.writer.add_scalar('train/loss_classify', avg_loss_classify.item(), epoch)
            self.logger.info('epoch: {}, train_loss_classify: {:.4f}'.format(epoch, avg_loss_classify.item()))
            self.logger.info('----- in val progress... -----')
            val_metrics = self.test(val_loader, device)
            mark = recorder.add(val_metrics)
            self.logger.info('epoch: {}, test_loss_classify: {:.4f}'.format(epoch, val_metrics['val_loss']))

            if mark == Decision.SAVE:
                torch.save(self.model.text_encoder.state_dict(), os.path.join(self.save_path, 'text_model.pth'))
                torch.save(self.model.image_encoder.state_dict(), os.path.join(self.save_path, 'image_model.pth'))
            if mark == Decision.ESCAPE:
                break

        self.logger.info('----- in test progress... -----')
        self.model.text_encoder.load_state_dict(torch.load(os.path.join(self.save_path, 'text_model.pth')))
        self.model.image_encoder.load_state_dict(torch.load(os.path.join(self.save_path, 'image_model.pth')))
        test_metrics = self.test(test_loader, device)
        self.logger.info("test metrics: {}.".format(test_metrics['val_loss']))
        self.writer.add_scalars(self.running_tag, test_metrics)

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
        loss_fn = contrastive_loss
        self.model.eval()
        data_iter = tqdm(dataloader)
        avg_loss = Averager()
        result_metrics = dict()
        for step_n, batch in enumerate(data_iter):
            with torch.no_grad():
                batch_data = data_to_device(
                    batch,
                    device
                )
                cos_sim_matrix = self.model(**batch_data)
                loss = loss_fn(cos_sim_matrix)
                avg_loss.add(loss.item(),cos_sim_matrix.size(0))

        result_metrics['val_loss'] = avg_loss.item()
        return result_metrics
