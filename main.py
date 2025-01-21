import argparse
import json
import os
import random

import numpy as np
import torch
import yaml

from Util import Util
from model.arg import Trainer as ARGTrainer

parser = argparse.ArgumentParser(description='ARG model training and evaluation')
parser.add_argument('--config_file_path', type=str, default='config/arg_qwen_gossipcop_win_config.yaml')


args = parser.parse_args()

config = yaml.load(open(args.config_file_path, 'r'), Loader=yaml.FullLoader)

seed = config['seed']
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


model_name2trainer_dict = {
    'ARG': ARGTrainer
}




# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':

    running_tag = f'{config["model"]["name"]}_{config["model"]["version"]}_{config["dataset"]["name"]}'
    if not os.path.exists(config['logging']['log_dir']):
        os.makedirs(config['logging']['log_dir'])
    log_file_path = os.path.join(config['logging']['log_dir'],running_tag+'.log')
    Util.setup_logger(log_file_path,level=config['logging']['level'])

    if not os.path.exists(config['logging']['json_result_dir']):
        os.makedirs(config['logging']['json_result_dir'])
    json_result_path = os.path.join(config['logging']['json_result_dir'],running_tag+'.json')

    trainer = model_name2trainer_dict[config['model']['name']](config)
    result = trainer.train()
    with open(json_result_path,'w') as f:
        json.dump(result,f,indent=4)




