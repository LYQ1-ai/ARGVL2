import argparse
import json
import multiprocessing
import os
import random
from multiprocessing import current_process

import numpy as np
import torch
import yaml

import optuna
from joblib import Parallel

from Util import Util
from model.arg import Trainer as ARGTrainer
from model.argVL import Trainer as ARGVLTrainer
from model.argVL2 import Trainer as ARGVL2Trainer
from model.ContrastiveModel import  Trainer as ContrastiveTrainer
from model.roberta import Trainer as RoBERTATrainer



os.environ["TOKENIZERS_PARALLELISM"] = "true"

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
    'ARG': ARGTrainer,
    'ARGVL':ARGVLTrainer,
    'ARGVL2':ARGVL2Trainer,
    'ContrastiveModel':ContrastiveTrainer,
    'RoBERTA':RoBERTATrainer
}

from multiprocessing import Manager
from contextlib import contextmanager

# 初始化可用 GPU 集合和进程锁
_manager = Manager()
_available_gpus = _manager.list()  # 使用 List 替代 Set（Manager 的 Set 支持较差）
_gpu_lock = multiprocessing.Lock()

def init_gpu_pool(gpu_ids):
    """初始化 GPU 资源池"""
    global _available_gpus
    _available_gpus.extend(gpu_ids)

@contextmanager
def acquire_gpu():
    """获取一个可用 GPU 的上下文管理器"""
    gpu_id = None
    try:
        # 加锁访问共享资源
        with _gpu_lock:
            if len(_available_gpus) == 0:
                raise RuntimeError("No available GPUs")
            gpu_id = _available_gpus.pop(0)
        yield gpu_id
    finally:
        # 归还 GPU 到资源池
        if gpu_id is not None:
            with _gpu_lock:
                _available_gpus.append(gpu_id)

def get_available_gpu():
    """直接获取 GPU ID（需手动释放）"""
    with _gpu_lock:
        if len(_available_gpus) == 0:
            return None
        return _available_gpus.pop(0)

def release_gpu(gpu_id):
    """释放 GPU ID"""
    with _gpu_lock:
        if gpu_id not in _available_gpus:
            _available_gpus.append(gpu_id)


def opt_config(trial):
    current_config = config.copy()
    trial_id = f"trial_{trial.number}"
    trial_save_path = os.path.join(current_config['train']['save_param_dir'], trial_id)
    os.makedirs(trial_save_path, exist_ok=True)
    current_config['train']['save_param_dir'] = trial_save_path
    current_config['dataset']['rationale_max_len'] = trial.suggest_int('max_len',200,300,step=10) # 10
    current_config['model']['num_heads'] = trial.suggest_categorical('num_heads',[1,2,4]) # 3
    current_config['model']['dropout'] = trial.suggest_float('dropout',0.1,0.5,step=0.1) # 5
    current_config['MV_layers'] = trial.suggest_int('MV_layers',1,4,step=1) # 4
    current_config['train']['lr'] = trial.suggest_categorical('lr', [5e-5,5e-4,5e-3,5e-2]) # 4
    current_config['train']['rationale_usefulness_evaluator_weight'] = trial.suggest_float('rationale_usefulness_evaluator_weight',1.0,2.5,step=0.1) # 15
    current_config['train']['llm_judgment_predictor_weight'] = trial.suggest_float('llm_judgment_predictor_weight',0.1,2.5,step=0.1) # 25
    current_config['train']['FocalLoss']['alpha'] = trial.suggest_float('alpha',0.0,0.7,step=0.1) # 7
    current_config['train']['FocalLoss']['gamma'] = trial.suggest_float('gamma',0.0,2.0,step=0.5) # 4
    trainer = model_name2trainer_dict[config['model']['name']](current_config,trial=trial)
    result = trainer.train()
    return result['classifier']['f1_macro']






# 按装订区域中的绿色按钮以运行脚本。
if __name__ == '__main__':
    init_gpu_pool([0,1,2,3])
    running_tag = f'{config["model"]["name"]}_{config["model"]["version"]}_{config["dataset"]["name"]}'
    if not os.path.exists(config['logging']['log_dir']):
        os.makedirs(config['logging']['log_dir'])
    log_file_path = os.path.join(config['logging']['log_dir'],running_tag+'.log')
    Util.setup_logger(log_file_path,level=config['logging']['level'])

    if not os.path.exists(config['logging']['json_result_dir']):
        os.makedirs(config['logging']['json_result_dir'])
    json_result_path = os.path.join(config['logging']['json_result_dir'],running_tag+'.json')

    stu = optuna.create_study(
        study_name=running_tag,
        direction='maximize',
        storage='sqlite:///test.db?check_same_thread=False',
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,  # 前5次试验不剪枝（保证初始探索）
            n_warmup_steps=10,  # 每个试验前10步不触发剪枝
        )
    )

    stu.optimize(opt_config,n_trials=100)

    print(stu.best_params)
    print(stu.best_trial)
    print(stu.best_trial.value)
    optuna.visualization.plot_param_importances(stu).show()
    optuna.visualization.plot_optimization_history(stu).show()
    optuna.visualization.plot_slice(stu).show()




