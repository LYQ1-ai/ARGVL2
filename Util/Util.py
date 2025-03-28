import logging

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix, accuracy_score

from Util.dataloader import label_int2str_dict, classified_label_int2str_dict



def setup_logger(log_file, level=logging.INFO):
    logging.basicConfig(
        level=level,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',  # 日志格式
        handlers=[
            logging.FileHandler(log_file),  # 输出到文件
            logging.StreamHandler()  # 同时输出到控制台
        ]
    )


def try_all_gpus():
    return [torch.device(f'cuda:{i}') for i in range(torch.cuda.device_count())]


from enum import Enum

class Decision(Enum):
    SAVE = 'save'
    ESCAPE = 'esc'
    CONTINUE = 'continue'

class Recorder:
    def __init__(self, patience, metric_name, mode='max'):
        """
        初始化 Recorder 实例。

        参数:
            patience (int): 早停的耐心值，即在性能不再提升的情况下可以容忍的最大轮数。
            metric_name (str): 度量名称，用于从字典中选择特定的度量值。
            mode (str): 'min' 或 'max', 指定是否要最小化或最大化度量值，默认为 'max'。
        """
        self.patience = patience
        self.best_index = 0
        self.current_index = 0
        self.early_stopping_metric_name = metric_name
        self.mode = mode

        if mode == 'min':
            self.best_metric = {
                self.early_stopping_metric_name:float('inf')
            }
        elif mode == 'max':
            self.best_metric = {
                self.early_stopping_metric_name:float('-inf')
            }
        else:
            raise ValueError("Mode must be 'min' or 'max'.")

    def add(self, metrics):
        """
        添加新的度量值并更新状态。

        参数:
            metrics (dict): 包含多个度量值的字典，键为度量名称，值为对应的度量值。

        返回:
            Decision: 决策结果，可能是 SAVE、ESCAPE 或 CONTINUE。
        """
        if self.early_stopping_metric_name not in metrics:
            raise ValueError(f"Metric '{self.early_stopping_metric_name}' not found in provided metrics dictionary.")

        current_metric = metrics
        self.current_index += 1

        decision = self._evaluate(current_metric)
        self._print_status(current_metric)
        return decision

    def _evaluate(self, current_metric):
        """
        评估当前度量值是否满足早停条件。

        返回:
            Decision: 决策结果。
        """
        if (self.mode == 'min' and current_metric[self.early_stopping_metric_name] < self.best_metric[self.early_stopping_metric_name]) or \
           (self.mode == 'max' and current_metric[self.early_stopping_metric_name] > self.best_metric[self.early_stopping_metric_name]):
            self.best_metric = current_metric
            self.best_index = self.current_index
            return Decision.SAVE

        if self.current_index - self.best_index >= self.patience:
            return Decision.ESCAPE

        return Decision.CONTINUE

    def _print_status(self, current_metric):
        """打印当前和最大度量值的状态信息。"""
        print(f"Current Metric : {current_metric}, \n"
              f"Best Metric : {self.best_metric}")

class Averager:

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x,batch_size):
        self.v += x * batch_size
        self.n += batch_size

    def item(self):
        return self.v / self.n if self.n > 0 else 0

def data_to_device(data,device):
    return {
            k:data[k].to(device)
            for k in data.keys() if data[k] is not None and isinstance(data[k], torch.Tensor)
        }


def calculate_binary_classification_metrics(y_true, y_pred_proba, label_names=None):
    """
    计算二元分类的各项指标，基于真实标签和预测概率，并且可以选择性地提供标签名称映射。

    参数:
        y_true (array-like of shape (n_samples,)): 真实标签
        y_pred_proba (array-like of shape (n_samples,)): 预测概率
        label_names (dict, optional): 将原始标签映射为更具描述性的名称的字典，默认为 None

    返回:
        dict: 包含所有计算出的性能指标
    """
    # 初始化返回字典
    metrics = {}

    # 设置默认标签名称，如果未提供自定义标签名称
    if label_names is None:
        label_names = {0: 'class_0', 1: 'class_1'}

    # 将预测概率转换为预测标签（通常用0.5作为阈值）
    y_pred = [1 if prob >= 0.5 else 0 for prob in y_pred_proba]

    # AUC - Area Under the ROC Curve
    metrics['auc'] = roc_auc_score(y_true, y_pred_proba)

    # F1 Score
    metrics['f1_macro'] = f1_score(y_true, y_pred,average='macro')

    # Recall/Sensitivity (True Positive Rate)
    metrics['recall'] = recall_score(y_true, y_pred)

    # Precision
    metrics['precision'] = precision_score(y_true, y_pred)

    metrics['acc'] = accuracy_score(y_true, y_pred)

    # Per-class metrics
    for label, name in label_names.items():
        # F1 Score per class
        metrics[f'f1_{name}'] = f1_score(y_true, y_pred, pos_label=label)

        # Recall per class
        metrics[f'recall_{name}'] = recall_score(y_true, y_pred, pos_label=label)

        # Precision per class
        metrics[f'precision_{name}'] = precision_score(y_true, y_pred, pos_label=label)

    return metrics


def cal_binary_metrics(y_pred, y_true,label_names):
    # 计算真实类和伪造类的各项指标
    recall_class0 = recall_score(y_true, y_pred, average=None, labels=[0])[0]
    recall_class1 = recall_score(y_true, y_pred, average=None, labels=[1])[0]
    precision_class0 = precision_score(y_true, y_pred, average=None, labels=[0])[0]
    precision_class1 = precision_score(y_true, y_pred, average=None, labels=[1])[0]
    f1_class0 = f1_score(y_true, y_pred, average=None, labels=[0])[0]
    f1_class1 = f1_score(y_true, y_pred, average=None, labels=[1])[0]

    # 宏平均指标由真实类和伪造类的指标算术平均得出
    recall_macro = (recall_class0 + recall_class1) / 2
    precision_macro = (precision_class0 + precision_class1) / 2
    f1_macro = (f1_class0 + f1_class1) / 2

    return {
        'acc': accuracy_score(y_true, y_pred),
        'recall_macro': recall_macro,
        f'recall_{label_names[0]}': recall_class0,
        f'recall_{label_names[1]}': recall_class1,
        'precision_macro': precision_macro,
        f'precision_{label_names[0]}': precision_class0,
        f'precision_{label_names[1]}': precision_class1,
        'f1_macro': f1_macro,
        f'f1_{label_names[0]}': f1_class0,
        f'f1_{label_names[1]}': f1_class1,
    }


class MetricsRecorder:


    def __init__(self):
        self.classifier_labels = []
        self.classifier_predictions = []

        self.llm_judgment_labels = []
        self.llm_judgment_predictions = []

        self.rationale_usefulness_labels = []
        self.rationale_usefulness_predictions = []

    def record(self,batch_data,res,rationale_names):
        batch_label = batch_data['label']
        self.classifier_labels.append(batch_label)
        self.classifier_predictions.append(res['classify_pred'])
        if rationale_names is not None:
            for r_name in rationale_names:
                if res[f'{r_name}_judge_pred'] is not None:
                    self.llm_judgment_predictions.append(res[f'{r_name}_judge_pred'])
                    self.llm_judgment_labels.append(batch_data[f'{r_name}_pred'])
                if res[f'{r_name}_rationale_useful_pred'] is not None:
                    self.rationale_usefulness_predictions.append(
                        res[f'{r_name}_rationale_useful_pred']
                    )
                    self.rationale_usefulness_labels.append(batch_data[f'{r_name}_acc'])






    def get_metrics(self):

        self.classifier_labels = torch.cat(self.classifier_labels, dim=0).cpu().numpy()
        self.classifier_predictions = torch.cat(self.classifier_predictions, dim=0)
        if self.classifier_predictions.dim() == 1:
            self.classifier_predictions = (self.classifier_predictions > 0.5).long().cpu().numpy()
        else:
            self.classifier_predictions = self.classifier_predictions.argmax(dim=1).long().cpu().numpy()
        classifier_metrics = cal_binary_metrics(self.classifier_predictions,
                                                self.classifier_labels,
                                                label_names=classified_label_int2str_dict
                                                )
        res_metrics = {'classifier': classifier_metrics}
        if len(self.llm_judgment_labels) > 0:
            self.llm_judgment_labels = torch.cat(self.llm_judgment_labels, dim=0).cpu().numpy()
            self.llm_judgment_predictions = torch.cat(self.llm_judgment_predictions, dim=0).argmax(dim=1).cpu().numpy()
            llm_judgment_metrics = cal_binary_metrics(self.llm_judgment_predictions,
                                                      self.llm_judgment_labels,
                                                      label_names=classified_label_int2str_dict)
            res_metrics['llm_judgment'] = llm_judgment_metrics
        if len(self.rationale_usefulness_labels) > 0:
            self.rationale_usefulness_labels = torch.cat(self.rationale_usefulness_labels, dim=0).cpu().numpy()
            self.rationale_usefulness_predictions = (torch.cat(self.rationale_usefulness_predictions, dim=0) > 0.5).long().cpu().numpy()
            rationale_usefulness_metrics = cal_binary_metrics(self.rationale_usefulness_predictions,
                                                              self.rationale_usefulness_labels,
                                                              label_names={0: 'unuseful', 1: 'useful'})  # TODO 动态设置标签名称
            res_metrics['rationale_usefulness'] = rationale_usefulness_metrics


        return res_metrics
