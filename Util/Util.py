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


class Recorder:

    def __init__(self, patience, metric_name, mode='max'):
        """
        初始化 Recorder 实例。

        参数:
            patience (int): 早停的耐心值，即在性能不再提升的情况下可以容忍的最大轮数。
            metric_name (str): 度量名称，用于从字典中选择特定的度量值。
            mode (str): 'min' 或 'max', 指定是否要最小化或最大化度量值，默认为 'min'。
        """
        self.patience = patience
        self.best_index = 0
        self.current_index = 0
        self.early_stopping_metric_name = metric_name
        self.mode = mode

        if mode == 'min':
            self.best_metric = {
                metric_name: float('inf'),
            }
        elif mode == 'max':
            self.best_metric = {
                metric_name: float('-inf'),
            }
        else:
            raise ValueError("Mode must be 'min' or 'max'.")



    def add(self, metrics):
        """
        添加新的度量值并更新状态。

        参数:
            metrics (dict): 包含多个度量值的字典，键为度量名称，值为对应的度量值。
        """
        if self.early_stopping_metric_name not in metrics:
            raise ValueError(f"Metric '{self.early_stopping_metric_name}' not found in provided metrics dictionary.")

        self.current_metric = metrics
        self.current_index += 1

        decision = self._evaluate()
        if decision == 'save':
            self._print_status()

        return decision

    def _evaluate(self):
        """
        评估当前度量值是否满足早停条件。

        返回:
            str: 决策结果，可能是 'save', 'esc' 或 'continue'。
        """
        current_metric_value = self.current_metric[self.early_stopping_metric_name]
        best_metric_value = self.best_metric[self.early_stopping_metric_name]

        if (self.mode == 'min' and current_metric_value < best_metric_value) or \
           (self.mode == 'max' and current_metric_value > best_metric_value):
            self.best_metric = self.current_metric
            self.best_index = self.current_index
            return 'save'

        if self.current_index - self.best_index >= self.patience:
            return 'esc'

        return 'continue'

    def _print_status(self):
        """打印当前和最大度量值的状态信息。"""
        print(f"Current Metric: {self.current_metric}, Best Metric: {self.best_metric}")

class Averager:

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x,batch_size):
        self.v += x * batch_size
        self.n += batch_size

    def item(self):
        return self.v / self.n if self.n > 0 else 0

def data_to_device(data,use_cuda):
    device = torch.device('cuda:0') if torch.cuda.is_available() and use_cuda else torch.device('cpu')
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


def calculate_multi_classification_metrics(y_true, y_pred_proba, label_names=None):
    """
    计算三元分类的各项指标，基于真实标签和预测概率，并且可以选择性地提供标签名称映射。

    参数:
        y_true (array-like of shape (n_samples,)): 真实标签
        y_pred_proba (array-like of shape (n_samples, n_classes)): 预测概率矩阵
        label_names (dict, optional): 将原始标签映射为更具描述性的名称的字典，默认为 None

    返回:
        dict: 包含所有计算出的性能指标
    """
    # 初始化返回字典
    metrics = {}

    # 将预测概率转换为预测标签（选择最大概率对应的类别）
    y_pred = np.argmax(y_pred_proba, axis=1)

    # Macro-average metrics
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
    # Per-class metrics
    for label, name in label_names.items():
        metrics[f'f1_{name}'] = f1_score(y_true, y_pred, labels=[label], average='macro')
        metrics[f'recall_{name}'] = recall_score(y_true, y_pred, labels=[label], average='macro')
        metrics[f'precision_{name}'] = precision_score(y_true, y_pred, labels=[label], average='macro')

    # One-vs-Rest ROC AUC scores
    try:
        metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='macro')
    except ValueError:
        print("ROC AUC could not be computed, possibly due to all predictions being the same class.")
        metrics['roc_auc_ovr'] = None

    # Confusion Matrix
    #metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    metrics['acc'] = accuracy_score(y_true, y_pred)
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
        'recall': recall_macro,
        f'recall_{label_names[0]}': recall_class0,
        f'recall_{label_names[1]}': recall_class1,
        'precision': precision_macro,
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

    def record(self,batch_data,res):
        batch_label = batch_data['label']
        self.classifier_labels.append(batch_label)
        self.classifier_predictions.append(res['classify_pred'])
        self.llm_judgment_labels.append(torch.cat((batch_data['td_pred'], batch_data['cs_pred']), dim=0))
        llm_judgment_prediction = torch.cat(
            (res['td_judge_pred'], res['cs_judge_pred'])
            ,dim=0)
        self.llm_judgment_predictions.append(llm_judgment_prediction)
        self.rationale_usefulness_labels.append(torch.cat((batch_data['td_acc'], batch_data['cs_acc']), dim=0))
        self.rationale_usefulness_predictions.append(torch.cat((res['td_rationale_useful_pred'], res['cs_rationale_useful_pred']), dim=0))


    def get_metrics(self):
        self.classifier_labels = torch.cat(self.classifier_labels, dim=0).cpu().numpy()
        self.classifier_predictions = (torch.cat(self.classifier_predictions, dim=0) > 0.5).long().cpu().numpy()
        self.llm_judgment_labels = torch.cat(self.llm_judgment_labels, dim=0).cpu().numpy()
        self.llm_judgment_predictions = torch.cat(self.llm_judgment_predictions, dim=0).argmax(dim=1).cpu().numpy()
        self.rationale_usefulness_labels = torch.cat(self.rationale_usefulness_labels, dim=0).cpu().numpy()
        self.rationale_usefulness_predictions = (torch.cat(self.rationale_usefulness_predictions, dim=0) > 0.5).long().cpu().numpy()

        classifier_metrics = cal_binary_metrics(self.classifier_labels,
                                                                     self.classifier_predictions,
                                                                     label_names=classified_label_int2str_dict
                                                                     )

        llm_judgment_metrics = cal_binary_metrics(self.llm_judgment_labels,
                                                                      self.llm_judgment_predictions,
                                                                      label_names=classified_label_int2str_dict)
        rationale_usefulness_metrics = cal_binary_metrics(self.rationale_usefulness_labels,
                                                                               self.rationale_usefulness_predictions,
                                                                               label_names={0:'unuseful',1:'useful'}) # TODO 动态设置标签名称

        return {
            'classifier': classifier_metrics,
            'llm_judgment': llm_judgment_metrics,
            'rationale_usefulness': rationale_usefulness_metrics
        }
