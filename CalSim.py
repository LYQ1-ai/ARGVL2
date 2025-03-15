from typing import List

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AutoImageProcessor, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
import Util.dataloader
from Util.Util import data_to_device


class SimModel(nn.Module):

    def __init__(self,text_model_path,image_model_path):
        super(SimModel,self).__init__()
        self.text_model = AutoModel.from_pretrained(text_model_path)
        self.image_model = AutoModel.from_pretrained(image_model_path)

    def forward(self,text_input,text_mask,image_input):
        text_features = self.text_model(text_input,attention_mask=text_mask).pooler_output
        image_features = self.image_model(image_input).pooler_output
        return nn.functional.cosine_similarity(text_features,image_features) + 1.0


from typing import Optional


def visualize_similarity_distribution(
        similarities_list: List[List[float]],
        name: List[str],
        save_path: str,
        normalize: bool = False,
        transform: Optional[str] = None,
        bw_adjust: float = 0.8,
        xlim: Optional[tuple] = None
) -> None:
    """
    增强版分布可视化（支持任意范围数据）

    新增参数：
    normalize: 是否自动归一化到[0,1]区间（默认False）
    xlim: 手动指定x轴范围（如(0.5, 1.0)）
    """
    plt.figure(figsize=(10, 6))

    # 数据预处理流程
    processed_data = []
    for scores in similarities_list:
        arr = np.array(scores)

        # 异常值处理
        if normalize:
            arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-8)  # 防止除零

        # 数据变换
        if transform == 'sqrt':
            arr = np.sqrt(arr - np.min(arr))  # 保证非负
        elif transform == 'cbrt':
            arr = np.cbrt(arr - np.min(arr))

        processed_data.append(arr)

    # 自动颜色映射
    palette = sns.color_palette("husl", len(processed_data))

    # 主绘图逻辑
    for data, color, label in zip(processed_data, palette, name):
        sns.kdeplot(data,
                    color=color,
                    linewidth=2,
                    bw_adjust=bw_adjust,
                    label=f"{label}\nmin={np.min(data):.2f}\nmax={np.max(data):.2f}")

    # 坐标轴设置
    if xlim is None:
        all_data = np.concatenate(processed_data)
        x_min = np.min(all_data) - 0.05 * (np.ptp(all_data))
        x_max = np.max(all_data) + 0.05 * (np.ptp(all_data))
        plt.xlim(x_min, x_max)
    else:
        plt.xlim(xlim)

    # 标签动态生成
    xlabel = "Similarity Score"
    if normalize:
        xlabel += " (Normalized)"
    if transform:
        xlabel += f"\nTransform: {transform.upper()}"

    # 图例与样式
    plt.title(f"Similarity Distributions ({'Raw' if not normalize else 'Normalized'} Data)",
              fontsize=13, pad=15)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(title="Data Statistics",
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               borderpad=1)
    plt.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_cumulative_distribution(
        similarities_list: List[List[float]],
        names: List[str],
        save_path: str,
        transform: Optional[str] = None,
        diff_enhance: bool = False,
        **kwargs
) -> None:
    """
    增强版CDF可视化（支持数据变换与差异强化）

    新增参数：
    transform: 数据变换类型，可选['logit', 'pow2', 'rank']
    diff_enhance: 是否添加一阶导数曲线（显示变化率差异）
    """
    plt.figure(figsize=(12, 6 if diff_enhance else 6))

    # 数据预处理管道
    processed_data = []
    for data in similarities_list:
        arr = np.array(data)

        # 数据变换
        if transform == 'logit':
            arr = np.log(arr / (1 - arr + 1e-8))  # Logit变换，+1e-8防止除零
        elif transform == 'pow2':
            arr = arr ** 2  # 平方增强高值区差异
        elif transform == 'rank':
            arr = rankdata(arr) / len(arr)  # 分位数变换
        processed_data.append(arr)

    # 主坐标轴绘制
    ax1 = plt.gca()
    lines = []
    for idx, (data, name) in enumerate(zip(processed_data, names)):
        # 排序数据并计算CDF
        sorted_data = np.sort(data)
        cum_prob = np.linspace(0, 1, len(sorted_data))

        # 绘制主CDF曲线
        line, = ax1.plot(sorted_data, cum_prob,
                         label=name,
                         lw=2.5,
                         color=f'C{idx}',
                         alpha=0.8)
        lines.append(line)

        # 差异强化：添加一阶导数
        if diff_enhance:
            deriv = np.gradient(cum_prob, sorted_data)
            ax1.plot(sorted_data[1:-1], deriv[1:-1] / np.max(deriv),  # 归一化导数
                     color=f'C{idx}',
                     linestyle=':',
                     alpha=0.6)

    # 当启用差异增强时添加右侧坐标轴
    if diff_enhance:
        ax2 = ax1.twinx()
        ax2.set_ylabel('Normalized Derivative', color='gray', fontsize=10)
        ax2.tick_params(axis='y', labelcolor='gray')

    # 坐标轴装饰
    xlabel = "Similarity Score"
    if transform == 'logit':
        xlabel += r" (logit scale)"
    elif transform == 'pow2':
        xlabel += r" (squared)"
    elif transform == 'rank':
        xlabel += r" (rank transformed)"

    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel("Cumulative Probability", fontsize=12)
    ax1.grid(True, alpha=0.3)

    # 组合图例
    ax1.legend(handles=lines,
               loc='upper left',
               bbox_to_anchor=(1.01, 1),
               borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def load_sim_list():
    gossipcop_sim_df = pd.read_csv('gossipcop_sim.csv')['sim'].tolist()
    twitter_sim_df = pd.read_csv('twitter_sim.csv')['sim'].tolist()
    weibo_sim_df = pd.read_csv('weibo_sim.csv')['sim'].tolist()
    return [gossipcop_sim_df, twitter_sim_df, weibo_sim_df],['gossipcop','twitter','weibo']



if __name__ == '__main__':
    # text_model_path = '/media/shared_d/lyq/Model/chinese-roberta-wwm-ext'
    # image_model_path = '/media/shared_d/lyq/Model/swinv2-tiny-patch4-window16-256'
    # tokenizer = AutoTokenizer.from_pretrained(text_model_path)
    # image_processor = AutoImageProcessor.from_pretrained(image_model_path)
    # data = Util.dataloader.load_qwen_weibo_data('/media/shared_d/lyq/DataSet/FakeNews/weibo_dataset',
    #                                                 'weibo_llm_rationale',tokenizer=tokenizer,image_processor=image_processor,max_len=256,rationale_max_len=4,use_cache=True,use_image=True)
#
    # sim_model = SimModel(text_model_path,image_model_path)
    # sim_model.eval()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # sim_model.to(device)
    # batch_size = 256
    # sim = []
    # with torch.no_grad():
    #     for i in tqdm(range(0,len(data),batch_size)):
    #         batch = data_to_device(data[i:i+batch_size], device)
    #         batch_sim = sim_model(text_input=batch['content'],text_mask=batch['content_mask'], image_input=batch['image'])
    #         sim.extend(batch_sim.tolist())
#
    # sim_df = pd.DataFrame({'sim':sim})
    # sim_df.to_csv('weibo_sim.csv',index=False)
    sim_list,names =load_sim_list()
    # visualize_similarity_distribution(sim_list,names,'sim_distribution.png')
    plot_cumulative_distribution(sim_list,names,'CDF.png')







