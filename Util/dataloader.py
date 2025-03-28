import logging
import os
import pickle

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoImageProcessor

label_str2int_dict = {
    "real": 0,
    "fake": 1,
    "other": 2,
}

classified_label_str2int_dict = {
    "real": 0,
    "fake": 1
}

label_int2str_dict = {v: k for k, v in label_str2int_dict.items()}
classified_label_int2str_dict = {v: k for k, v in classified_label_str2int_dict.items()}



def sent2tensor(text, max_len, tokenizer):
    encoded = tokenizer.encode_plus(
        text,
        add_special_tokens=True,  # 添加特殊标记 [CLS] 和 [SEP]
        max_length=max_len,       # 最大长度
        truncation=True,          # 超出部分截断
        padding='max_length',     # 填充到最大长度
        return_tensors="pt"       # 返回 PyTorch 的张量
    )
    input_ids = encoded['input_ids'].squeeze(0)  # 提取 input_ids 并去掉多余的维度
    attention_mask = encoded['attention_mask'].squeeze(0)  # 提取 attention_mask 并去掉多余的维度
    return input_ids, attention_mask

def texts2tensor(texts, max_len, tokenizer):
    token_ids = []
    for i, text in enumerate(texts):
        token_ids.append(
            tokenizer.encode(text, max_length=max_len, add_special_tokens=True, padding='max_length',
                             truncation=True))
    token_ids = torch.tensor(token_ids)
    masks = torch.zeros(token_ids.shape)
    mask_token_id = tokenizer.pad_token_id
    for i, tokens in enumerate(token_ids):
        masks[i] = (tokens != mask_token_id)
    return token_ids, masks


def load_image_list(image_path_list, image_processor):
    """
    :param image_path_list: list[ str ]
    :param image_processor:
    :return: Tensor shape (batch_size, 3, height, width)
    """
    images = [Image.open(image_path).convert("RGB") for image_path in image_path_list]
    return image_processor(images = images,return_tensors = 'pt').pixel_values




class ARGDataset(Dataset):


    def text_data2tensor(self,data,max_len,tokenizer,rationale_names,rationale_max_len):
        data['content'],data['content_mask'] = texts2tensor(data['content'],max_len,tokenizer)
        for rationale_name in rationale_names:
            data[f'{rationale_name}_rationale'],data[f'{rationale_name}_rationale_mask'] = texts2tensor(data[f'{rationale_name}_rationale'],rationale_max_len,tokenizer)
            data[f'{rationale_name}_pred'] = torch.tensor(data[f'{rationale_name}_pred'], dtype=torch.long)
            data[f'{rationale_name}_acc'] = torch.tensor(data[f'{rationale_name}_acc'], dtype=torch.long)
        if 'caption' in data.keys():
            data['caption'],data['caption_mask'] = texts2tensor(data['caption'],max_len,tokenizer)

        data['label'] = torch.tensor(data['label'],dtype=torch.long)
        return data



    def __init__(self, df,use_cache,image_cache_path,tokenizer,image_processor,max_len,rationale_max_len,use_image):
        """
        :param df: {
            "content":"",
            "label":"0",
            "publish_date":1448118196000,
            "source_id":893,
            "td_rationale":"",
            "td_pred":"0",
            "td_acc":1,
            "cs_rationale":"无法确定。因为没有给出具体的消息内容，无法判断其真实性。",
            "cs_pred":"0",
            "cs_acc":1,
            "split":"train",
            'image_path': list[str],
        }
        :param use_cache: bool
        :param image_cache_path: str
        """
        logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len

        self.data = df.to_dict("list")
        self.rationale_names = [col_name.split('_')[0] for col_name in self.data.keys() if col_name.endswith("_rationale")]
        self.num_rationales = len(self.rationale_names)
        self.data = self.text_data2tensor(self.data, max_len, tokenizer,self.rationale_names,rationale_max_len=rationale_max_len) # 文本数据转换为tensor
        # 读取图片数据
        if use_image:
            if use_cache and os.path.exists(image_cache_path):
                logger.info("Loading image data from {}".format(image_cache_path))
                self.data['image'] = torch.load(image_cache_path,weights_only=True)
            else:
                logger.info('read image data.........')
                self.data['image'] = load_image_list(self.data['image_path'],self.image_processor)
                if use_cache:
                    cache_dir = os.path.dirname(image_cache_path)
                    if not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    logger.info("Save image cache at {}".format(image_cache_path))
                    torch.save(self.data['image'], image_cache_path)

        logger.info(
            f"load sum: {len(self.data['source_id'])} "
            f"real {(self.data['label'] == label_str2int_dict['real']).sum().item()} "
            f"fake {(self.data['label'] == label_str2int_dict['fake']).sum().item()}"
        )

    def __len__(self):
        return len(self.data['source_id'])


    def __getitem__(self, idx):
        """
        :param idx: int
        :return: {
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
        """
        item = {k:v[idx] for k,v in self.data.items()}
        return item




def merge_caption(df,root_path):
    caption_file_path = f'{root_path}/caption.csv'
    if os.path.exists(caption_file_path):
        caption_df = pd.read_csv(caption_file_path, encoding='utf-8')
        caption_df.rename(columns={'id': 'source_id'}, inplace=True)
        df = df.merge(caption_df, on='source_id', how='left')

    return df


def load_qwen_gossipcop_data(root_path,data_type,tokenizer,image_processor,max_len,rationale_max_len,use_cache,use_image):
    df = pd.read_csv(f'{root_path}/{data_type}.csv', encoding='utf-8')
    df = merge_caption(df,root_path)
    if use_image:
        df['image_path'] = df['image_id'].apply(lambda x: f'{root_path}/images/{x}_top_img.png')
    return ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_image_cache.pt', tokenizer, image_processor, max_len,rationale_max_len,
                         use_image)


def load_gpt_gossipcop_data(root_path,data_type,tokenizer,image_processor,max_len,rationale_max_len,use_cache,use_image):
    df = pd.read_json(f'{root_path}/{data_type}.json', encoding='utf-8')
    return ARGDataset(df, use_cache, None, tokenizer, image_processor, max_len,rationale_max_len,
                         use_image)

def load_gpt_weibo_data(root_path,data_type,tokenizer,image_processor,max_len,rationale_max_len,use_cache,use_image):
    df = pd.read_json(f'{root_path}/{data_type}.json', encoding='utf-8')
    df['label'] = df['label'].apply(lambda x: label_str2int_dict[x])
    df['td_pred'] = df['td_pred'].apply(lambda x: label_str2int_dict[x])
    df['cs_pred'] = df['cs_pred'].apply(lambda x: label_str2int_dict[x])
    return ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_image_cache.pt', tokenizer, image_processor, max_len,rationale_max_len,
                         use_image)

def load_qwen_weibo_data(root_path,data_type,tokenizer,image_processor,max_len,rationale_max_len,use_cache,use_image):
    def get_image_dict(root_path):
        image_dir_list = [f'{root_path}/nonrumor_images/', f'{root_path}/rumor_images/']
        image_dict = {}
        for image_dir in image_dir_list:
            image_dict.update({
                f.split('.')[0]: f'{image_dir}/{f}' for f in os.listdir(image_dir)
            })
        return image_dict

    df = pd.read_csv(f'{root_path}/{data_type}.csv', encoding='utf-8')
    df = merge_caption(df,root_path)
    if use_image:
        image_dict = get_image_dict(root_path)
        df['image_path'] = df['image_id'].apply(lambda x: image_dict[x])

    return ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_image_cache.pt', tokenizer, image_processor, max_len,rationale_max_len,
                         use_image)


def load_qwen_twitter_data(root_path,data_type,tokenizer,image_processor,max_len,rationale_max_len,use_cache,use_image):
    def get_image_path_dict():
        image_dir = f'{root_path}/images/'
        return { f.split('.')[0]: f'{image_dir}/{f}' for f in os.listdir(image_dir)}

    data_file_name = f'{root_path}/{data_type}.csv'
    df = pd.read_csv(data_file_name, encoding='utf-8')
    df = merge_caption(df,root_path)
    if use_image:
        image_dict = get_image_path_dict()
        df['image_path'] = df['image_id'].apply(lambda image_id: image_dict[image_id])

    return ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_image_cache.pt', tokenizer, image_processor, max_len,rationale_max_len,
                         use_image)


from torch.utils.data import DataLoader, DistributedSampler


def load_data(name,
              root_path,
              text_encoder_path,

              max_len,
              batch_size,
              shuffle,
              use_cache,
              image_encoder_path=None,
              **kwargs
              ):
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)
    use_image = kwargs.get('use_image',False)
    image_processor = AutoImageProcessor.from_pretrained(image_encoder_path) if use_image else None

    dataset_func_dict = {
        "qwen_gossipcop": load_qwen_gossipcop_data,
        "gpt_gossipcop": load_gpt_gossipcop_data,
        'gpt_weibo': load_gpt_weibo_data,
        "qwen_weibo": load_qwen_weibo_data,
        'qwen_twitter': load_qwen_twitter_data,
    }

    result = []

    for data_type in ['train', 'val', 'test']:
        shuffle_param = shuffle if data_type == 'train' else False
        rationale_max_len = kwargs.get('rationale_max_len',max_len)
        dataset = dataset_func_dict[name](root_path,data_type,tokenizer,image_processor,max_len,rationale_max_len,use_cache,use_image)

        # 创建DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle_param if data_type=='train' else False,  # 非分布式时才打乱
            num_workers=kwargs.get('num_workers', 4),  # 从kwargs获取num_workers
            pin_memory=kwargs.get('pin_memory', True)  # 从kwargs获取pin_memory
        )

        result.append(dataloader)  # 将DataLoader添加到结果列表

    return result



