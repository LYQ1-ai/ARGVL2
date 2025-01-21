import logging
import os
import pickle

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
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
    """
    :param texts: List[ str ]
    :param max_len: int
    :param tokenizer:
    :return: Tensor shape (batch_size, max_len), Tensor shape (batch_size, max_len)
    """
    encodings = tokenizer(
        texts,
        add_special_tokens=True,  # 添加特殊标记 [CLS] 和 [SEP]
        max_length=max_len,  # 最大长度
        truncation=True,  # 超出部分截断
        padding='max_length',  # 填充到最大长度
        return_attention_mask=True,  # 返回 attention mask
        return_tensors="pt"  # 返回 PyTorch 的张量
    )
    return encodings['input_ids'], encodings['attention_mask']

def load_image_list(image_path_list, image_processor):
    """
    :param image_path_list: list[ str ]
    :param image_processor:
    :return: Tensor shape (batch_size, 3, height, width)
    """
    images = [Image.open(image_path).convert("RGB") for image_path in image_path_list]
    return image_processor(images = images,return_tensors = 'pt').pixel_values





class ARGDataset(Dataset):


    def data2tensor(self,data,max_len,tokenizer,image_processor,use_image):
        data['content'],data['content_mask'] = texts2tensor(data['content'],max_len,tokenizer)
        data['td_rationale'],data['td_rationale_mask'] = texts2tensor(data['td_rationale'],max_len,tokenizer)
        data['cs_rationale'],data['cs_rationale_mask'] = texts2tensor(data['cs_rationale'],max_len,tokenizer)
        if use_image:
            data['image'] = load_image_list(data['image_path'],image_processor)
        if 'caption' in data.keys():
            data['caption'],data['caption_mask'] = texts2tensor(data['caption'],max_len,tokenizer)

        data['label'] = torch.tensor(data['label'],dtype=torch.long)
        data['td_pred'] = torch.tensor(data['td_pred'],dtype=torch.long)
        data['td_acc'] = torch.tensor(data['td_acc'],dtype=torch.long)
        data['cs_pred'] = torch.tensor(data['cs_pred'],dtype=torch.long)
        data['cs_acc'] = torch.tensor(data['cs_acc'],dtype=torch.long)
        return data



    def __init__(self, df,use_cache,cache_path,tokenizer,image_processor,max_len,use_image):
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
            "split":"train"
        }
        :param use_cache: bool
        :param cache_path: str
        """
        logger = logging.getLogger(__name__)
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_len = max_len
        if use_cache:
            logger.info(f"Using cache: {cache_path}")
            if os.path.exists(cache_path):
                logger.info(f"Loading data from cache: {cache_path}")
                with open(cache_path, "rb") as f:
                    self.data = pickle.load(f)
            else:
                logger.info(f"Creating data from cache: {cache_path}")
                self.data = df.to_dict("list")
                self.data = self.data2tensor(self.data,max_len,tokenizer,image_processor,use_image)
                cache_dir = os.path.dirname(cache_path)
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)

                with open(cache_path, "wb") as f:
                    pickle.dump(self.data, f)
        else:
            logger.info(f"Creating data from scratch,not using cache")
            self.data = df.to_dict("list")
            self.data = self.data2tensor(self.data, max_len, tokenizer, image_processor, use_image)

        logger.info(f"Creating dataset with {len(self.data)} examples")
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


def load_qwen_gossipcop_data(root_path,data_type,tokenizer,image_processor,max_len,batch_size,shuffle,use_cache,use_image):
    df = pd.read_csv(f'{root_path}/{data_type}.csv', encoding='utf-8')
    df = merge_caption(df,root_path)
    if use_image:
        df['image_path'] = df['image_id'].apply(lambda x: f'{root_path}/images/{x}.png')
    dataset = ARGDataset(df,use_cache,f'{root_path}/cache/{data_type}_cache.pkl',tokenizer,image_processor,max_len,use_image)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=4,pin_memory=True)


def load_gpt_gossipcop_data(root_path,data_type,tokenizer,image_processor,max_len,batch_size,shuffle,use_cache,use_image):
    df = pd.read_json(f'{root_path}/{data_type}.json', encoding='utf-8')
    dataset = ARGDataset(df,use_cache,f'{root_path}/cache/{data_type}_cache.pkl',tokenizer,image_processor,max_len,use_image)
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,num_workers=4,pin_memory=True)

def load_gpt_weibo_data(root_path,data_type,tokenizer,image_processor,max_len,batch_size,shuffle,use_cache,use_image):
    df = pd.read_json(f'{root_path}/{data_type}.json', encoding='utf-8')
    df['label'] = df['label'].apply(lambda x: label_str2int_dict[x])
    df['td_pred'] = df['td_pred'].apply(lambda x: label_str2int_dict[x])
    df['cs_pred'] = df['cs_pred'].apply(lambda x: label_str2int_dict[x])
    dataset = ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_cache.pkl', tokenizer, image_processor, max_len,
                         use_image)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True)

def load_qwen_weibo_data(root_path,data_type,tokenizer,image_processor,max_len,batch_size,shuffle,use_cache,use_image):
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

    dataset = ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_cache.pkl', tokenizer, image_processor, max_len,
                         use_image)
    return DataLoader(
        dataset,  # 传入自定义的 Dataset
        batch_size=batch_size,  # 批量大小
        shuffle=shuffle,  # 是否打乱数据
        num_workers=4,  # 数据加载线程数
        pin_memory=True
    )

def load_qwen_twitter_data(root_path,data_type,tokenizer,image_processor,max_len,batch_size,shuffle,use_cache,use_image):
    def get_image_path_dict():
        image_dir = f'{path}/images/'
        return { f.split('.')[0]: f'{image_dir}/{f}' for f in os.listdir(image_dir)}

    data_file_name = f'{root_path}/{data_type}.csv'
    df = pd.read_csv(data_file_name, encoding='utf-8')
    df = merge_caption(df,root_path)
    if use_image:
        image_dict = get_image_path_dict()
        df['image_path'] = df['image_id'].apply(lambda image_id: image_dict[image_id])

    dataset = ARGDataset(df, use_cache, f'{root_path}/cache/{data_type}_cache.pkl', tokenizer, image_processor, max_len,
               use_image)
    return DataLoader(
        dataset,  # 传入自定义的 Dataset
        batch_size=batch_size,  # 批量大小
        shuffle=True,  # 是否打乱数据
        num_workers=4,
        pin_memory=True
    )



def load_data(name,
              root_path,
              text_encoder_path,
              image_encoder_path,
              max_len,
              batch_size,
              shuffle,
              use_cache,
              use_image,
              ):
    tokenizer = AutoTokenizer.from_pretrained(text_encoder_path)

    image_processor = AutoImageProcessor.from_pretrained(image_encoder_path) if use_image else None

    dataset_func_dict = {
        "qwen_gossipcop": load_qwen_gossipcop_data,
        "gpt_gossipcop": load_gpt_gossipcop_data,
        'gpt_weibo': load_gpt_weibo_data,
        "qwen_weibo": load_qwen_weibo_data,
        'qwen_twitter': load_qwen_twitter_data,
        # TODO: add more datasets here
    }
    result = []
    for data_type in ['train','val','test']:
        result.append(dataset_func_dict[name]
                      (root_path,data_type,tokenizer,image_processor,max_len,batch_size,shuffle,use_cache,use_image))

    return result




