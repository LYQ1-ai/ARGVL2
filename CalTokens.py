import pandas as pd
from transformers import AutoTokenizer
# file_path = 'C:\\Users\\lyq\\DataSet\\FakeNews\\gossipcop\\gossipcop_llm_rationales.csv'
file_path = 'C:\\Users\\lyq\\DataSet\\FakeNews\\weibo_dataset\\weibo_llm_rationale.csv'
model_path = 'C:\\Users\\lyq\\Model\\chinese-roberta-wwm-ext'
df = pd.read_csv(file_path)

tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_text_tokens(text):
    return tokenizer.encode_plus(text, add_special_tokens=True, return_tensors="pt").input_ids.shape[-1]

for col_name in df.columns:
    if col_name.endswith('_rationale') or col_name=='content':
        texts = df[col_name].tolist()
        seq_len = df[col_name].apply(get_text_tokens).sum() // len(df[col_name])
        print(f'{col_name} seq_len: {seq_len}')


