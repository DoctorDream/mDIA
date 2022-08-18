import os

import numpy as np
import pandas as pd

np.random.seed(66)
global_seed = 66
cleaned_data_path = '../data/cleaned_data'
train_data_path = '../data/train_data'
eval_data_path = '../data/eval_data'
test_data_path = '../data/test_data'
# cleaned_data_path = '../data/translated_data'
# train_data_path = '../data/translated_data/train_data'
# eval_data_path = '../data/translated_data/eval_data'
# test_data_path = '../data/translated_data/test_data'


def write(lang, data: list):
    if os.path.exists(train_data_path) is not True:
        os.mkdir(train_data_path)
    if os.path.exists(eval_data_path) is not True:
        os.mkdir(eval_data_path)
    if os.path.exists(test_data_path) is not True:
        os.mkdir(test_data_path)
    save_path = [f'{train_data_path}/{lang}_train.csv',
                 f'{eval_data_path}/{lang}_eval.csv',
                 f'{test_data_path}/{lang}_test.csv', ]

    data[0].to_csv(save_path[0], index=False, header=True, encoding='utf-8', line_terminator='\n')



    data[1].to_csv(save_path[1], index=False, header=True, encoding='utf-8', line_terminator='\n')



    data[2].to_csv(save_path[2], index=False, header=True, encoding='utf-8', line_terminator='\n')



def less(langs):
    for lang in langs:
        path = f'{cleaned_data_path}/{lang}_cleaned_dialogue_data.csv'
        if not os.path.exists(path):
            raise Exception(f'{path} is not exist')
        print(f"开始处理{path}")
        dialogue = pd.read_csv(path, lineterminator='\n')
        length = len(dialogue)
        train_len = int((length - 1000) / 1.1)
        train_data = dialogue[:train_len]
        eval_data = dialogue[train_len:-1000]
        test_data = dialogue[-1000:]
        data = [train_data, eval_data, test_data]
        write(lang, data)


def more(langs):
    for lang in langs:
        # path = f'{cleaned_data_path}/{lang}_cleaned_dialogue_data.csv'
        path = f'{cleaned_data_path}/{lang}_cleaned_dialogue_data.csv'
        if not os.path.exists(path):
            raise Exception(f'{path} is not exist')
        print(f"Dealing {path}")
        dialogue = pd.read_csv(path, lineterminator='\n')
        length = len(dialogue)
        train_len = int(length * 0.85)
        eval_len = int(length * 0.93)

        train_data = dialogue[:train_len]
        train_data = train_data.iloc[np.random.default_rng(global_seed).permutation(len(train_data))][:10000]

        eval_data = dialogue[train_len:eval_len]
        eval_data = eval_data.iloc[np.random.default_rng(global_seed).permutation(len(eval_data))][:1000]

        test_data = dialogue[eval_len:]
        test_data = test_data.iloc[np.random.default_rng(global_seed).permutation(len(test_data))][:1000]

        data = [train_data, eval_data, test_data]
        write(lang, data)


if __name__ == '__main__':
    langs_q = ['sw', 'cy', 'ca', 'th', 'ml', 'af', 'vi', 'mk', 'hi', 'lv', 'fa', 'ko', 'sq', 'sk', 'uk', 'lt', 'bg']
    langs_w = ['cs', 'et', 'es', 'sl', 'sv', 'nl', 'ar', 'pl', 'he', 'id', 'no', 'ja', 'da', 'pt', 'el', 'zh', 'ro',
               'fi', 'tr', 'hu', 'de', 'tl', 'it', 'hr', 'ru', 'fr']
    langs_t = ['pa', 'mr', 'ur', 'bn', 'cy', 'ca', 'af', 'ml', 'th', 'mk', 'vi', 'hi', 'lv', 'uk', 'bg', 'ko', 'sq',
               'sk', 'cs', 'et', 'tr', 'ja', 'tl', 'id', 'hu', 'pl', 'ar', 'de', 'it', 'nl', 'es', 'fr', 'sv', 'ru',
               'zh', 'da', 'fi']
    # less(langs_q)
    more(['en'])
