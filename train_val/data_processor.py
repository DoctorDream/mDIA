import os
from typing import Optional

import pandas as pd
import transformers
from datasets import load_dataset, Dataset
from tqdm import trange

from config import Config
from translator import Translator


class DataProcessor(object):
    def __init__(self, config: Config):

        self.raw_path = config.path['raw_data_path']
        self.train_path = '../data/train_data'
        self.eval_path = '../data/eval_data'
        self.test_path = '../data/test_data'

        self.lang = config.lang
        self.seed = config.seed

        self.raw_datasets = None
        self.train_datasets = None
        self.val_datasets = None
        self.test_datasets = None

    def generate_split(self, split):

        if len(split) == 2 and 0 <= split[0] <= split[1] and split[1] < 100:
            return [f'train[:{split[0]}%]', f'train[{split[0]}%:{split[1]}%]', f'train[{split[1]}%:]']
        else:
            raise Exception('wrong split data')

    def read_raw_data(self):

        file_path = {'train': f'{self.train_path}/{self.lang}_train.csv',
                     'eval': f'{self.eval_path}/{self.lang}_eval.csv',
                     'test': f'{self.test_path}/{self.lang}_test.csv'}
        raw_datasets = load_dataset('csv',
                                    data_files=file_path,
                                    # split=self.split,
                                    lineterminator='\n')
        self.raw_datasets = raw_datasets
        return raw_datasets

    def get_datasets(self, name, number=None, shuffle=False):

        if not self.raw_datasets:
            self.read_raw_data()
        if name == 'raw':
            return self.raw_datasets
        elif name == 'train':
            if not self.train_datasets:
                self.train_datasets = self.raw_datasets['train']
            dataset = self.train_datasets
        elif name == 'val':
            if not self.val_datasets:
                self.val_datasets = self.raw_datasets['eval']
            dataset = self.val_datasets
        elif name == 'test':
            if not self.test_datasets:
                self.test_datasets = self.raw_datasets['test']
            dataset = self.test_datasets
        else:
            raise Exception("Please input correct dataset naem in ['raw', 'train', 'val', 'test']")
        if shuffle:
            dataset = dataset.shuffle(seed=self.seed)
        if number:
            number = min(len(dataset), number)
            dataset = dataset.select(range(number))
        return dataset


class DataTokenizer(object):
    def __init__(self, tokenizer, config: Config):
        self.tokenizer: transformers.GPT2TokenizerFast = tokenizer

        self.max_input_length = config.token_parameters['max_input_length']
        self.max_target_length = config.token_parameters['max_target_length']
        self.prefix = config.token_parameters['prefix']
        self.suffix = config.token_parameters['suffix']
        self.batch_size = config.token_parameters['batch_size']

        self.data_processor = DataProcessor(config)

    def preprocess_function(self, example):

        inputs = example['source_body']
        targets = example['target_body']


        model_inputs = self.tokenizer(inputs,
                                      padding='max_length',
                                      max_length=self.max_input_length, truncation=True)

        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(targets,
                                    padding='max_length',
                                    max_length=self.max_target_length, truncation=True)

        model_inputs['labels'] = labels['input_ids']

        return model_inputs


    def tokenize_datasets(self, datasets: Dataset):
        tokenized_datasets = datasets.map(self.preprocess_function, batched=True, batch_size=self.batch_size)
        return tokenized_datasets

    def get_tokenized_datasets(self, name, number: Optional[int] = None, shuffle: bool = False):
        datasets = self.data_processor.get_datasets(name, number, shuffle)
        tokenized_datasets = self.tokenize_datasets(datasets)
        return tokenized_datasets


class DataTranslator(object):
    def __init__(self, config: Config):
        self.lang = config.lang
        self.translator = Translator(self.lang, 'en')
        self.raw_data_path = config.path['raw_data_path']
        self.save_data_path = '../data/translated_data'
        self.total_num = 15000
        self.translate_batch = 25

    def read_and_write(self):
        print(f"\nReading {self.lang}_cleaned_dialogue_data.csv")
        dialogue = self.read()
        print(f"Data has been processed")

        print(f"\nTranslating {self.lang}_cleaned_dialogue_data.csv")
        translated_data = self.translate(dialogue)
        print("Data has been translated")

        save_path = f'{self.save_data_path}/{self.lang}2en_cleaned_dialogue_data.csv'
        print(f"\nWriting into {self.lang}2en_cleaned_dialogue_data.csv")
        self.write(save_path, translated_data)
        print(f"Writing over")

    def read(self):
        path = f'{self.raw_data_path}/{self.lang}_cleaned_dialogue_data.csv'
        if not os.path.exists(path):
            raise Exception(f'{path} is not exist')
        dialogue = pd.read_csv(path, nrows=self.total_num, lineterminator='\n')
        return dialogue

    def translate(self, dialogue):
        translated = pd.DataFrame(
            columns=['lang', 'title', 'source_body', 'target_body', 'link_id', 'source_id', 'target_id',
                     'translated_source_body', 'translated_target_body'])
        source_body = list(dialogue['source_body'])
        target_body = list(dialogue['target_body'])
        total_len = len(source_body)
        translated_source_body, translated_target_body = [], []

        source_body = [source_body[i:i + self.translate_batch] for i in
                       range(0, len(source_body), self.translate_batch)]
        target_body = [target_body[i:i + self.translate_batch] for i in
                       range(0, len(target_body), self.translate_batch)]

        print(f"{total_len} conversations in total，{self.translate_batch} conversations will be translated every time")
        for i in trange(len(source_body), desc='Process'):
            s, t = source_body[i], target_body[i]
            translated_source_body.extend(self.translator.translate(s))
            translated_target_body.extend(self.translator.translate(t))
        for index in dialogue.index:
            line: pd.Series = dialogue.loc[index]
            line['translated_source_body'] = translated_source_body[index]
            line['translated_target_body'] = translated_target_body[index]
            translated = translated.append(line, ignore_index=True)
        return translated

    def write(self, save_path, data):
        if os.path.exists(save_path):
            header = False
        else:
            header = True
        data.to_csv(save_path, mode='a+', index=False, header=header, encoding='utf-8', line_terminator='\n')


if __name__ == '__main__':

    config = Config('zh')
    a = DataProcessor(config)
    raw = a.read_raw_data()
    print(raw)
    print(raw['train'])

    langs = ['lt', 'cs', 'es', 'sl', 'nl', 'ar', 'he', 'id', 'no', 'da', 'ro', 'el', 'fi', 'it']
    for lang in langs:
        try:
            config = Config(lang)
            datatranslator = DataTranslator(config)
            datatranslator.read_and_write()
        except OSError:
            pass
        except ValueError:
            continue
