from typing import Optional

import transformers
from datasets import load_dataset, Dataset

from config import Config
from data_processor import DataProcessor, DataTokenizer


class DialogptDataProcessor(DataProcessor):
    def __init__(self, config: Config, translated=False):
        super().__init__(config)
        self.translated = translated
        self.save_data_path = '../data/translated_data'
        self.train_path = '../data/train_data'
        self.eval_path = '../data/eval_data'
        self.test_path = '../data/test_data'
        self.translated_train_path = '../data/translated_data/train_data'
        self.translated_eval_path = '../data/translated_data/eval_data'
        self.translated_test_path = '../data/translated_data/test_data'

    def read_raw_data(self):
        if self.translated:
            # file_path = f'{self.save_data_path}/{self.lang}2en_cleaned_dialogue_data.csv'
            file_path = {'train': f'{self.translated_train_path}/{self.lang}2en_train.csv',
                         'eval': f'{self.translated_eval_path}/{self.lang}2en_eval.csv',
                         'test': f'{self.translated_test_path}/{self.lang}2en_test.csv'}
        else:
            # file_path = f'{self.raw_path}/{self.lang}_cleaned_dialogue_data.csv'
            file_path = {'train': f'{self.train_path}/{self.lang}_train.csv',
                         'eval': f'{self.eval_path}/{self.lang}_eval.csv',
                         'test': f'{self.test_path}/{self.lang}_test.csv'}
        raw_datasets = load_dataset('csv',
                                    data_files=file_path,
                                    # split=self.split,
                                    lineterminator='\n')
        self.raw_datasets = raw_datasets
        return raw_datasets


class DialogptDataTokenizer(DataTokenizer):
    def __init__(self, tokenizer, config: Config, translated=False):
        self.tokenizer: transformers.GPT2TokenizerFast = tokenizer
        self.translated = translated

        self.max_input_length = config.token_parameters['max_input_length']
        self.max_target_length = config.token_parameters['max_target_length']
        self.suffix = self.tokenizer.eos_token
        self.batch_size = config.token_parameters['batch_size']

        self.data_processor = DialogptDataProcessor(config, self.translated)

    def train_preprocess_function(self, example):
        if self.translated:
            source, target = 'translated_source_body', 'translated_target_body'
        else:
            source, target = 'source_body', 'target_body'
        inputs = example[source]
        targets = example[target]


        inputs = self.tokenizer(inputs, max_length=self.max_input_length - 1, truncation=True)
        targets = self.tokenizer(targets, max_length=self.max_input_length - 1, truncation=True)

        input = []
        for i, t in zip(inputs['input_ids'], targets['input_ids']):
            input.append(i + [self.tokenizer.eos_token_id] + t + [self.tokenizer.eos_token_id])
        input = self.tokenizer.batch_decode(input)
        model_inputs = self.tokenizer(input)
        print(model_inputs)
        model_inputs['labels'] = model_inputs['input_ids']

        return model_inputs


    def test_preprocess_function(self, example):

        return None

    def tokenize_datasets(self, datasets: Dataset, name):

        preprocess_dict = {'train': self.train_preprocess_function,
                           'val': self.train_preprocess_function,
                           'test': self.test_preprocess_function}
        tokenized_datasets = datasets.map(preprocess_dict[name], batched=True, batch_size=self.batch_size)
        return tokenized_datasets

    def get_tokenized_datasets(self, name, number: Optional[int] = None, shuffle: bool = False):
        datasets = self.data_processor.get_datasets(name, number, shuffle)
        tokenized_datasets = self.tokenize_datasets(datasets, name)
        return tokenized_datasets