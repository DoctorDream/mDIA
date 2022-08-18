import transformers
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from transformers import Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

from config import Config
from data_processor import DataTokenizer
from dialogpt_data_processor import DialogptDataTokenizer
from evaluator import MyMetric


class Trainer(object):

    def __init__(self, model_name: str, config: Config, translated=False):

        if 'mt5' in model_name:
            self.trainer: Trainer_mT5 = Trainer_mT5(model_name, config)
        elif 'DialoGPT' in model_name:
            self.trainer: Trainer_DialoGPT = Trainer_DialoGPT(model_name, config, translated)
        else:
            raise Exception('Wrong model')

    def train(self, small=False):
        self.trainer.train(small)


class Trainer_mT5(object):

    def __init__(self, model_name, config: Config):


        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)


        self.args = config.args_dict
        self.args = Seq2SeqTrainingArguments(**self.args)
        self.config = config
        self.max_input_length = config.token_parameters['max_input_length']
        self.max_target_length = config.token_parameters['max_target_length']


        self.metric = MyMetric(self.tokenizer, self.config)

    def train(self, small=False):


        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        tokenized_train_datasets, tokenized_val_datasets = self.load_data(small)
        trainer = Seq2SeqTrainer(
            self.model,
            self.args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_val_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.metric.compute_metrics(self.config.metric),
        )
        trainer.train()

    def load_data(self, small):

        data_tokenizer = DataTokenizer(self.tokenizer, self.config)
        if small:
            if type(small) == bool:
                small = 1000
            tokenized_train_datasets = data_tokenizer.get_tokenized_datasets('train', small, True)
            tokenized_val_datasets = data_tokenizer.get_tokenized_datasets('val', int(small / 10), True)
        else:
            tokenized_train_datasets = data_tokenizer.get_tokenized_datasets('train', shuffle=True)
            tokenized_val_datasets = data_tokenizer.get_tokenized_datasets('val', shuffle=True)
        return tokenized_train_datasets, tokenized_val_datasets


class Trainer_DialoGPT(object):
    def __init__(self, model_name, config: Config, translated):

        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer: transformers.GPT2TokenizerFast = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # dialoGPT使用eos_token作为pad

        self.translated = translated
        self.config = config
        if translated:
            self.config.args_dict['output_dir'] = f'../saved_model/translated_DialoGPT/{self.config.lang}_{self.config.args_dict["learning_rate"]}'
        else:
            self.config.args_dict['output_dir'] = f'../saved_model/DialoGPT/{self.config.lang}_{self.config.args_dict["learning_rate"]}'
        self.args = self.config.args_dict
        self.args = Seq2SeqTrainingArguments(**self.args)

        self.config.token_parameters['suffix'] = self.tokenizer.eos_token
        self.max_input_length = config.token_parameters['max_input_length']
        self.max_target_length = config.token_parameters['max_target_length']


        self.metric = MyMetric(self.tokenizer, self.config)

    def train(self, small=False):

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=self.model)
        tokenized_train_datasets, tokenized_val_datasets = self.load_data(small)
        trainer = Seq2SeqTrainer(
            self.model,
            self.args,
            train_dataset=tokenized_train_datasets,
            eval_dataset=tokenized_val_datasets,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.metric.compute_metrics(self.config.metric),
        )
        trainer.train()

    def load_data(self, small):

        data_tokenizer = DialogptDataTokenizer(self.tokenizer, self.config, self.translated)
        if small:
            if type(small) == bool:
                small = 10000
            tokenized_train_datasets = data_tokenizer.get_tokenized_datasets('train', small, True)
            tokenized_val_datasets = data_tokenizer.get_tokenized_datasets('val', int(small / 10), True)
        else:
            tokenized_train_datasets = data_tokenizer.get_tokenized_datasets('train', shuffle=True)
            tokenized_val_datasets = data_tokenizer.get_tokenized_datasets('val', shuffle=True)

        return tokenized_train_datasets, tokenized_val_datasets


