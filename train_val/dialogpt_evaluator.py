import numpy as np
from datasets import load_metric
from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM
from tqdm import trange
from evaluator import MyMetric
from data_processor import DataTokenizer, DataProcessor
from dialogpt_data_processor import DialogptDataTokenizer, DialogptDataProcessor
from evaluator import Predictor
from config import Config
from translator import Translator
import torch
device = torch.device(f"cuda")
cpu = torch.device("cpu")

class DialogptPredictor(Predictor):
    def __init__(self, model, tokenizer, config: Config):
        super(DialogptPredictor, self).__init__(model, tokenizer, config)
        self.text_process = lambda text: text + self.tokenizer.eos_token

    def greedy_predict(self, source_text: str):

        source_text = self.text_process(source_text)
        inputs = self.tokenizer.encode(source_text, return_tensors='pt').to(device)
        outputs = self.model.generate(inputs, max_length=200, pad_token_id=self.tokenizer.eos_token_id).to(cpu)
        decoded_sentences = self.tokenizer.batch_decode(outputs[:, inputs.shape[-1]:], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded_sentences

    def beam_search_predict(self, source_text: str, num_beams=5, no_repeat_ngram_size=3, num_return_sequences=1):

        source_text = self.text_process(source_text)
        inputs = self.tokenizer.encode(source_text, return_tensors='pt').to(device)

        outputs = self.model.generate(inputs,
                                      max_length=200,
                                      num_beams=num_beams,
                                      early_stopping=True,
                                      no_repeat_ngram_size=no_repeat_ngram_size,
                                      num_return_sequences=num_return_sequences,
                                      pad_token_id=self.tokenizer.eos_token_id).to(cpu)
        decoded_sentences = self.tokenizer.batch_decode(outputs[:, inputs.shape[-1]:], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded_sentences

    def sample_predict(self, source_text: str, temperature=0.7, top_k=0, top_p: float = 0, num_return_sequences=1):
        source_text = self.text_process(source_text)
        inputs = self.tokenizer.encode(source_text, return_tensors='pt').to(device)
        if top_k or top_p:
            outputs = self.model.generate(inputs,
                                          do_sample=True,
                                          max_length=200,
                                          top_k=top_k,
                                          top_p=top_p,
                                          temperature=temperature,
                                          num_return_sequences=num_return_sequences,
                                          pad_token_id=self.tokenizer.eos_token_id).to(cpu)
        else:
            outputs = self.model.generate(inputs,
                                          do_sample=True,
                                          max_length=200,
                                          temperature=temperature,
                                          num_return_sequences=num_return_sequences,
                                          pad_token_id=self.tokenizer.eos_token_id).to(cpu)
        decoded_sentences = self.tokenizer.batch_decode(outputs[:, inputs.shape[-1]:], skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded_sentences


class DialogptEvaluator(object):
    def __init__(self, model_name, config: Config, translated=False):

        # 加载模型和分词器
        print(f"Loading {model_name} ...")
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        print("Loading done")

        self.translated = translated
        self.config = config
        self.max_input_length = config.token_parameters['max_input_length']
        self.max_target_length = config.token_parameters['max_target_length']
        self.predict_strategy = config.predict_strategy
        self.lang = self.config.lang
        self.config.token_parameters['suffix'] = self.tokenizer.eos_token

        self.predictor = DialogptPredictor(self.model, self.tokenizer, self.config)
        self.metric = MyMetric(self.tokenizer, self.config)

    def predict(self, source_text):
        if self.predict_strategy[0] == 'greedy':
            respond = self.predictor.greedy_predict(source_text)
        elif self.predict_strategy[0] == 'beam_search':
            respond = self.predictor.beam_search_predict(source_text, **self.predict_strategy[1])
        elif self.predict_strategy[0] == 'sample':
            respond = self.predictor.sample_predict(source_text, **self.predict_strategy[1])
        return respond

    def load_data(self, small):

        data_tokenizer = DialogptDataTokenizer(self.tokenizer, self.config, self.translated)


        if small:
            if type(small) == bool:
                small = 200
            tokenized_test_datasets = data_tokenizer.get_tokenized_datasets('test', small, True)
        else:
            tokenized_test_datasets = data_tokenizer.get_tokenized_datasets('test', shuffle=True)
        return tokenized_test_datasets


    def evaluate(self, small=True):

        datasets = self.load_data(small)
        print('-' * 100 + f'\nModel evaluating:\nPredict strategy: {self.config.predict_strategy[0]} \nmetric: {self.config.metric} \n' + '-' * 100)

        if self.translated:
            results = self.eval_translated(datasets)
        else:
            results = self.eval_raw(datasets)
        if self.config.metric == 'all':
            print('-' * 100 + f'\nScores:')
            for strategy, result in results.items():
                print(f'\n{strategy} score are:')
                for key, value in result.items():
                    print(f'{key}: {value}')

        else:
            print('-' * 100 + f'\nScores:\n\n{self.config.metric} score are:')
            for key, value in results.items():
                print(f'{key}: {value}')
            print('-' * 100)

    def eval_raw(self, datasets):
        eval_preds = [[], []]
        examples = []
        for i in trange(len(datasets)):
            line = datasets[i]
            source_body = self.tokenizer.encode(line['source_body'], max_length=self.max_input_length, truncation=True)
            source_body = self.tokenizer.decode(source_body, skip_special_tokens=True)
            label = self.tokenizer.encode(line['target_body'], max_length=self.max_target_length, truncation=True)
            pred = self.predict(source_body)
            if i < 10:
                examples.append([i, source_body, self.tokenizer.decode(label, skip_special_tokens=True), pred[0]])
            pred = self.tokenizer.encode(pred[0])
            label = self.tokenizer.encode(line['target_body'], max_length=self.max_target_length, truncation=True)
            eval_preds[0].append(pred)
            eval_preds[1].append(label)
        result = self.metric.compute_metrics(self.config.metric)(eval_preds)
        print('-' * 100 + f"\n{len(examples)} conversations：")
        for example in examples:
            print(f"The {example[0] + 1} conversation:")
            print(f"Context: {example[1]}")
            print(f"Response: {example[2]}")
            print(f"Prediction: {example[3]}\n")
        return result

    def eval_translated(self, datasets):
        eval_preds = [[], []]
        examples = []
        translator = Translator('en', self.lang)
        for i in trange(len(datasets)):
            line = datasets[i]

            translated_source_body = self.tokenizer.encode(line['translated_source_body'], max_length=self.max_input_length, truncation=True)
            translated_source_body = self.tokenizer.decode(translated_source_body, skip_special_tokens=True)
            label = self.tokenizer.encode(line['target_body'], max_length=self.max_target_length, truncation=True)
            pred_en = self.predict(translated_source_body)
            pred = translator.translate(pred_en)
            if i < 10:
                examples.append([i,
                                 line['source_body'],
                                 self.tokenizer.decode(label, skip_special_tokens=True),
                                 pred[0],
                                 translated_source_body,
                                 line['translated_target_body'],
                                 pred_en[0]])
            pred = self.tokenizer.encode(pred[0])
            eval_preds[0].append(pred)
            eval_preds[1].append(label)
        result = self.metric.compute_metrics(self.config.metric)(eval_preds)
        print('-' * 100 + f"\n{len(examples)} conversations：")
        for example in examples:
            print(f"The {example[0] + 1} conversation:")
            print(f"{self.lang}Context: {example[1]}")
            print(f"{self.lang}Response: {example[2]}")
            print(f"{self.lang}Prediction: {example[3]}")
            print(f"en Context: {example[4]}")
            print(f"en Response: {example[5]}")
            print(f"en Prediction: {example[6]}\n")
        return result




if __name__ == '__main__':
    config = Config('ja', strategy='sample')
    evaluator = DialogptEvaluator("microsoft/DialoGPT-large", config, translated=True)
    print(evaluator.predict("逆に欲しい定期"))
    # evaluator.evaluate(small=10)