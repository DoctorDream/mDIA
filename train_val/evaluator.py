import numpy as np
from datasets import load_metric
from tqdm import trange
from transformers import PreTrainedTokenizerFast, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from config import Config
from data_processor import DataTokenizer, DataProcessor
from distinct_n import distinct_n_sentence_level, distinct_n_corpus_level, entropy
import torch
device = torch.device(f"cuda")
cpu = torch.device("cpu")

class Predictor(object):

    def __init__(self, model, tokenizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = [config.token_parameters['max_input_length'], config.token_parameters['max_target_length']]

    def greedy_predict(self, source_text: str):
        inputs = self.tokenizer(source_text, return_tensors='pt', padding='max_length', max_length=self.max_length[0],
                                truncation=True).to(device)
        outputs = self.model.generate(inputs.input_ids, max_length=self.max_length[1]).to(cpu)
        decoded_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded_sentences

    def beam_search_predict(self, source_text: str, num_beams=5, no_repeat_ngram_size=2, num_return_sequences=1):
        inputs = self.tokenizer(source_text, return_tensors='pt', padding='max_length', max_length=self.max_length[0],
                                truncation=True).to(device)
        outputs = self.model.generate(inputs.input_ids,
                                      max_length=self.max_length[1],
                                      num_beams=num_beams,
                                      early_stopping=True,
                                      no_repeat_ngram_size=no_repeat_ngram_size,
                                      num_return_sequences=num_return_sequences).to(cpu)
        decoded_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded_sentences

    def sample_predict(self, source_text: str, temperature=0.7, top_k=0, top_p: float = 0, num_return_sequences=1):
        inputs = self.tokenizer(source_text, return_tensors='pt', padding='max_length', max_length=self.max_length[0],
                                truncation=True).to(device)
        if top_k or top_p:
            outputs = self.model.generate(inputs.input_ids,
                                          do_sample=True,
                                          max_length=self.max_length[1],
                                          top_k=top_k,
                                          top_p=top_p,
                                          temperature=temperature,
                                          num_return_sequences=num_return_sequences).to(cpu)
        else:
            outputs = self.model.generate(inputs.input_ids,
                                          do_sample=True,
                                          max_length=self.max_length[1],
                                          temperature=temperature,
                                          num_return_sequences=num_return_sequences).to(cpu)
        decoded_sentences = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        torch.cuda.empty_cache()
        return decoded_sentences


class MyMetric(object):

    def __init__(self, tokenizer, config):
        self.metric = None
        self.tokenizer = tokenizer
        self.mt5_tokenizer = AutoTokenizer.from_pretrained('google/mt5-base')
        self.predict_strategy = config.predict_strategy
        self.config = config

    def compute_metrics(self, name):
        if name == 'all':
            return self.evaluate_all
        elif name == 'bleu':
            self.metric = load_metric('bleu')
            return self.bleu
        elif name == 'sacrebleu':
            self.metric = load_metric('sacrebleu')
            return self.sacrebleu
        elif name == 'bertscore':
            self.metric = load_metric('bertscore')
            return self.bertscore
        elif name == 'sentence_distinct':
            return self.sentence_distinct
        elif name == 'corpus_distinct':
            return self.corpus_distinct
        elif name == 'entropy':
            return self.entropy

    def evaluate_all(self, eval_preds):
        all_result = {}
        self.metric = load_metric('bleu')
        all_result['bleu'] = self.bleu(eval_preds)
        self.metric = load_metric('sacrebleu')
        all_result['sacrebleu'] = self.sacrebleu(eval_preds)
        self.metric = load_metric('bertscore')
        all_result['bertscore'] = self.bertscore(eval_preds)
        all_result['sentence_distinct'] = self.sentence_distinct(eval_preds)
        all_result['corpus_distinct'] = self.corpus_distinct(eval_preds)
        all_result['entropy'] = self.entropy(eval_preds)
        return all_result

    def sacrebleu(self, eval_preds):
        tokenizer: PreTrainedTokenizerFast = self.tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]


        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        return result

    def bleu(self, eval_preds):

        tokenizer: PreTrainedTokenizerFast = self.tokenizer

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds, decoded_labels = [], []

        for pred in preds:
            pred = tokenizer.decode(pred, skip_special_tokens=True)
            pred = self.mt5_tokenizer.tokenize(pred)
            pred = [_pred.strip('▁') for _pred in pred]
            decoded_preds.append(pred)
        for label in labels:
            label = np.where(label != -100, label, tokenizer.pad_token_id)
            label = tokenizer.decode(label, skip_special_tokens=True)
            label = self.mt5_tokenizer.tokenize(label)
            # label = [label]
            label = [[_label.strip('▁') for _label in label]]
            decoded_labels.append(label)

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)

        return result

    def bertscore(self, eval_preds):
        tokenizer: PreTrainedTokenizerFast = self.tokenizer
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip('▁') for pred in decoded_preds]
        decoded_labels = [label.strip('▁') for label in decoded_labels]

        result: dict = self.metric.compute(predictions=decoded_preds, references=decoded_labels, lang=self.config.lang)
        avg_result = {'precision': sum(result['precision']) / len(result['precision']),
                      'recall': sum(result['recall']) / len(result['recall']),
                      'f1': sum(result['f1']) / len(result['f1']),
                      'hashcode': result['hashcode']}
        return avg_result

    def corpus_distinct(self, eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        preds = [self.mt5_tokenizer.encode(pred) for pred in preds]


        result = {'distinct_1': [], 'distinct_2': [], 'distinct_4': []}
        result['distinct_1'] = distinct_n_corpus_level(preds, 1)
        result['distinct_2'] = distinct_n_corpus_level(preds, 2)
        result['distinct_4'] = distinct_n_corpus_level(preds, 4)
        # print(result)

        return result

    def sentence_distinct(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        preds = [self.mt5_tokenizer.encode(pred) for pred in preds]

        result = {'distinct_1': [], 'distinct_2': [], 'distinct_4': []}
        for sentence in preds:
            result['distinct_1'].append(distinct_n_sentence_level(sentence, 1))
            result['distinct_2'].append(distinct_n_sentence_level(sentence, 2))
            result['distinct_4'].append(distinct_n_sentence_level(sentence, 4))
        avg_result = {'distinct_1': sum(result['distinct_1']) / len(result['distinct_1']),
                      'distinct_2': sum(result['distinct_2']) / len(result['distinct_2']),
                      'distinct_4': sum(result['distinct_4']) / len(result['distinct_4'])}
        return avg_result

    def entropy(self, eval_preds):

        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        preds = [self.mt5_tokenizer.encode(pred) for pred in preds]

        result = {'entropy_1': [], 'entropy_2': [], 'entropy_4': []}
        result['entropy_1'] = entropy(preds, 1)
        result['entropy_2'] = entropy(preds, 2)
        result['entropy_4'] = entropy(preds, 4)
        print(result)

        return result

class Evaluator(object):
    def __init__(self, model_name, config: Config):
        self.config = config

        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.max_input_length = config.token_parameters['max_input_length']
        self.max_target_length = config.token_parameters['max_target_length']
        self.predict_strategy = config.predict_strategy

        self.predictor = Predictor(self.model, self.tokenizer, self.config)
        self.metric = MyMetric(self.tokenizer, self.config)

    def predict(self, source_text):
        if self.predict_strategy[0] == 'greedy':
            return self.predictor.greedy_predict(source_text)
        elif self.predict_strategy[0] == 'beam_search':
            return self.predictor.beam_search_predict(source_text, **self.predict_strategy[1])
        elif self.predict_strategy[0] == 'sample':
            return self.predictor.sample_predict(source_text, **self.predict_strategy[1])

    def load_data(self, small):


        data_tokenizer = DataTokenizer(self.tokenizer, self.config)
        if small:
            if type(small) == bool:
                small = 200

            tokenized_test_datasets = data_tokenizer.get_tokenized_datasets('test', small, True)
        else:

            tokenized_test_datasets = data_tokenizer.get_tokenized_datasets('test', shuffle=True)
        return tokenized_test_datasets

    def evaluate(self, small=True):

        datasets = self.load_data(small)
        eval_preds = [[], []]
        print('-' * 100 + f'\nModel evaluating:\nPredict strategy: {self.config.predict_strategy[0]} \nmetric: {self.config.metric} \n' + '-' * 100)
        examples = []
        for i in trange(len(datasets)):
            line = datasets[i]
            pred = self.predict(line['source_body'])
            if i < 10:
                examples.append([i, line['source_body'], line['target_body'], pred[0]])
            pred = self.tokenizer.encode(pred[0])
            label = line['labels']
            eval_preds[0].append(pred)
            eval_preds[1].append(label)
        result = self.metric.compute_metrics(self.config.metric)(eval_preds)
        print('-' * 100 + f"\n{len(examples)} conversations：")
        for example in examples:
            print(f"The {example[0] + 1} conversation:")
            print(f"Context: {example[1]}")
            print(f"Response: {example[2]}")
            print(f"Prediction: {example[3]}\n")
        if self.config.metric == 'all':
            print('-' * 100 + f'\nScores:')
            for strategy, result in result.items():
                print(f'\n{strategy} Score:')
                for key, value in result.items():
                    print(f'{key}: {value}')
            print('-' * 100)
        else:
            print('-' * 100 + f'\nModel Scores:\n\n{self.config.metric} Scores:')
            for key, value in result.items():
                print(f'{key}: {value}')
            print('-' * 100)


if __name__ == '__main__':
    from dialogpt_evaluator import DialogptEvaluator
    config = Config('de')
    config.metric = 'all'
    evaluator = DialogptEvaluator("microsoft/DialoGPT-large", config)
    print(evaluator.predict('Does money buy happiness?'))
    evaluator.evaluate(small=10)