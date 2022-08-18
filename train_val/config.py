
raw_data_path = '../data/cleaned_data'


split = [85, 93]
seed = 66


max_input_length = 50
max_target_length = 50
prefix = ''
suffix = ''
map_batch_size = 1000


all_strategy = [
    ['greedy'],
    ['beam_search',
     {'num_beams': 5, 'no_repeat_ngram_size': 2, 'num_return_sequences': 1}
     ],
    ['sample',
     {'temperature': 0.8, 'top_k': 100, 'top_p': 0.7, 'num_return_sequences': 1}]
]
predict_strategy = all_strategy[2]


metric = 'sacrebleu'


class Config(object):
    def __init__(self, lang: str, batch_size=16, strategy='greedy'):
        self.path = {
            'raw_data_path': raw_data_path,
            # 'save_path': save_path
        }
        self.lang = lang
        self.split = split
        self.seed = seed
        self.token_parameters = {
            'max_input_length': max_input_length,
            'max_target_length': max_target_length,
            'prefix': prefix,
            'suffix': suffix,
            'batch_size': map_batch_size
        }
        self.args_dict = {
            "output_dir": f'../saved_model/mt5/{self.lang}',
            "save_strategy": 'epoch',
            "evaluation_strategy": 'epoch',
            "seed": seed,
            "per_device_train_batch_size": batch_size,
            "weight_decay": 0.01,
            "adafactor": True,
            "logging_steps": 50,
            "num_train_epochs": 15,
            "gradient_accumulation_steps": 2,
            'predict_with_generate': True,
            'generation_max_length': 200,
            'load_best_model_at_end': True,
            'learning_rate': 1e-5,
            'warmup_steps': 624
        }
        if strategy == 'greedy':
            self.predict_strategy = all_strategy[0]
        elif strategy == 'beam_search':
            self.predict_strategy = all_strategy[1]
        elif strategy == 'sample':
            self.predict_strategy = all_strategy[2]
        else:
            raise Exception('Please input correct strategy')
        self.metric = metric